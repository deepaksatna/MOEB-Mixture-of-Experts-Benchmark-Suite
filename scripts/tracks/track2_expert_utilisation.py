"""Track 2 — Expert utilisation analysis."""
import json, os, time
from collections import Counter
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = os.environ.get("MOEB_MODEL", "/home/opc/moeb/models/Qwen3-30B-A3B")
PROMPT_FILE = os.environ.get("MOEB_PROMPTS", "/home/opc/moeb/data/ShareGPT_V3_unfiltered_cleaned_split.json")
OUT = Path(os.environ.get("MOEB_OUT", "/home/opc/moeb/results/a100_40gb_8x/track2_expert_util"))
N_PROMPTS = int(os.environ.get("MOEB_N", "32"))
MAX_NEW = int(os.environ.get("MOEB_MAX_NEW", "128"))
DEVICE_MAP = os.environ.get("MOEB_DEVICE_MAP", "auto")
OUT.mkdir(parents=True, exist_ok=True)

def load_prompts(path, n):
    data = json.load(open(path))
    out = []
    for row in data:
        for c in row.get("conversations") or []:
            if c.get("from") == "human" and c.get("value"):
                out.append(c["value"]); break
        if len(out) >= n: break
    return out

def gini(x):
    x = np.asarray(x, dtype=np.float64)
    if x.sum() == 0: return 0.0
    x = np.sort(x); n = len(x)
    return float((2 * (np.arange(1, n+1) * x).sum() / (n * x.sum())) - (n+1)/n)

def kl_uniform(p):
    p = np.asarray(p, dtype=np.float64)
    if p.sum() == 0: return 0.0
    p = p / p.sum()
    u = 1.0 / len(p)
    m = p > 0
    return float(np.sum(p[m] * (np.log(p[m]) - np.log(u))))

print(f"[t2] loading {MODEL_PATH} ...", flush=True)
tok = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16, device_map=DEVICE_MAP, trust_remote_code=True)
model.eval()

counters = {}
n_experts, top_k = None, None
hooks = []

layers = model.model.layers
gates = []
for i, layer in enumerate(layers):
    # Qwen3-MoE uses layer.mlp; Mixtral uses layer.block_sparse_moe.
    mlp = getattr(layer, "mlp", None) or getattr(layer, "block_sparse_moe", None)
    if mlp is None:
        continue
    gate = getattr(mlp, "gate", None)
    if gate is not None and hasattr(mlp, "experts"):
        gates.append((i, gate, mlp))
print(f"[t2] found {len(gates)} MoE layers", flush=True)
if gates:
    sample = gates[0][2]
    n_experts = len(sample.experts)
    top_k = getattr(sample, "top_k", None) or getattr(model.config, "num_experts_per_tok", None)
    print(f"[t2] n_experts={n_experts}  top_k={top_k}", flush=True)

def make_hook(layer_idx, k):
    def hook(module, inp, out):
        logits = out if isinstance(out, torch.Tensor) else out[0]
        if logits.dim() == 3:
            logits = logits.reshape(-1, logits.size(-1))
        idx = torch.topk(logits, k, dim=-1).indices.detach().cpu().numpy()
        c = counters.setdefault(layer_idx, Counter())
        for row in idx:
            for e in row:
                c[int(e)] += 1
    return hook

for idx, gate, _ in gates:
    hooks.append(gate.register_forward_hook(make_hook(idx, top_k or 2)))

prompts = load_prompts(PROMPT_FILE, N_PROMPTS)
print(f"[t2] running {len(prompts)} prompts × max_new={MAX_NEW} ...", flush=True)
t0 = time.time()
with torch.inference_mode():
    for i, p in enumerate(prompts):
        ids = tok(p, return_tensors="pt", truncation=True, max_length=1024).to(model.device)
        _ = model.generate(**ids, max_new_tokens=MAX_NEW, do_sample=False, pad_token_id=tok.eos_token_id or 0)
        if (i+1) % 4 == 0:
            print(f"  [{i+1}/{len(prompts)}] elapsed={time.time()-t0:.1f}s", flush=True)
elapsed = time.time() - t0
print(f"[t2] done in {elapsed:.1f}s", flush=True)
for h in hooks: h.remove()

report = {"model": MODEL_PATH, "n_prompts": len(prompts), "max_new_tokens": MAX_NEW,
          "n_experts_per_layer": n_experts, "top_k": top_k,
          "wall_seconds": elapsed, "layers": {}}
all_g, all_k, dead_total, expert_total = [], [], 0, 0
for li, c in sorted(counters.items()):
    counts = [c.get(e, 0) for e in range(n_experts)]
    g = gini(counts); k = kl_uniform(counts); dead = sum(1 for v in counts if v == 0)
    report["layers"][str(li)] = {"total_tokens_routed": sum(counts), "expert_counts": counts,
                                  "gini": g, "kl_from_uniform": k, "dead_experts": dead}
    all_g.append(g); all_k.append(k); dead_total += dead; expert_total += n_experts

mg = float(np.mean(all_g)) if all_g else 0
mk = float(np.mean(all_k)) if all_k else 0
report["summary"] = {"mean_gini": mg, "mean_kl_from_uniform": mk,
                     "dead_experts_total": dead_total, "expert_slots_total": expert_total,
                     "dead_fraction": dead_total / expert_total if expert_total else 0}

out_file = OUT / f"track2_{Path(MODEL_PATH).name}.json"
out_file.write_text(json.dumps(report, indent=2))
print(f"[t2] wrote {out_file}", flush=True)
print(f"[t2] mean Gini={mg:.4f}  mean KL={mk:.4f}  dead={dead_total}/{expert_total}", flush=True)
