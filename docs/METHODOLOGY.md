# Methodology

This document describes how MOEB v1 was executed. The intent is that
any team with an 8× A100 40 GB node (or comparable) can reproduce the
51 result rows from a clean install.

## 1. Hardware fingerprint

`results/a100_40gb_8x/node_fingerprint.txt` is captured automatically
at run start. v1 ran on:

| Field | Value |
|---|---|
| Host OS | Oracle Linux Server 9.7 · kernel 5.15.0-319.201.4.4 |
| CPU | 2× AMD EPYC 7542 (64 cores / 128 threads · 8 NUMA nodes) |
| RAM | 2.0 TiB |
| GPU | 8× NVIDIA A100-SXM4-40 GB · driver 590.48.01 · CUDA cap 8.0 |
| NVLink | NV12 mesh, 25 GB/s × 12 lanes per GPU = 300 GB/s aggregate |
| Inference | vLLM 0.10.0 · torch 2.7.1+cu128 · CUDA 12.8 |

## 2. Workload

| Field | Value |
|---|---|
| Prompt source | ShareGPT V3 unfiltered cleaned split (~95 K conversations, 672 MB) |
| Driver | `vllm bench serve` — emits TTFT, TPOT, ITL, e2eL with p50/p90/p95/p99 |
| Arrival | Poisson at fixed `--request-rate`, no client-side max-concurrency cap |
| Tokeniser | Local copy of the model's `tokenizer.json` (no remote calls) |

## 3. Sweep design

| Track | Sweep |
|---|---|
| **T4** TP scaling | TP ∈ {2, 4, 8} × rate ∈ {1, 5, 10, 25, 50} prompts/s |
| **T1** cross-model | model ∈ {Qwen3-30B-A3B, Mixtral 8×7B, Llama-4-Scout, Llama-4-Maverick} |
| **T5** quantisation | dtype ∈ {BF16 TP=2, GPTQ-Int4 TP=2, BF16 TP=8 reference} |
| **T6** long context | max_model_len ∈ {4K, 8K, 16K, 32K, 64K, 128K}, input ∈ {1K, 4K, 8K, 16K} |
| **T2** expert routing | 24 ShareGPT prompts × ≤128 generated tokens, hooks on every gate |

Per rate run, prompt count is scaled (200 at rate=1, up to 1500 at
rate=50) so each run completes in 30–90 s of wall time.

## 4. What was honestly skipped, and why

When a configuration genuinely cannot run on the hardware, MOEB writes
`*_SKIPPED.json` with the reason rather than fabricating a number. v1
honestly skipped:

| Variant | File | Reason |
|---|---|---|
| Qwen3-30B-A3B FP8 | `track5_fp8_SKIPPED.json` | Block-scaled FP8 needs Hopper Cutlass kernels |
| Qwen3-30B-A3B AWQ | `track5_awq_SKIPPED.json` | No first-party AWQ checkpoint exists |
| Qwen3-30B-A3B GPTQ-Int4 TP={4, 8} | `track5_gptq_int4_tp4_SKIPPED.json`, `track5v2_gptq_int4_SKIPPED.json` | `moe_intermediate_size = 768`; 768/8 = 96 < Marlin BLOCK_SIZE_K |
| Qwen3-30B-A3B TP=6 | `track4_qwen3_30b_tp6_SKIPPED.json` | 32 attention heads not divisible by 6 |
| Mixtral 8×7B TP=2 | `track1_mixtral_tp2_SKIPPED.json` | OOM (47 GB / 2 ranks > 40 GB after KV) |
| Llama-4-Scout TP=2 | `track4_llama4_scout_tp2_SKIPPED.json` | OOM (109 GB / 2 ranks > 40 GB) |
| Llama 4 Track 2 routing | `track2_Llama4-Scout_SKIPPED.json` | `Llama4ForConditionalGeneration` is multimodal + uses `compressed-tensors` — needs custom router-extraction path |
| T6 ctx ≥ 64 K | `track6_qwen3_30b_ctx{64,128}k_SKIPPED.json` | Beyond model native 40 960 window — would need YaRN rope-scaling |

This is a deliberate design choice. Benchmarks lose credibility the
moment they fabricate numbers for configurations the hardware cannot
support. Every skipped row in MOEB has a known reason.

## 5. Reproducing the analysis from the included data

The repository ships with all 51 raw bench JSONs plus the two Track 2
routing histograms. To regenerate the CSV, Markdown summary, and PNG
plot pack from this data:

```bash
python3 scripts/analysis/aggregate_results.py
python3 scripts/analysis/make_plots.py
```

Output:

```
results/moeb_summary.csv         (51 rows)
results/moeb_summary.md
results/plots/fig{1..10}.png
```

## 6. Reproducing on a new GPU node

### 6.1 Environment setup (on the GPU node)

```bash
# Use a virtualenv. Python 3.11 strongly recommended.
uv venv .venv && source .venv/bin/activate
uv pip install \
  "vllm==0.10.0" \
  "torch==2.7.1+cu128" "torchvision==0.22.1+cu128" "torchaudio==2.7.1+cu128" \
  "transformers==4.55.0" \
  "accelerate>=1.13.0" \
  "ninja" \
  --extra-index-url https://download.pytorch.org/whl/cu128

# Hugging Face login (the gated Llama 4 weights need it)
export HF_TOKEN=<your-hf-token>
```

### 6.2 Model download

Place models under `~/moeb/models/`:

| Model | Repo | Disk |
|---|---|---|
| Qwen3-30B-A3B | `Qwen/Qwen3-30B-A3B` | 57 GB BF16 |
| Qwen3-30B-A3B GPTQ-Int4 | `Qwen/Qwen3-30B-A3B-GPTQ-Int4` | 16 GB |
| Mixtral 8×7B | `mistralai/Mixtral-8x7B-Instruct-v0.1` | 87 GB BF16 (drop the legacy `consolidated.*.pt` after pull — saves 87 GB) |
| Llama-4-Scout FP8 | `RedHatAI/Llama-4-Scout-17B-16E-Instruct-FP8-dynamic` | 107 GB |
| Llama-4-Maverick INT4 | `RedHatAI/Llama-4-Maverick-17B-128E-Instruct-quantized.w4a16` | 200 GB |

`hf download <repo> --local-dir ~/moeb/models/<name>` is the canonical
command.

### 6.3 Data download

```bash
mkdir -p ~/moeb/data && cd ~/moeb/data
wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json
```

### 6.4 Running

Each driver is independent. Run the ones you need:

```bash
bash scripts/run_track4_resume.sh         # T4: TP={2,4,8} × 5 rates  (~75 min)
bash scripts/run_track1_mixtral.sh        # T1: Mixtral TP={4,8}      (~25 min)
bash scripts/run_llama4_scout.sh          # T1: Scout TP={4,8}        (~30 min)
bash scripts/run_llama4_maverick.sh       # T1: Maverick INT4 TP=8    (~25 min)
bash scripts/run_track5_v2.sh             # T5: GPTQ-Int4 TP=2        (~15 min)
bash scripts/run_track6_long_context.sh   # T6: ctx={4,8,16,32,64,128}K (~40 min)
python3 scripts/tracks/track2_expert_utilisation.py    # T2: routing  (~15 min/model)
```

Or run the orchestrator: `bash scripts/run_master_pipeline.sh`.

### 6.5 Pulling results to your laptop

```bash
# On your laptop:
MOEB_KEY=~/.ssh/your_key MOEB_HOST=user@gpu-node \
  bash scripts/sync_from_vm.sh
```

This rsyncs the result tree, regenerates `moeb_summary.{csv,md}`, and
rebuilds the 10 PNGs locally.

## 7. Common pitfalls (lessons from v1)

These bit us during v1 — recording them so they don't bite future
runs:

| Pitfall | Symptom | Fix |
|---|---|---|
| `pip install sglang[all]` upgrades torch | vLLM `_C.abi3.so: undefined symbol` | Install SGLang in a separate venv (`.venv-sglang`); never share with vLLM |
| `deep_gemm` left behind after torch downgrade | `deep_gemm_cpp.abi3.so: undefined symbol` | `rm -rf .venv/lib/.../deep_gemm*` — `uv pip uninstall` is unreliable here |
| `ninja` not on PATH inside vLLM workers | `FileNotFoundError: ninja` from worker subprocess | `export PATH=$VENV/bin:$PATH` at the top of every driver |
| Wait-loop self-match | `pgrep -f "hf download Foo"` matches itself when the wait-loop bash command contains "Foo" | Use file-existence checks (`while [ ! -f $MOD/config.json ]`) |
| Block-scaled FP8 on A100 | `CutlassBlockScaledGroupedGemm not supported on the current platform` | A100 needs per-tensor FP8 (e.g. `RedHatAI/*-FP8-dynamic`) — block-scaled is Hopper-only |
| Marlin `BLOCK_SIZE_K` divisibility | `RuntimeError: size_k must divisible by BLOCK_SIZE_K` | `moe_intermediate_size / TP` must be ≥ 128. For Qwen3-30B-A3B, TP=2 works; TP=4/8 fails |
| Llama 4 Track 2 hangs on load | 99 % CPU, 0 % GPU, 6+ minutes, no output | Llama 4 is `Llama4ForConditionalGeneration` (multimodal); the generic gate-hook script written for Qwen/Mixtral cannot be loaded that way |

## 8. Citing

Use the run fingerprint from the included `node_fingerprint.txt`:

```
MOEB v1 — 8× NVIDIA A100 40 GB SXM4 · vLLM 0.10.0 · ShareGPT V3
Captured 2026-05-09 · 51 result rows · 4 MoE families · 6 tracks
```
