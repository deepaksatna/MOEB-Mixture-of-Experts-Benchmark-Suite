"""Walk the results directory, aggregate every vllm bench JSON into a single
flat CSV plus a Markdown summary. Keys we care about:
  date, model, backend, tp, request_rate, num_prompts, duration_s,
  request_throughput, output_throughput, total_token_throughput,
  mean_ttft_ms, median_ttft_ms, p99_ttft_ms,
  mean_tpot_ms, median_tpot_ms, p99_tpot_ms,
  mean_itl_ms, median_itl_ms, p99_itl_ms,
  mean_e2el_ms, median_e2el_ms, p99_e2el_ms.
"""
import json, csv, os, re, sys
from pathlib import Path

# Default to the repo-relative path; override with MOEB_RES env var when
# pointing at a different result tree (e.g. another node).
_DEFAULT_RES = Path(__file__).resolve().parents[2] / "results" / "a100_40gb_8x"
RESULTS = Path(os.environ.get("MOEB_RES", str(_DEFAULT_RES)))
OUT_CSV = RESULTS.parent / "moeb_summary.csv"
OUT_MD  = RESULTS.parent / "moeb_summary.md"

def parse_filename(p: Path):
    """Extract track/model/tp/rate/ctx from filename."""
    s = p.stem
    out = {"file": p.name, "track": "", "model": "", "tp": "", "rate": "", "ctx": "", "variant": ""}
    if s.startswith("baseline_vllm_"):
        out["track"] = "baseline"
        out["model"] = s.replace("baseline_vllm_", "")
    m = re.match(r"track(\d+)_(.+?)(?:_tp(\d+))?(?:_rate(\d+))?(?:_ctx([^_]+))?$", s)
    if m:
        out["track"] = "T" + m.group(1)
        out["model"] = m.group(2)
        out["tp"] = m.group(3) or ""
        out["rate"] = m.group(4) or ""
        out["ctx"] = m.group(5) or ""
    return out

ROWS = []
for p in sorted(RESULTS.glob("*.json")):
    if "_SKIPPED" in p.name: continue
    try:
        d = json.loads(p.read_text())
    except Exception as e:
        print(f"skip {p.name}: {e}", file=sys.stderr); continue
    meta = parse_filename(p)
    row = {**meta,
           "date": d.get("date", ""),
           "backend": d.get("backend", ""),
           "duration_s": d.get("duration", ""),
           "completed": d.get("completed", ""),
           "request_throughput": d.get("request_throughput", ""),
           "output_throughput": d.get("output_throughput", ""),
           "total_token_throughput": d.get("total_token_throughput", ""),
           "mean_ttft_ms": d.get("mean_ttft_ms", ""),
           "median_ttft_ms": d.get("median_ttft_ms", ""),
           "p99_ttft_ms": d.get("p99_ttft_ms", ""),
           "mean_tpot_ms": d.get("mean_tpot_ms", ""),
           "median_tpot_ms": d.get("median_tpot_ms", ""),
           "p99_tpot_ms": d.get("p99_tpot_ms", ""),
           "mean_itl_ms": d.get("mean_itl_ms", ""),
           "median_itl_ms": d.get("median_itl_ms", ""),
           "p99_itl_ms": d.get("p99_itl_ms", ""),
           "mean_e2el_ms": d.get("mean_e2el_ms", ""),
           "median_e2el_ms": d.get("median_e2el_ms", ""),
           "p99_e2el_ms": d.get("p99_e2el_ms", ""),
    }
    ROWS.append(row)

if not ROWS:
    print("no results yet"); sys.exit(0)

cols = list(ROWS[0].keys())
with OUT_CSV.open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
    for r in ROWS: w.writerow(r)
print(f"wrote {OUT_CSV}  rows={len(ROWS)}")

# Markdown summary, sorted by track > model > tp > rate
def num(x):
    try: return float(x)
    except: return -1
ROWS.sort(key=lambda r: (r["track"], r["model"], num(r["tp"]), num(r["rate"]), r["ctx"]))

md = []
md.append("# MOEB results summary")
md.append("")
md.append(f"Source: `{RESULTS}`  ({len(ROWS)} runs)")
md.append("")
md.append("| Track | Model | TP | Rate | Ctx | Reqs/s | Out tok/s | Total tok/s | TTFT p50 | TTFT p99 | TPOT p50 | E2EL p99 |")
md.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
for r in ROWS:
    fmt = lambda v: f"{float(v):.2f}" if v not in ("", None) else "—"
    md.append(f"| {r['track']} | {r['model']} | {r['tp']} | {r['rate']} | {r['ctx']} | "
              f"{fmt(r['request_throughput'])} | {fmt(r['output_throughput'])} | "
              f"{fmt(r['total_token_throughput'])} | {fmt(r['median_ttft_ms'])} | "
              f"{fmt(r['p99_ttft_ms'])} | {fmt(r['median_tpot_ms'])} | {fmt(r['p99_e2el_ms'])} |")
OUT_MD.write_text("\n".join(md))
print(f"wrote {OUT_MD}")
