"""Generate the canonical MOEB plot pack.

All plots:
- 1080p (dpi 130), large readable fonts
- Saved into results/plots/ as fig{N}_{name}.png
- Self-contained: no auto-aggregation; reads moeb_summary.csv + track2 JSONs

Plot pack
=========
fig1_tp_scaling          — TP scaling, total throughput vs request rate
fig2_ttft_under_load     — TTFT p50/p99 vs request rate, by TP (grid)
fig3_model_compare       — 4-way model comparison at TP=8
fig4_pareto              — latency vs throughput Pareto (Qwen across TPs)
fig5_quant_compare       — GPTQ-Int4 vs BF16 (same-TP, apples-to-apples)
fig6_long_context        — TTFT and throughput vs context length
fig7_routing_gini        — per-layer Gini, Mixtral vs Qwen3
fig8_routing_histogram   — expert activation histogram for representative layers
fig9_dead_experts        — dead-expert count per layer
fig10_arch_overview      — architecture × deployment matrix summary
"""
from __future__ import annotations
from pathlib import Path
import csv
import json
import math
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

# ─── paths ──────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
CSV_PATH = ROOT / "results" / "moeb_summary.csv"
T2_DIR   = ROOT / "results" / "a100_40gb_8x" / "track2_expert_util"
OUT      = ROOT / "results" / "plots"
OUT.mkdir(exist_ok=True)

# ─── palette ────────────────────────────────────────────────────────
ORACLE_RED   = "#C74634"   # warm red — primary series
DARK_NAVY    = "#17365D"   # navy     — secondary series
ACCENT_GREEN = "#52A569"
ACCENT_AMBER = "#F1B73C"
ACCENT_VIOLET= "#8862B7"
GREY         = "#4A4A4A"
LIGHT_GRID   = "#DCDCDC"
PALETTE_TP    = {"2": ORACLE_RED, "4": DARK_NAVY, "8": ACCENT_GREEN}
PALETTE_MODEL = {
    "qwen3_30b":     ORACLE_RED,
    "mixtral_8x7b":  DARK_NAVY,
    "llama4_scout":  ACCENT_AMBER,
    "llama4_maverick": ACCENT_GREEN,
}
PALETTE_QUANT = {"bf16": DARK_NAVY, "gptq_int4": ORACLE_RED, "fp8": ACCENT_AMBER}

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.titlecolor": DARK_NAVY,
    "axes.labelsize": 10.5,
    "axes.labelcolor": GREY,
    "xtick.color": GREY,
    "ytick.color": GREY,
    "axes.edgecolor": GREY,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
    "legend.fontsize": 10,
})


def _f(x):
    try: return float(x)
    except: return math.nan


def _style(ax, title=None, xlabel=None, ylabel=None):
    if title is not None: ax.set_title(title, pad=10)
    if xlabel is not None: ax.set_xlabel(xlabel)
    if ylabel is not None: ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", linestyle=":", color=LIGHT_GRID, alpha=0.9, zorder=0)
    ax.set_axisbelow(True)


def _save(fig, name, layout=True):
    path = OUT / name
    if layout:
        fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path.name}")


def load_csv():
    if not CSV_PATH.exists():
        print(f"missing {CSV_PATH}", file=sys.stderr); sys.exit(1)
    with CSV_PATH.open() as f:
        return list(csv.DictReader(f))


# ─── fig1 — TP scaling ───────────────────────────────────────────────
def fig1_tp_scaling(rows):
    sub = [r for r in rows if r["track"] == "T4" and r["model"] == "qwen3_30b"]
    fig, ax = plt.subplots(figsize=(9, 5.2), dpi=130)
    for tp in ("2", "4", "8"):
        pts = sorted([(int(r["rate"]), _f(r["total_token_throughput"])) for r in sub if r["tp"] == tp])
        if not pts: continue
        xs, ys = zip(*pts)
        ax.plot(xs, ys, marker="o", linewidth=2.6, markersize=8,
                color=PALETTE_TP[tp], label=f"TP={tp}", zorder=3)
        # annotate peak
        peak_x, peak_y = max(pts, key=lambda p: p[1])
        ax.annotate(f"{peak_y:.0f}", xy=(peak_x, peak_y), xytext=(0, 8),
                    textcoords="offset points", fontsize=9, color=PALETTE_TP[tp],
                    weight="bold", ha="center")
    _style(ax,
           "Tensor-parallel scaling — Qwen3-30B-A3B BF16 on 8× A100 40 GB",
           "Concurrent request rate (req/s, Poisson)",
           "Total token throughput (tok/s)")
    ax.legend(loc="upper left", title="Parallelism", title_fontsize=10)
    ax.text(0.99, 0.02, "Higher = better · vLLM 0.10.0 · ShareGPT V3",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=8, color=GREY, style="italic")
    _save(fig, "fig1_tp_scaling.png")


# ─── fig2 — TTFT under load (p50 + p99 grid) ─────────────────────────
def fig2_ttft_under_load(rows):
    sub = [r for r in rows if r["track"] == "T4" and r["model"] == "qwen3_30b"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=130, sharey=True)
    for tp in ("2", "4", "8"):
        pts = sorted([(int(r["rate"]),
                       _f(r["median_ttft_ms"]),
                       _f(r["p99_ttft_ms"])) for r in sub if r["tp"] == tp])
        if not pts: continue
        xs, p50, p99 = zip(*pts)
        ax1.plot(xs, p50, marker="o", linewidth=2.4, markersize=7, color=PALETTE_TP[tp], label=f"TP={tp}")
        ax2.plot(xs, p99, marker="s", linewidth=2.4, markersize=7, color=PALETTE_TP[tp], label=f"TP={tp}")
    _style(ax1, "TTFT median (p50)", "Request rate (req/s)", "Time-to-first-token (ms, log scale)")
    _style(ax2, "TTFT tail (p99)", "Request rate (req/s)", None)
    ax1.set_yscale("log"); ax2.set_yscale("log")
    ax1.legend(loc="upper left"); ax2.legend(loc="upper left")
    fig.suptitle("First-token latency under concurrency — Qwen3-30B-A3B BF16",
                 color=DARK_NAVY, fontsize=14, weight="bold", y=1.02)
    fig.text(0.5, -0.02,
             "Lower = better · TTFT spike at saturation marks the queue knee",
             ha="center", fontsize=9, color=GREY, style="italic")
    _save(fig, "fig2_ttft_under_load.png")


# ─── fig3 — 4-way cross-model at TP=8 ────────────────────────────────
def fig3_model_compare(rows):
    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=130)
    series = [
        ("qwen3_30b",       "T4", "Qwen3-30B-A3B  BF16   (3 B active / 30 B total · 128 experts × top-8)",  PALETTE_MODEL["qwen3_30b"], "o"),
        ("mixtral_8x7b",    "T1", "Mixtral 8×7B  BF16    (13 B active / 47 B total · 8 experts × top-2)",   PALETTE_MODEL["mixtral_8x7b"], "s"),
        ("llama4_scout",    "T4", "Llama-4-Scout  FP8    (17 B active / 109 B total · 16 experts × top-1)", PALETTE_MODEL["llama4_scout"], "^"),
        ("llama4_maverick", "T4", "Llama-4-Maverick INT4 (17 B active / 400 B total · 128 experts × top-1)",PALETTE_MODEL["llama4_maverick"], "D"),
    ]
    for model, track, label, colour, marker in series:
        pts = sorted([(int(r["rate"]), _f(r["total_token_throughput"]))
                      for r in rows if r["track"] == track and r["model"] == model and r["tp"] == "8"])
        if not pts: continue
        xs, ys = zip(*pts)
        ax.plot(xs, ys, marker=marker, linewidth=2.4, markersize=8,
                color=colour, label=label, zorder=3)
    _style(ax,
           "Cross-model throughput — TP=8 on 8× A100 40 GB",
           "Concurrent request rate (req/s, Poisson)",
           "Total token throughput (tok/s)")
    ax.legend(loc="upper left", fontsize=9)
    ax.text(0.99, 0.02,
            "Throughput tracks ACTIVE-parameter count, not total — smaller active wins per-token",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=8, color=GREY, style="italic")
    _save(fig, "fig3_model_compare.png")


# ─── fig4 — Pareto: latency vs throughput ────────────────────────────
def fig4_pareto(rows):
    """Each TP traces a curve through (TTFT-p50, request_throughput)."""
    sub = [r for r in rows if r["track"] == "T4" and r["model"] == "qwen3_30b"]
    fig, ax = plt.subplots(figsize=(9, 5.2), dpi=130)
    for tp in ("2", "4", "8"):
        pts = sorted([(_f(r["request_throughput"]), _f(r["median_ttft_ms"]), int(r["rate"]))
                      for r in sub if r["tp"] == tp], key=lambda p: p[2])
        if not pts: continue
        thr = [p[0] for p in pts]
        ttft = [p[1] for p in pts]
        rates = [p[2] for p in pts]
        ax.plot(thr, ttft, marker="o", linewidth=2.4, markersize=8,
                color=PALETTE_TP[tp], label=f"TP={tp}", zorder=3)
        for x, y, r in zip(thr, ttft, rates):
            ax.annotate(f"r{r}", xy=(x, y), xytext=(6, 4), textcoords="offset points",
                        fontsize=8, color=PALETTE_TP[tp])
    _style(ax,
           "Latency–throughput Pareto — Qwen3-30B-A3B BF16",
           "Request throughput (req/s)",
           "TTFT median (ms, log scale)")
    ax.set_yscale("log")
    # SLA bands
    ax.axhspan(0, 100, alpha=0.06, color=ACCENT_GREEN, zorder=1)
    ax.axhspan(100, 500, alpha=0.06, color=ACCENT_AMBER, zorder=1)
    ax.axhspan(500, 100000, alpha=0.06, color=ORACLE_RED, zorder=1)
    ax.text(0.01, 0.04, "interactive (<100 ms)", transform=ax.transAxes, fontsize=8, color=ACCENT_GREEN, weight="bold")
    ax.text(0.01, 0.36, "responsive (100–500 ms)", transform=ax.transAxes, fontsize=8, color="#A0741A", weight="bold")
    ax.text(0.01, 0.78, "saturated (>500 ms)", transform=ax.transAxes, fontsize=8, color=ORACLE_RED, weight="bold")
    ax.legend(loc="upper right", title="Parallelism")
    _save(fig, "fig4_pareto.png")


# ─── fig5 — Quantisation, same TP ───────────────────────────────────
def fig5_quant_compare(rows):
    bf16_2 = sorted([(int(r["rate"]), _f(r["total_token_throughput"]))
                     for r in rows if r["track"] == "T4" and r["model"] == "qwen3_30b" and r["tp"] == "2"])
    gptq_2 = sorted([(int(r["rate"]), _f(r["total_token_throughput"]))
                     for r in rows if r["track"] == "T5" and r["model"] == "qwen3_30b_gptq_int4"])
    bf16_8 = sorted([(int(r["rate"]), _f(r["total_token_throughput"]))
                     for r in rows if r["track"] == "T4" and r["model"] == "qwen3_30b" and r["tp"] == "8"])
    if not (bf16_2 and gptq_2): return
    fig, ax = plt.subplots(figsize=(9.5, 5.5), dpi=130)
    ax.plot(*zip(*bf16_2), marker="o", linewidth=2.6, markersize=8,
            color=PALETTE_QUANT["bf16"], label="BF16  TP=2  (~57 GB weights · 2 GPUs)")
    ax.plot(*zip(*gptq_2), marker="s", linewidth=2.6, markersize=8,
            color=PALETTE_QUANT["gptq_int4"], label="GPTQ-Int4  TP=2  (~16 GB weights · 2 GPUs)")
    if bf16_8:
        ax.plot(*zip(*bf16_8), marker="^", linewidth=2.0, markersize=7,
                color=ACCENT_GREEN, label="BF16  TP=8  (reference, 8 GPUs)",
                linestyle="--", alpha=0.85)
    _style(ax,
           "Quantisation — apples-to-apples at TP=2, plus TP=8 reference",
           "Concurrent request rate (req/s, Poisson)",
           "Total token throughput (tok/s)")
    ax.legend(loc="upper left", fontsize=9.5)
    ax.text(0.99, 0.02,
            "Same hardware budget: GPTQ-Int4 wins by +20-30% via larger KV cache headroom",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=8, color=GREY, style="italic")
    _save(fig, "fig5_quant_compare.png")


# ─── fig6 — Long-context envelope ────────────────────────────────────
def fig6_long_context(rows):
    sub = [r for r in rows if r["track"] == "T6"]
    if not sub: return
    order = {"1k": 1024, "4k": 4096, "8k": 8192, "16k": 16384, "32k": 32768}
    pts = sorted([(order[r["ctx"]],
                   _f(r["median_ttft_ms"]),
                   _f(r["request_throughput"]),
                   _f(r["total_token_throughput"]),
                   r["ctx"])
                  for r in sub if r["ctx"] in order])
    if not pts: return
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=130)
    xs = [p[0] for p in pts]
    labels = [p[4] for p in pts]
    ax1.plot(xs, [p[1] for p in pts], marker="o", linewidth=2.6, markersize=8, color=ORACLE_RED, zorder=3)
    ax2.plot(xs, [p[3] for p in pts], marker="o", linewidth=2.6, markersize=8, color=DARK_NAVY, zorder=3)
    _style(ax1, "TTFT vs context length", "Input tokens (log)", "TTFT median (ms, log)")
    _style(ax2, "Token throughput vs context length", "Input tokens (log)", "Total tok/s")
    for ax in (ax1, ax2):
        ax.set_xscale("log", base=2)
        ax.set_xticks(xs); ax.set_xticklabels(labels)
    ax1.set_yscale("log")
    fig.suptitle("Long-context envelope — Qwen3-30B-A3B BF16 (TP=8)",
                 color=DARK_NAVY, fontsize=14, weight="bold", y=1.02)
    fig.text(0.5, -0.02,
             "Native window 40 960 — beyond requires YaRN scaling (≥64K skipped)",
             ha="center", fontsize=9, color=GREY, style="italic")
    _save(fig, "fig6_long_context.png")


# ─── Track 2 helpers ─────────────────────────────────────────────────
def load_t2():
    out = {}
    for f in T2_DIR.glob("*.json"):
        if "_SKIPPED" in f.name: continue
        d = json.loads(f.read_text())
        if "Mixtral" in f.name:
            out["mixtral"] = d
        elif "Qwen3" in f.name:
            out["qwen"] = d
    return out


# ─── fig7 — per-layer Gini ───────────────────────────────────────────
def fig7_routing_gini(t2):
    if not t2: return
    fig, ax = plt.subplots(figsize=(10, 5.2), dpi=130)
    for key, label, colour, marker in [
        ("mixtral", f"Mixtral 8×7B  ({len(t2.get('mixtral',{}).get('layers',{}))} layers · 8 experts × top-2)",
            DARK_NAVY, "s"),
        ("qwen",    f"Qwen3-30B-A3B ({len(t2.get('qwen',{}).get('layers',{}))} layers · 128 experts × top-8)",
            ORACLE_RED, "o"),
    ]:
        if key not in t2: continue
        d = t2[key]
        layers = sorted(int(k) for k in d["layers"])
        gini = [d["layers"][str(li)]["gini"] for li in layers]
        ax.plot(layers, gini, marker=marker, linewidth=2.0, markersize=6,
                color=colour, label=label, zorder=3, alpha=0.92)
        ax.axhline(d["summary"]["mean_gini"], linestyle=":", color=colour, alpha=0.6, linewidth=1.4)
    _style(ax,
           "Expert-routing imbalance (Gini coefficient) by layer",
           "Decoder layer index",
           "Gini coefficient (0 = uniform, 1 = single-expert monopoly)")
    ax.legend(loc="upper left", fontsize=9.5)
    ax.set_ylim(0, 0.6)
    ax.text(0.99, 0.02,
            "Lower is healthier (uniform routing). Dashed line = mean across layers.",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=8, color=GREY, style="italic")
    _save(fig, "fig7_routing_gini.png")


# ─── fig8 — expert activation histogram ──────────────────────────────
def fig8_routing_histogram(t2):
    if not t2: return
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), dpi=130)
    for r, (key, family, label_short, colour) in enumerate([
        ("qwen",    "Qwen3-30B-A3B",  "128 experts × top-8", ORACLE_RED),
        ("mixtral", "Mixtral 8×7B",   "8 experts × top-2",   DARK_NAVY),
    ]):
        if key not in t2: continue
        d = t2[key]
        n_layers = len(d["layers"])
        picks = [0, n_layers // 2, n_layers - 1]
        for c, li in enumerate(picks):
            ax = axes[r, c]
            counts = d["layers"][str(li)]["expert_counts"]
            total = sum(counts) or 1
            pct = [100 * v / total for v in counts]
            ax.bar(range(len(counts)), pct, color=colour, width=1.0, edgecolor="none", zorder=3)
            uniform = 100 / len(counts) * d["top_k"]
            ax.axhline(uniform, color=GREY, linestyle="--", linewidth=1.2, alpha=0.7)
            ax.set_title(f"{family} · layer {li}",
                         fontsize=11.5, color=colour, pad=8, weight="bold")
            ax.set_xlabel("Expert index", fontsize=10)
            if c == 0:
                ax.set_ylabel("Activation share\n(% of top-k slots)", fontsize=10)
            ax.grid(True, axis="y", linestyle=":", color=LIGHT_GRID, alpha=0.9, zorder=0)
            ax.set_axisbelow(True)
            ax.set_ylim(0, max(pct) * 1.22)
            ax.text(0.99, 0.97,
                    f"Gini {d['layers'][str(li)]['gini']:.3f}\n"
                    f"Dead {d['layers'][str(li)]['dead_experts']}",
                    transform=ax.transAxes, ha="right", va="top",
                    fontsize=9.5, color=GREY,
                    bbox=dict(facecolor="white", edgecolor=LIGHT_GRID, boxstyle="round,pad=0.34"))
    fig.suptitle("Expert-activation histograms — representative layers (token=ShareGPT, top:Qwen3 / bot:Mixtral)",
                 color=DARK_NAVY, fontsize=13.5, weight="bold")
    fig.text(0.5, 0.005,
             "Dashed line = uniform-routing baseline (top_k / n_experts × 100). "
             "Tall single bars = routing collapse onto a few experts.",
             ha="center", fontsize=9, color=GREY, style="italic")
    fig.subplots_adjust(top=0.92, bottom=0.08, hspace=0.48, wspace=0.28)
    _save(fig, "fig8_routing_histogram.png", layout=False)


# ─── fig9 — dead experts per layer ───────────────────────────────────
def fig9_dead_experts(t2):
    if not t2: return
    fig, ax = plt.subplots(figsize=(11, 5.4), dpi=130)
    qwen_layers, qwen_pct = [], []
    mix_layers, mix_pct = [], []
    if "qwen" in t2:
        d = t2["qwen"]
        for li in sorted(int(k) for k in d["layers"]):
            qwen_layers.append(li)
            qwen_pct.append(100 * d["layers"][str(li)]["dead_experts"] / d["n_experts_per_layer"])
    if "mixtral" in t2:
        d = t2["mixtral"]
        for li in sorted(int(k) for k in d["layers"]):
            mix_layers.append(li)
            mix_pct.append(100 * d["layers"][str(li)]["dead_experts"] / d["n_experts_per_layer"])
    if qwen_pct:
        ax.bar(qwen_layers, qwen_pct, width=0.85, color=ORACLE_RED,
               label=f"Qwen3-30B-A3B  (128 experts/layer · {sum(t2['qwen']['layers'][str(li)]['dead_experts'] for li in qwen_layers)} total dead)",
               zorder=3)
    if mix_pct:
        # plot at slight x offset so the zero baseline is visible
        ax.scatter(mix_layers, mix_pct, marker="s", color=DARK_NAVY, s=44, zorder=4,
                   label=f"Mixtral 8×7B  (8 experts/layer · {sum(t2['mixtral']['layers'][str(li)]['dead_experts'] for li in mix_layers)} total dead)")
    _style(ax,
           "Dead experts per layer — never selected on benchmark prompts",
           "Decoder layer index",
           "Dead experts (% of experts in that layer)")
    ax.set_ylim(-0.5, max([10] + qwen_pct + mix_pct) * 1.15)
    ax.legend(loc="upper left", fontsize=9.5)
    ax.text(0.5, -0.16,
            "Sample: 24 ShareGPT V3 prompts × ≤128 generated tokens. "
            "Mixtral remains at 0 % across all 32 layers; "
            "Qwen3 collapse concentrates in the deepest layers.",
            transform=ax.transAxes, ha="center", va="top",
            fontsize=9, color=GREY, style="italic")
    fig.subplots_adjust(bottom=0.18)
    _save(fig, "fig9_dead_experts.png", layout=False)


# ─── fig10 — architecture × deployment overview ──────────────────────
def fig10_arch_overview(rows):
    """Bar chart: peak total throughput at TP=8 by model, with active params labels."""
    series = []
    for model, track, label, active, total, colour in [
        ("qwen3_30b",       "T4", "Qwen3-30B-A3B\nBF16",  3,  30,  PALETTE_MODEL["qwen3_30b"]),
        ("mixtral_8x7b",    "T1", "Mixtral 8×7B\nBF16",   13, 47,  PALETTE_MODEL["mixtral_8x7b"]),
        ("llama4_scout",    "T4", "Llama-4-Scout\nFP8",   17, 109, PALETTE_MODEL["llama4_scout"]),
        ("llama4_maverick", "T4", "Llama-4-Maverick\nINT4", 17, 400, PALETTE_MODEL["llama4_maverick"]),
    ]:
        pts = [_f(r["total_token_throughput"])
               for r in rows if r["track"] == track and r["model"] == model and r["tp"] == "8"]
        if not pts: continue
        peak = max(pts)
        series.append((label, peak, active, total, colour))

    if not series: return
    # Single-line model labels: "Qwen3-30B-A3B (BF16)\n3 B active · 30 B total"
    # use newlines inside the tick label string and set linespacing
    labels = [
        f"{s[0].replace(chr(10), ' (')+'.'}".replace("\n", " (").replace(".)", ")") +
        f"\n{s[2]} B active  ·  {s[3]} B total"
        for s in series
    ]
    # cleaner: compose explicitly
    labels = []
    for s in series:
        # s[0] is e.g. "Qwen3-30B-A3B\nBF16" — flatten + put dtype in parens
        first, _, dtype = s[0].partition("\n")
        labels.append(f"{first}\n({dtype})\n{s[2]} B active  ·  {s[3]} B total")

    fig, ax = plt.subplots(figsize=(12, 6.6), dpi=130)
    xs = np.arange(len(series))
    ys = [s[1] for s in series]
    colours = [s[4] for s in series]
    bars = ax.bar(xs, ys, color=colours, width=0.58, zorder=3, edgecolor="white", linewidth=2)
    for b, s in zip(bars, series):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + max(ys) * 0.022,
                f"{s[1]:.0f} tok/s", ha="center", va="bottom",
                fontsize=11.5, weight="bold", color=DARK_NAVY)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=10, linespacing=1.55)
    ax.tick_params(axis="x", which="both", length=0, pad=10)
    ax.set_ylim(0, max(ys) * 1.22)
    ax.set_title("Peak total throughput at TP=8 — MoE model fleet on 8× A100 40 GB",
                 pad=14)
    ax.set_ylabel("Total token throughput (tok/s, peak across rate sweep)")
    ax.grid(True, axis="y", linestyle=":", color=LIGHT_GRID, alpha=0.9, zorder=0)
    ax.set_axisbelow(True)
    ax.text(0.99, 0.97,
            "ShareGPT V3  ·  vLLM 0.10.0  ·  BF16 / FP8 / INT4 as feasible on A100",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=9, color=GREY, style="italic")
    fig.subplots_adjust(bottom=0.22)
    _save(fig, "fig10_arch_overview.png", layout=False)


# ─── main ────────────────────────────────────────────────────────────
def main():
    rows = load_csv()
    t2 = load_t2()
    fig1_tp_scaling(rows)
    fig2_ttft_under_load(rows)
    fig3_model_compare(rows)
    fig4_pareto(rows)
    fig5_quant_compare(rows)
    fig6_long_context(rows)
    fig7_routing_gini(t2)
    fig8_routing_histogram(t2)
    fig9_dead_experts(t2)
    fig10_arch_overview(rows)
    print("done")


if __name__ == "__main__":
    main()
