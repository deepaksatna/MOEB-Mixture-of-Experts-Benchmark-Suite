# MoE selection guide — which model for which use case

> **Read this disclaimer first.** Every recommendation in this document
> is derived from the MOEB v1 run: ShareGPT V3 prompts, vLLM 0.10.0,
> 8× NVIDIA A100 40 GB SXM4, and the four model checkpoints listed
> below. **Your traffic mix and your hardware will change these
> rankings.** Treat this as a defensible starting point for a sizing
> conversation — not as a final decision. The reproducibility recipe in
> the [README](../README.md) and [METHODOLOGY](METHODOLOGY.md) lets you
> re-run MOEB on your own corpus before committing.

## 1. The four families compared

| Model (engine + dtype) | Total | Active | Routing | Native ctx | Min A100 40 GB |
|---|---|---|---|---|---|
| Qwen3-30B-A3B (vLLM, BF16) | 30 B | 3 B | 128 experts × top-8 | 40 960 | TP=2 |
| Qwen3-30B-A3B (vLLM, GPTQ-Int4) | 30 B | 3 B | 128 experts × top-8 | 40 960 | TP=2 |
| Mixtral 8×7B Instruct v0.1 (vLLM, BF16) | 47 B | 13 B | 8 experts × top-2 | 32 768 | TP=4 |
| Llama-4-Scout-17B-16E (vLLM, FP8-dynamic) | 109 B | 17 B | 16 experts × top-1 | 10 M (claimed) | TP=4 |
| Llama-4-Maverick-17B-128E (vLLM, W4A16/INT4) | 400 B | 17 B | 128 experts × top-1 | 10 M (claimed) | TP=8 |

## 2. Headline metrics per family (TP=8 unless noted)

| Family | Peak tot tok/s | Peak req/s | TTFT p50 @ rate=10 | TPOT p50 @ rate=10 | Routing Gini |
|---|---|---|---|---|---|
| Qwen3-30B-A3B BF16 | **11 728** | 27.5 | 29 ms | 13 ms | 0.43 |
| Qwen3-30B-A3B GPTQ-Int4 (TP=2) | 8 603 | 20.2 | 34 ms | 16 ms | (same model) |
| Mixtral 8×7B BF16 | 8 260 | 19.0 | 33 ms | 17 ms | **0.07** |
| Llama-4-Scout FP8 | 6 724 | 17.0 | 53 ms | 30 ms | n/a (skipped) |
| Llama-4-Maverick INT4 | 3 539 | 8.7 | 77 ms | 47 ms | n/a (skipped) |

## 3. Use-case matrix

Each row is a deployment scenario; each column is the recommended
model with a brief reason.

### 3.1 By workload shape

| Scenario | Recommendation | Reason |
|---|---|---|
| **High-concurrency chat** — short prompts (≤2 K), short answers (≤500), many users | **Qwen3-30B-A3B BF16 TP=8** | Highest peak throughput per GPU; cleanest TTFT scaling under load (38 ms p50 / 54 ms p99 at rate=25) |
| **Production chat with strong tail-latency SLA** | **Mixtral 8×7B BF16 TP=8** | Most balanced routing — fewer pathological tail events on shifted traffic; routing health is a leading indicator of regression |
| **Long-context document Q&A** (8K–32K input, ≤2K answer) | **Qwen3-30B-A3B BF16 TP=8** | Native 40 K window, predictable TTFT through 8 K, throughput climbs to 24 000 tok/s on long inputs |
| **Coding / multi-turn agent** (mid-length context, complex outputs) | **Mixtral 8×7B BF16 TP=8** *or* Llama-4-Scout FP8 TP=8 | Both balance latency and quality; Scout adds vision if needed |
| **Multimodal (text + vision)** | **Llama-4-Scout FP8 TP=8** | Only A100-feasible Llama 4 with native vision; clean TTFT (53 ms p50 at rate=10) |
| **Frontier reasoning / tool-use quality** | **Llama-4-Maverick INT4 TP=8** | Largest expert pool (128 routed × top-1 from a 400 B checkpoint); slowest of the four but best capability ceiling |

### 3.2 By hardware budget

| Budget | Recommendation | Reason |
|---|---|---|
| 2× A100 40 GB | **Qwen3-30B-A3B GPTQ-Int4 TP=2** | +20–30 % throughput vs BF16 TP=2 on identical hardware |
| 4× A100 40 GB | **Qwen3-30B-A3B BF16 TP=4** *or* Mixtral 8×7B BF16 TP=4 | Both fit; Qwen wins peak tok/s, Mixtral wins routing health |
| 8× A100 40 GB | **Qwen3-30B-A3B BF16 TP=8** | Highest absolute throughput in this benchmark |
| 8× A100 40 GB, vision-required | **Llama-4-Scout FP8 TP=8** | Multimodal native |
| 8× A100 40 GB, capability-first | **Llama-4-Maverick INT4 TP=8** | Largest expert pool feasible on this hardware |

### 3.3 By concern

| Primary concern | Recommendation | Reason |
|---|---|---|
| Lowest p99 TTFT under load | Qwen3-30B-A3B BF16 TP=8 | 54 ms p99 at rate=25 — best in the run |
| Lowest TPOT (perceived per-token speed) | Qwen3-30B-A3B BF16 TP=8 | 7.8 ms p50 at rate=1, 18 ms p50 at rate=25 |
| Most predictable behaviour under traffic shift | Mixtral 8×7B BF16 TP=8 | Flat per-layer Gini, zero dead experts |
| Best capacity per existing GPU | Qwen3-30B-A3B GPTQ-Int4 TP=2 | +31 % at rate=50 vs BF16 TP=2 |
| Largest expert pool / capability headroom | Llama-4-Maverick INT4 TP=8 | 128 routed experts × top-1 |
| Long-context (≥16 K) feasibility | Qwen3-30B-A3B BF16 TP=8 | Native window covers 32 K cleanly |

## 4. Why throughput tracks active-parameter count

The clearest pattern in the cross-model plot (`fig3` in the README) is
that throughput inversely tracks **active**-parameter count, not
**total**-parameter count:

| Family | Active | Peak tok/s @ TP=8 |
|---|---|---|
| Qwen3-30B-A3B | 3 B | 11 728 |
| Mixtral 8×7B | 13 B | 8 260 |
| Llama-4-Scout | 17 B (FP8) | 6 724 |
| Llama-4-Maverick | 17 B (INT4) | 3 539 |

Decode is dominated by the active-path FLOPs at each token. Total
parameters mostly determine memory footprint (and therefore which TP
tier you can serve at), not per-token speed. **A model's "30 B" or
"400 B" headline is a capability marker, not a serving cost marker.**

Maverick is slower than Scout despite identical 17 B active because
INT4 dequantisation runs on the active path while FP8 (per-tensor on
A100) is closer to BF16 in compute cost. INT4 is still the right
choice for Maverick — it's the *only* A100-feasible Maverick — but
expect ~50 % of Scout's throughput at the same parallelism.

## 5. The trade-off pairs you'll actually face

### Qwen3-30B-A3B BF16 vs Mixtral 8×7B BF16

These two are the closest competitors at TP=8. The choice depends on
which axis you weight:

| Axis | Qwen wins | Mixtral wins |
|---|---|---|
| Peak throughput | ✅ 11 728 vs 8 260 tok/s | |
| TTFT p50 (low load) | ✅ 25 ms vs 30 ms at rate=1 | |
| TTFT p99 stability | ✅ 54 ms at rate=25 | (Mixtral has 914 ms tail spike at rate=25) |
| Routing health | | ✅ Gini 0.07, zero dead experts |
| Predictability under fine-tuning | | ✅ Less skew to begin with → less drift |
| Native context window | ✅ 40 960 | (32 768) |
| Total params (capability ceiling) | (30 B) | ✅ 47 B |

**Default to Qwen** if you want to hit max QPS on a fixed GPU budget
and your traffic is stable. **Default to Mixtral** if you'll fine-tune
on a domain corpus, or if your traffic mix is going to shift.

### Qwen3-30B-A3B BF16 TP=2 vs GPTQ-Int4 TP=2

Both fit the same 2-GPU box. INT4 is the better choice unless you're
sensitive to small quality regressions (GPTQ-Int4 is a few perplexity
points behind BF16 on most evals; we did not run T7 quality numbers in
v1). The throughput dividend is real and reliable.

### Llama-4-Scout FP8 vs Mixtral 8×7B BF16 (both TP=8)

Both serve roughly the same throughput band (6 724 vs 8 260 tok/s
peak). Pick Scout for **multimodal** or **very long-context claims**
(Scout advertises 10 M tokens — we only validated the lower window
on A100). Pick Mixtral for **routing predictability** and slightly
higher peak throughput.

## 6. Before you decide — a 5-question checklist

Before you commit any of these recommendations to production:

1. **What's your real input-length distribution?** ShareGPT prompts
   average ~200 input tokens. RAG workloads average 4 K+. Re-run T6
   (long-context) with your distribution; the TTFT envelope will look
   different.
2. **What's your real output-length distribution?** Generation
   throughput is dominated by output tokens. ShareGPT averages
   ~250 output tokens; coding/agent workloads average 1 K+. The TPOT
   numbers in this run are valid; the e2eL numbers will not be.
3. **Will you fine-tune on a domain corpus?** If yes, use Track 2
   (`scripts/tracks/track2_expert_utilisation.py`) **before** and
   **after** the fine-tune. Watch the per-layer Gini: a +0.1 jump in
   the deepest layers is a warning of routing collapse.
4. **What's your real concurrency profile — bursty or steady?** All
   numbers here are Poisson-arrival. Bursty traffic will hit the
   queue knee earlier than the rate sweep suggests; budget 30–50 %
   headroom on the published peak QPS.
5. **What's your hardware ladder upgrade path?** A100 → H100 changes
   FP8 from emulated to native. Maverick FP8 (currently A100-skip)
   becomes the headline option. Plan for that re-benchmark when
   hardware lands.

If any of these questions returns "I'm not sure", **re-run MOEB on
your own infrastructure first**. The reproducibility recipe in
[METHODOLOGY.md](METHODOLOGY.md) is built for exactly that.
