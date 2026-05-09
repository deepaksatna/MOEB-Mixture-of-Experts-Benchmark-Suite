#!/usr/bin/env bash
# MOEB master pipeline — runs Track 5, Track 6, Track 1 Mixtral, then Track 2.
# Designed to launch the moment Track 4 reports complete.
set -uo pipefail
cd /home/opc/moeb
export PATH=/home/opc/moeb/.venv/bin:$PATH
LOG=/home/opc/moeb/logs

# Wait for track4 to finish if it is still running
echo "[master] waiting on track4 ..."
until grep -q "TRACK 4 RESUME COMPLETE" "$LOG/track4-resume.log" 2>/dev/null; do sleep 30; done
echo "[master] track4 done — pipeline starting at $(date -u +%FT%TZ)"

# Track 5 — quantisation. Skip the bf16 variant (we already have track4 TP=8 BF16).
# Only fp8 will run; awq stays skipped.
{
  echo "===== MASTER: TRACK 5 ====="
  bash /home/opc/moeb/run_track5_quantisation.sh
} 2>&1 | tee "$LOG/master-track5.log"

# Track 6 — long context. Re-uses Qwen3-30B-A3B at TP=8 BF16, varies ctx.
{
  echo "===== MASTER: TRACK 6 ====="
  bash /home/opc/moeb/run_track6_long_context.sh
} 2>&1 | tee "$LOG/master-track6.log"

# Track 1 Mixtral — different model family. TP=2,4,8 at rates 1,5,10,25.
{
  echo "===== MASTER: TRACK 1 MIXTRAL ====="
  bash /home/opc/moeb/run_track1_mixtral.sh
} 2>&1 | tee "$LOG/master-track1mix.log"

# Track 2 — expert utilisation, runs after vLLM is freed. Uses HF transformers
# directly. Run for both Qwen3-30B-A3B and Mixtral.
pkill -f "vllm serve" 2>/dev/null; sleep 5
{
  echo "===== MASTER: TRACK 2 (Qwen3-30B-A3B) ====="
  CUDA_VISIBLE_DEVICES=0,1,2,3 MOEB_MODEL=/home/opc/moeb/models/Qwen3-30B-A3B \
    MOEB_N=24 MOEB_MAX_NEW=128 \
    /home/opc/moeb/.venv/bin/python /home/opc/moeb/tracks/track2_expert_utilisation.py
  echo "===== MASTER: TRACK 2 (Mixtral 8x7B) ====="
  CUDA_VISIBLE_DEVICES=4,5,6,7 MOEB_MODEL=/home/opc/moeb/models/Mixtral-8x7B-Instruct-v0.1 \
    MOEB_N=24 MOEB_MAX_NEW=128 \
    /home/opc/moeb/.venv/bin/python /home/opc/moeb/tracks/track2_expert_utilisation.py
} 2>&1 | tee "$LOG/master-track2.log"

# Final aggregate
/home/opc/moeb/.venv/bin/python /home/opc/moeb/analysis/aggregate_results.py | tee "$LOG/master-aggregate.log"

echo "===== MASTER PIPELINE COMPLETE ====="
