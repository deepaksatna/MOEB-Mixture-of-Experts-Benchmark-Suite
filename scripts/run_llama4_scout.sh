#!/usr/bin/env bash
# MOEB — Llama-4-Scout pipeline. v2 — uses file-existence check, no pgrep self-match.
set -uo pipefail
cd /home/opc/moeb
export PATH=/home/opc/moeb/.venv/bin:$PATH
LOG=/home/opc/moeb/logs
RES=/home/opc/moeb/results/a100_40gb_8x
DATA=/home/opc/moeb/data/ShareGPT_V3_unfiltered_cleaned_split.json
MOD=/home/opc/moeb/models/Llama-4-Scout-17B-FP8-dynamic

if [ ! -f "$MOD/config.json" ]; then
  echo "[llama4] checkpoint missing"; exit 1
fi
echo "[llama4] checkpoint present at $MOD ($(du -sh $MOD | cut -f1))"

start_serve(){
  local TP=$1
  local DEV
  case "$TP" in
    4) DEV=0,1,2,3 ;;
    8) DEV=0,1,2,3,4,5,6,7 ;;
  esac
  pkill -f "vllm serve" 2>/dev/null; sleep 5
  echo "[serve] llama4-scout TP=$TP on $DEV"
  CUDA_VISIBLE_DEVICES=$DEV nohup .venv/bin/vllm serve "$MOD" \
    --served-model-name llama4-scout \
    --tensor-parallel-size $TP \
    --max-model-len 32768 --gpu-memory-utilization 0.92 \
    --port 8000 --disable-log-requests \
    > $LOG/serve-llama4-scout-tp${TP}.log 2>&1 &
  echo $! > $LOG/serve.pid
  for i in $(seq 1 120); do
    sleep 10
    curl -s -m 2 http://localhost:8000/v1/models >/dev/null 2>&1 && { echo "[serve] READY llama4-scout TP=$TP after ${i}0s"; return 0; }
    kill -0 $(cat $LOG/serve.pid) 2>/dev/null || { echo "[serve] DIED llama4-scout TP=$TP"; tail -25 $LOG/serve-llama4-scout-tp${TP}.log; return 1; }
  done
  echo "[serve] TIMEOUT llama4-scout TP=$TP"; return 1
}
stop_serve(){ pkill -f "vllm serve" 2>/dev/null; sleep 5; }

run_sweep(){
  local TP=$1
  for RATE in 1 5 10 25 50; do
    local N=400; [ "$RATE" = "1" ] && N=200; [ "$RATE" = "50" ] && N=1500
    echo "=== llama4-scout TP=$TP rate=$RATE n=$N ==="
    .venv/bin/vllm bench serve --backend vllm \
      --model llama4-scout --tokenizer "$MOD" \
      --base-url http://localhost:8000 --endpoint /v1/completions \
      --dataset-name sharegpt --dataset-path "$DATA" \
      --num-prompts $N --request-rate $RATE \
      --percentile-metrics "ttft,tpot,itl,e2el" --metric-percentiles "50,90,95,99" \
      --save-result --result-dir "$RES" \
      --result-filename "track4_llama4_scout_tp${TP}_rate${RATE}.json" \
      > "$LOG/llama4_scout_tp${TP}_rate${RATE}.log" 2>&1
    tail -25 "$LOG/llama4_scout_tp${TP}_rate${RATE}.log"
    sleep 3
  done
}

stop_serve
echo "{\"model\":\"llama4-scout\",\"tp\":2,\"status\":\"skipped\",\"reason\":\"weights 109GB / 2 > 40GB per A100\"}" \
  > $RES/track4_llama4_scout_tp2_SKIPPED.json

for TP in 4 8; do
  echo "===== LLAMA4 — TP=$TP ====="
  if start_serve $TP; then
    run_sweep $TP
  else
    echo "{\"model\":\"llama4-scout\",\"tp\":$TP,\"status\":\"skipped\",\"reason\":\"vllm serve failed\"}" > $RES/track4_llama4_scout_tp${TP}_SKIPPED.json
  fi
  stop_serve
done

# Track 2 expert utilisation
pkill -f "vllm serve" 2>/dev/null; sleep 5
echo "===== LLAMA4 — TRACK 2 expert utilisation ====="
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 MOEB_MODEL=$MOD \
  MOEB_N=24 MOEB_MAX_NEW=128 \
  /home/opc/moeb/.venv/bin/python /home/opc/moeb/tracks/track2_expert_utilisation.py \
  > $LOG/llama4_track2.log 2>&1 || echo "[t2 llama4] failed — see $LOG/llama4_track2.log"

/home/opc/moeb/.venv/bin/python /home/opc/moeb/analysis/aggregate_results.py | tail -3
echo "===== LLAMA4 PIPELINE COMPLETE ====="
