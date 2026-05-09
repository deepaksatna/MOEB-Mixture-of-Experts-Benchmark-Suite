#!/usr/bin/env bash
# Track 4 resume — TP=2 rate {10,25,50}, then TP=4 full sweep, then TP=8 full sweep.
# Skip TP=6 (32 attention heads not divisible by 6).
set -uo pipefail
RES=/home/opc/moeb/results/a100_40gb_8x
LOG=/home/opc/moeb/logs
TOK=/home/opc/moeb/models/Qwen3-30B-A3B
DATA=/home/opc/moeb/data/ShareGPT_V3_unfiltered_cleaned_split.json
mkdir -p "$RES" "$LOG"; cd /home/opc/moeb
export PATH=/home/opc/moeb/.venv/bin:$PATH

start_serve(){
  local TP=$1
  local DEV
  case "$TP" in
    2) DEV=0,1 ;;
    4) DEV=0,1,2,3 ;;
    8) DEV=0,1,2,3,4,5,6,7 ;;
  esac
  pkill -f "vllm serve" 2>/dev/null; sleep 5
  echo "[serve] TP=$TP on $DEV"
  CUDA_VISIBLE_DEVICES=$DEV nohup .venv/bin/vllm serve "$TOK" \
    --served-model-name qwen3-30b-a3b \
    --tensor-parallel-size $TP --dtype bfloat16 \
    --max-model-len 32768 --gpu-memory-utilization 0.90 \
    --port 8000 --disable-log-requests \
    > $LOG/serve-resume-tp${TP}.log 2>&1 &
  echo $! > $LOG/serve.pid
  for i in $(seq 1 90); do
    sleep 10
    curl -s -m 2 http://localhost:8000/v1/models >/dev/null 2>&1 && { echo "[serve] READY TP=$TP after ${i}0s"; return 0; }
    kill -0 $(cat $LOG/serve.pid) 2>/dev/null || { echo "[serve] DIED TP=$TP"; tail -25 $LOG/serve-resume-tp${TP}.log; return 1; }
  done
  echo "[serve] TIMEOUT TP=$TP"; return 1
}
stop_serve(){ pkill -f "vllm serve" 2>/dev/null; sleep 5; }

run_one(){
  local TP=$1 RATE=$2 N=$3
  echo "=== TP=$TP rate=$RATE n=$N ==="
  .venv/bin/vllm bench serve \
    --backend vllm --model qwen3-30b-a3b --tokenizer "$TOK" \
    --base-url http://localhost:8000 --endpoint /v1/completions \
    --dataset-name sharegpt --dataset-path "$DATA" \
    --num-prompts $N --request-rate $RATE \
    --percentile-metrics "ttft,tpot,itl,e2el" --metric-percentiles "50,90,95,99" \
    --save-result --result-dir "$RES" \
    --result-filename "track4_qwen3_30b_tp${TP}_rate${RATE}.json" \
    > "$LOG/track4_tp${TP}_rate${RATE}.log" 2>&1
  tail -25 "$LOG/track4_tp${TP}_rate${RATE}.log"
  sleep 3
}

stop_serve

# TP=2 — only run rates 10,25,50 (rate 1 and 5 already captured)
echo "===== TP=2 (resume rates 10,25,50) ====="
if start_serve 2; then
  run_one 2 10 500
  run_one 2 25 500
  run_one 2 50 1500
fi
stop_serve

# TP=4 full sweep
rm -f $RES/track4_qwen3_30b_tp4_SKIPPED.json
echo "===== TP=4 ====="
if start_serve 4; then
  run_one 4 1 200
  run_one 4 5 400
  run_one 4 10 500
  run_one 4 25 500
  run_one 4 50 1500
else
  echo "{\"tp\":4,\"status\":\"skipped\"}" > $RES/track4_qwen3_30b_tp4_SKIPPED.json
fi
stop_serve

# TP=8 full sweep
rm -f $RES/track4_qwen3_30b_tp8_SKIPPED.json
echo "===== TP=8 ====="
if start_serve 8; then
  run_one 8 1 200
  run_one 8 5 400
  run_one 8 10 500
  run_one 8 25 500
  run_one 8 50 1500
else
  echo "{\"tp\":8,\"status\":\"skipped\"}" > $RES/track4_qwen3_30b_tp8_SKIPPED.json
fi
stop_serve

echo "===== TRACK 4 RESUME COMPLETE ====="
