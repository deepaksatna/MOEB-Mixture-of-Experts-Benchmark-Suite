#!/usr/bin/env bash
# Track 1 — cross-model: Mixtral-8x7B at TP=2,4,8 BF16
set -uo pipefail
RES=/home/opc/moeb/results/a100_40gb_8x
LOG=/home/opc/moeb/logs
MOD=/home/opc/moeb/models/Mixtral-8x7B-Instruct-v0.1
TOK=$MOD
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
  echo "[serve] mixtral TP=$TP on $DEV"
  CUDA_VISIBLE_DEVICES=$DEV nohup .venv/bin/vllm serve "$MOD" \
    --served-model-name mixtral-8x7b \
    --tensor-parallel-size $TP --dtype bfloat16 \
    --max-model-len 16384 --gpu-memory-utilization 0.92 \
    --port 8000 --disable-log-requests \
    > $LOG/serve-mixtral-tp${TP}.log 2>&1 &
  echo $! > $LOG/serve.pid
  for i in $(seq 1 90); do
    sleep 10
    if curl -s -m 2 http://localhost:8000/v1/models >/dev/null 2>&1; then
      echo "[serve] READY mixtral TP=$TP"; return 0; fi
    if ! kill -0 $(cat $LOG/serve.pid) 2>/dev/null; then
      echo "[serve] DIED mixtral TP=$TP"; tail -25 $LOG/serve-mixtral-tp${TP}.log; return 1; fi
  done
  echo "[serve] TIMEOUT mixtral TP=$TP"; return 1
}
stop_serve(){ pkill -f "vllm serve" 2>/dev/null; sleep 5; }

run_sweep(){
  local TP=$1
  for RATE in 1 5 10 25; do
    local N=400; [ "$RATE" = "1" ] && N=200; [ "$RATE" = "25" ] && N=800
    echo "=== mixtral TP=$TP rate=$RATE n=$N ==="
    .venv/bin/vllm bench serve \
      --backend vllm --model mixtral-8x7b --tokenizer "$TOK" \
      --base-url http://localhost:8000 --endpoint /v1/completions \
      --dataset-name sharegpt --dataset-path "$DATA" \
      --num-prompts "$N" --request-rate "$RATE" \
      --percentile-metrics "ttft,tpot,itl,e2el" --metric-percentiles "50,90,95,99" \
      --save-result --result-dir "$RES" \
      --result-filename "track1_mixtral_8x7b_tp${TP}_rate${RATE}.json" \
      2>&1 | tail -25 | tee -a "$LOG/track1_mixtral_tp${TP}_rate${RATE}.log"
    sleep 3
  done
}

stop_serve
for TP in 2 4 8; do
  echo "===== TRACK 1 MIXTRAL TP=$TP ====="
  if start_serve $TP; then run_sweep $TP; else
    echo "{\"model\":\"mixtral-8x7b\",\"tp\":$TP,\"status\":\"skipped\"}" > $RES/track1_mixtral_tp${TP}_SKIPPED.json
  fi
  stop_serve
done
echo "===== TRACK 1 MIXTRAL COMPLETE ====="
