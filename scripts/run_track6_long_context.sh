#!/usr/bin/env bash
# Track 6 — long-context KV-cache sweep. Vary --max-model-len and use
# vllm bench serve --dataset-name random with --random-input-len.
set -uo pipefail
RES=/home/opc/moeb/results/a100_40gb_8x
LOG=/home/opc/moeb/logs
TOK=/home/opc/moeb/models/Qwen3-30B-A3B
mkdir -p "$RES" "$LOG"; cd /home/opc/moeb
export PATH=/home/opc/moeb/.venv/bin:$PATH

start_serve(){
  local CTX=$1
  pkill -f "vllm serve" 2>/dev/null; sleep 5
  echo "[serve] TP=8 ctx=$CTX"
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 nohup .venv/bin/vllm serve "$TOK" \
    --served-model-name qwen3-30b-a3b \
    --tensor-parallel-size 8 --dtype bfloat16 \
    --max-model-len $CTX --gpu-memory-utilization 0.92 \
    --port 8000 --disable-log-requests \
    > $LOG/serve-track6-ctx${CTX}.log 2>&1 &
  echo $! > $LOG/serve.pid
  for i in $(seq 1 90); do
    sleep 10
    if curl -s -m 2 http://localhost:8000/v1/models >/dev/null 2>&1; then
      echo "[serve] READY ctx=$CTX"; return 0; fi
    if ! kill -0 $(cat $LOG/serve.pid) 2>/dev/null; then
      echo "[serve] DIED ctx=$CTX"; tail -20 $LOG/serve-track6-ctx${CTX}.log; return 1; fi
  done
  echo "[serve] TIMEOUT ctx=$CTX"; return 1
}

stop_serve(){ pkill -f "vllm serve" 2>/dev/null; sleep 5; }

# Contexts and per-context input lengths. Output 256 tokens.
# Format: ctx in_len label
PAIRS=("4096 1024 1k" "8192 4096 4k" "16384 8192 8k" "32768 16384 16k" "65536 32768 32k" "131072 65536 64k")

for pair in "${PAIRS[@]}"; do
  read CTX INLEN LABEL <<< "$pair"
  echo "===== Track 6 — ctx=$CTX input=$INLEN label=$LABEL ====="
  if start_serve $CTX; then
    .venv/bin/vllm bench serve \
      --backend vllm --model qwen3-30b-a3b --tokenizer "$TOK" \
      --base-url http://localhost:8000 --endpoint /v1/completions \
      --dataset-name random --random-input-len $INLEN --random-output-len 256 \
      --num-prompts 64 --request-rate 4 \
      --percentile-metrics "ttft,tpot,itl,e2el" --metric-percentiles "50,90,95,99" \
      --save-result --result-dir "$RES" \
      --result-filename "track6_qwen3_30b_ctx${LABEL}.json" \
      2>&1 | tail -25 | tee -a "$LOG/track6_ctx${LABEL}.log"
    # Capture KV memory snapshot
    nvidia-smi --query-gpu=index,memory.used --format=csv > "$RES/track6_qwen3_30b_ctx${LABEL}_gpu_mem.csv"
  else
    echo "{\"ctx\": $CTX, \"status\": \"skipped\"}" > "$RES/track6_qwen3_30b_ctx${LABEL}_SKIPPED.json"
  fi
  stop_serve
done
echo "===== TRACK 6 COMPLETE ====="
