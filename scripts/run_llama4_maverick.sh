#!/usr/bin/env bash
# MOEB stretch goal — Llama-4-Maverick-INT4 (~200GB on 8× A100 40GB).
# Tight fit; only TP=8 has enough KV headroom.
set -uo pipefail
cd /home/opc/moeb
export PATH=/home/opc/moeb/.venv/bin:$PATH
LOG=/home/opc/moeb/logs
RES=/home/opc/moeb/results/a100_40gb_8x
DATA=/home/opc/moeb/data/ShareGPT_V3_unfiltered_cleaned_split.json
MOD=/home/opc/moeb/models/Llama-4-Maverick-w4a16

# Wait for Scout pipeline to fully complete
until grep -q "LLAMA4 PIPELINE COMPLETE" $LOG/llama4-driver.log 2>/dev/null; do sleep 30; done
echo "[maverick] scout pipeline done — proceeding"

# Free disk if needed: drop Mixtral now (we have all its results)
if [ -d /home/opc/moeb/models/Mixtral-8x7B-Instruct-v0.1 ]; then
  echo "[maverick] freeing 87 GB by removing Mixtral (all results captured)"
  rm -rf /home/opc/moeb/models/Mixtral-8x7B-Instruct-v0.1
fi

if [ ! -d "$MOD" ]; then
  echo "[maverick] downloading Maverick-w4a16 (~200 GB)"
  HF_TOKEN=${HF_TOKEN} \
    /home/opc/moeb/.venv/bin/hf download RedHatAI/Llama-4-Maverick-17B-128E-Instruct-quantized.w4a16 \
    --local-dir "$MOD" \
    --token ${HF_TOKEN} \
    > $LOG/dl-llama4-maverick.log 2>&1
fi

du -sh "$MOD"

start_serve(){
  pkill -f "vllm serve" 2>/dev/null; sleep 5
  echo "[serve] llama4-maverick TP=8"
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 nohup .venv/bin/vllm serve "$MOD" \
    --served-model-name llama4-maverick \
    --tensor-parallel-size 8 --quantization compressed-tensors \
    --max-model-len 16384 --gpu-memory-utilization 0.93 \
    --port 8000 --disable-log-requests \
    > $LOG/serve-llama4-maverick.log 2>&1 &
  echo $! > $LOG/serve.pid
  for i in $(seq 1 120); do
    sleep 10
    curl -s -m 2 http://localhost:8000/v1/models >/dev/null 2>&1 && { echo "[serve] READY maverick after ${i}0s"; return 0; }
    kill -0 $(cat $LOG/serve.pid) 2>/dev/null || { echo "[serve] DIED maverick"; tail -25 $LOG/serve-llama4-maverick.log; return 1; }
  done
  echo "[serve] TIMEOUT maverick"; return 1
}
stop_serve(){ pkill -f "vllm serve" 2>/dev/null; sleep 5; }

stop_serve
if start_serve; then
  for RATE in 1 5 10 25; do
    N=400; [ "$RATE" = "1" ] && N=200
    echo "=== llama4-maverick TP=8 rate=$RATE ==="
    .venv/bin/vllm bench serve --backend vllm \
      --model llama4-maverick --tokenizer "$MOD" \
      --base-url http://localhost:8000 --endpoint /v1/completions \
      --dataset-name sharegpt --dataset-path "$DATA" \
      --num-prompts $N --request-rate $RATE \
      --percentile-metrics "ttft,tpot,itl,e2el" --metric-percentiles "50,90,95,99" \
      --save-result --result-dir "$RES" \
      --result-filename "track4_llama4_maverick_tp8_rate${RATE}.json" \
      > "$LOG/llama4_maverick_rate${RATE}.log" 2>&1
    tail -25 "$LOG/llama4_maverick_rate${RATE}.log"
    sleep 3
  done
else
  echo "{\"model\":\"llama4-maverick\",\"tp\":8,\"status\":\"skipped\",\"reason\":\"vllm serve failed\"}" \
    > $RES/track4_llama4_maverick_tp8_SKIPPED.json
fi
stop_serve

/home/opc/moeb/.venv/bin/python /home/opc/moeb/analysis/aggregate_results.py | tail -3
echo "===== MAVERICK STRETCH COMPLETE ====="
