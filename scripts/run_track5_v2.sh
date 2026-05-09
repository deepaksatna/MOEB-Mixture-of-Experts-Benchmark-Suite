#!/usr/bin/env bash
# Track 5 v2 — A100-compatible quantisation
# Compares Qwen3-30B-A3B BF16 vs GPTQ-Int4 (W4A16) at TP=8.
# Skips block-scaled FP8 (Hopper-only) per Track 5 v1 outcome.
set -uo pipefail
cd /home/opc/moeb
export PATH=/home/opc/moeb/.venv/bin:$PATH
RES=/home/opc/moeb/results/a100_40gb_8x
LOG=/home/opc/moeb/logs
DATA=/home/opc/moeb/data/ShareGPT_V3_unfiltered_cleaned_split.json
mkdir -p "$RES" "$LOG"

# variants: name model_path quant_arg
VARIANTS=(
  "gptq_int4 /home/opc/moeb/models/Qwen3-30B-A3B-GPTQ-Int4 gptq_marlin"
)

start_serve_v2(){
  local name=$1 mod=$2 quant=$3
  pkill -f "vllm serve" 2>/dev/null; sleep 5
  echo "[serve] track5v2 variant=$name quant=$quant"
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 nohup .venv/bin/vllm serve "$mod" \
    --served-model-name qwen3-30b-${name} \
    --tensor-parallel-size 8 --quantization $quant \
    --max-model-len 16384 --gpu-memory-utilization 0.90 \
    --port 8000 --disable-log-requests \
    > $LOG/serve-track5v2-${name}.log 2>&1 &
  echo $! > $LOG/serve.pid
  for i in $(seq 1 90); do
    sleep 10
    curl -s -m 2 http://localhost:8000/v1/models >/dev/null 2>&1 && { echo "[serve] READY $name"; return 0; }
    kill -0 $(cat $LOG/serve.pid) 2>/dev/null || { echo "[serve] DIED $name"; tail -25 $LOG/serve-track5v2-${name}.log; return 1; }
  done
  echo "[serve] TIMEOUT $name"; return 1
}
stop_serve(){ pkill -f "vllm serve" 2>/dev/null; sleep 5; }

run_sweep(){
  local name=$1 mod=$2
  for RATE in 1 5 10 25 50; do
    local N=400; [ "$RATE" = "1" ] && N=200; [ "$RATE" = "50" ] && N=1500
    echo "=== track5v2 $name rate=$RATE ==="
    .venv/bin/vllm bench serve \
      --backend vllm --model qwen3-30b-${name} --tokenizer "$mod" \
      --base-url http://localhost:8000 --endpoint /v1/completions \
      --dataset-name sharegpt --dataset-path "$DATA" \
      --num-prompts $N --request-rate $RATE \
      --percentile-metrics "ttft,tpot,itl,e2el" --metric-percentiles "50,90,95,99" \
      --save-result --result-dir "$RES" \
      --result-filename "track5_qwen3_30b_${name}_rate${RATE}.json" \
      > "$LOG/track5v2_${name}_rate${RATE}.log" 2>&1
    tail -25 "$LOG/track5v2_${name}_rate${RATE}.log"
    sleep 3
  done
}

stop_serve
for V in "${VARIANTS[@]}"; do
  read NAME MOD QUANT <<< "$V"
  echo "===== TRACK 5 v2 — $NAME ====="
  if [ ! -d "$MOD" ]; then
    echo "{\"variant\":\"$NAME\",\"status\":\"skipped\",\"reason\":\"checkpoint missing\"}" > $RES/track5v2_${NAME}_SKIPPED.json
    continue
  fi
  if start_serve_v2 "$NAME" "$MOD" "$QUANT"; then
    run_sweep "$NAME" "$MOD"
  else
    echo "{\"variant\":\"$NAME\",\"status\":\"skipped\",\"reason\":\"serve failed\"}" > $RES/track5v2_${NAME}_SKIPPED.json
  fi
  stop_serve
done
echo "===== TRACK 5 v2 COMPLETE ====="
