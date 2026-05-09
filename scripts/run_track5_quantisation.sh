#!/usr/bin/env bash
# Track 5 — quantisation impact (BF16 vs FP8 vs AWQ INT4) on Qwen3-30B-A3B
set -uo pipefail
RES=/home/opc/moeb/results/a100_40gb_8x
LOG=/home/opc/moeb/logs
DATA=/home/opc/moeb/data/ShareGPT_V3_unfiltered_cleaned_split.json
mkdir -p "$RES" "$LOG"; cd /home/opc/moeb
export PATH=/home/opc/moeb/.venv/bin:$PATH

# variants: name model_path quant_arg dtype tp
VARIANTS=(
  "bf16 /home/opc/moeb/models/Qwen3-30B-A3B - bfloat16 8"
  "fp8 /home/opc/moeb/models/Qwen3-30B-A3B-FP8 - auto 8"
  "awq /home/opc/moeb/models/Qwen3-30B-A3B-AWQ awq_marlin auto 4"
)

start_serve_variant(){
  local name=$1 mod=$2 quant=$3 dt=$4 tp=$5
  pkill -f "vllm serve" 2>/dev/null; sleep 5
  local QFLAG=""
  [ "$quant" != "-" ] && QFLAG="--quantization $quant"
  local DEV=$(seq -s, 0 $((tp-1)))
  echo "[serve] track5 variant=$name tp=$tp on $DEV  quant=$quant"
  CUDA_VISIBLE_DEVICES=$DEV nohup .venv/bin/vllm serve "$mod" \
    --served-model-name qwen3-30b-${name} \
    --tensor-parallel-size $tp --dtype $dt $QFLAG \
    --max-model-len 16384 --gpu-memory-utilization 0.90 \
    --port 8000 --disable-log-requests \
    > $LOG/serve-track5-${name}.log 2>&1 &
  echo $! > $LOG/serve.pid
  for i in $(seq 1 90); do
    sleep 10
    curl -s -m 2 http://localhost:8000/v1/models >/dev/null 2>&1 && { echo "[serve] READY $name"; return 0; }
    kill -0 $(cat $LOG/serve.pid) 2>/dev/null || { echo "[serve] DIED $name"; tail -25 $LOG/serve-track5-${name}.log; return 1; }
  done
  echo "[serve] TIMEOUT $name"; return 1
}
stop_serve(){ pkill -f "vllm serve" 2>/dev/null; sleep 5; }

run_sweep(){
  local name=$1 mod=$2
  for RATE in 1 5 10 25; do
    local N=400; [ "$RATE" = "1" ] && N=200
    echo "=== track5 $name rate=$RATE ==="
    .venv/bin/vllm bench serve \
      --backend vllm --model qwen3-30b-${name} --tokenizer "$mod" \
      --base-url http://localhost:8000 --endpoint /v1/completions \
      --dataset-name sharegpt --dataset-path "$DATA" \
      --num-prompts $N --request-rate $RATE \
      --percentile-metrics "ttft,tpot,itl,e2el" --metric-percentiles "50,90,95,99" \
      --save-result --result-dir "$RES" \
      --result-filename "track5_qwen3_30b_${name}_rate${RATE}.json" \
      2>&1 | tail -25 | tee -a "$LOG/track5_${name}_rate${RATE}.log"
    sleep 3
  done
}

stop_serve
for V in "${VARIANTS[@]}"; do
  read NAME MOD QUANT DT TP <<< "$V"
  echo "===== TRACK 5 — $NAME ====="
  if [ "$QUANT" = "fp8" ] && [ ! -d "$MOD" ]; then
    echo "AWQ checkpoint not found at $MOD — skipping"
    echo "{\"variant\":\"$NAME\",\"status\":\"skipped\",\"reason\":\"awq checkpoint missing\"}" > $RES/track5_${NAME}_SKIPPED.json
    continue
  fi
  if start_serve_variant "$NAME" "$MOD" "$QUANT" "$DT" "$TP"; then
    run_sweep "$NAME" "$MOD"
  else
    echo "{\"variant\":\"$NAME\",\"status\":\"skipped\",\"reason\":\"serve failed\"}" > $RES/track5_${NAME}_SKIPPED.json
  fi
  stop_serve
done
echo "===== TRACK 5 COMPLETE ====="
