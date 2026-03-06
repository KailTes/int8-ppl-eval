#!/bin/bash
# ============================================================
# run_pangu.sh 参考版本
#
# 环境变量:
#   MODEL_PATH — 模型路径 (FP16 或 INT8 均可，INT8 由 quantization_config 自动处理)
# ============================================================

source /usr/local/Ascend/ascend-toolkit/set_env.sh 2>/dev/null || true

MODEL_PATH="${MODEL_PATH:-/data/weights/pangu_v2/92B/iter_0059000_hf/}"
SERVE_PORT="${SERVE_PORT:-8000}"
TP_SIZE="${TP_SIZE:-8}"

vllm serve "${MODEL_PATH}" \
    --dtype bfloat16 \
    --tensor-parallel-size "${TP_SIZE}" \
    --enforce-eager \
    --served-model-name pangu \
    --trust-remote-code \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.9 \
    --host 0.0.0.0 \
    --port "${SERVE_PORT}"
