#!/bin/bash
# ============================================================
# run_pangu.sh 参考版本
#
# 支持通过环境变量 MODEL_PATH 传入模型路径 (FP16 或 INT8 均可)
#
# 用法:
#   MODEL_PATH=/data/models/pangu-RTN-W8A8 bash run_pangu.sh
#   bash run_pangu.sh   # 使用默认 FP16 路径
# ============================================================

source /usr/local/Ascend/ascend-toolkit/set_env.sh 2>/dev/null || true

MODEL_PATH="${MODEL_PATH:-/data/weights/pangu_v2/92B/iter_0059000_hf/}"
SERVE_PORT="${SERVE_PORT:-8000}"
TP_SIZE="${TP_SIZE:-8}"

# 自动检测是否为 INT8 量化模型 (config.json 中有 quantization_config)
IS_QUANTIZED=$(python3 -c "
import json, os
cfg = os.path.join('${MODEL_PATH}', 'config.json')
with open(cfg) as f:
    c = json.load(f)
print('yes' if 'quantization_config' in c else 'no')
" 2>/dev/null || echo "no")

if [ "${IS_QUANTIZED}" = "yes" ]; then
    echo "[INFO] Detected quantized model (INT8), using dtype=auto"
    DTYPE="auto"
else
    DTYPE="bfloat16"
fi

vllm serve "${MODEL_PATH}" \
    --dtype "${DTYPE}" \
    --tensor-parallel-size "${TP_SIZE}" \
    --enforce-eager \
    --served-model-name pangu \
    --trust-remote-code \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.9 \
    --host 0.0.0.0 \
    --port "${SERVE_PORT}"
