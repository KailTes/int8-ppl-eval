#!/bin/bash
# ============================================================
# run_pangu.sh 参考版本 (根据日志推断)
#
# 实际私密代码路径: /home/p00929643/omni-npu/start_server/run_pangu.sh
# 此文件仅用于记录假设参数，方便调试和讨论。
# ============================================================

source /usr/local/Ascend/ascend-toolkit/set_env.sh 2>/dev/null || true

MODEL_PATH="/data/weights/pangu_v2/92B/iter_0059000_hf/"
SERVE_PORT=8000
TP_SIZE=8

# 推断自日志:
#   - dtype=torch.bfloat16
#   - tensor_parallel_size=4 或 8 (日志有 TP0~TP7 但 npu-smi 显示 4 卡, 可能 TP=4 + EP=2)
#   - enforce_eager=True
#   - served_model_name=pangu
#   - max_seq_len=4096
#   - trust_remote_code=True
#   - gpu_memory_utilization 未知 (推测 0.9)

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
