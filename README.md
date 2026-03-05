# W8A8 INT8 真量化 PPL 评测工具

对任意模型进行 FP16 baseline + W8A8 INT8 真量化 WikiText-2 PPL 评测。

**评测方式**: 先用 `vllm serve` 拉起模型服务，再通过 OpenAI API 调用 lm-eval 评测 PPL。

提供两套量化流程:
- **torchao** — NVIDIA GPU (RTX 5090 / SM120+)，推荐
- **llmcompressor** — 昇腾 910 (omni-infer)

## 文件清单

```
├── torchao_eval.sh                  torchao 评测脚本 (NVIDIA GPU)
├── torchao_quantize_w8a8.py         torchao W8A8 量化脚本
├── llmcompressor_setup.sh           llmcompressor 评测脚本 (昇腾)
├── llmcompressor_quantize_w8a8.py   llmcompressor W8A8 量化脚本
├── wikitext_local.yaml              lm-eval 任务配置
├── preprocess_wikitext_local.py     word_perplexity 计算逻辑
├── run_pangu_reference.sh           pangu 参考脚本
└── data/
    └── wikitext2_doc_level/         WikiText-2 测试数据 (7MB)
```

## 方式一: torchao (NVIDIA GPU)

适用于 NVIDIA GPU，特别是 SM120 (RTX 5090) 上 compressed-tensors INT8 不可用的情况。

```bash
# 一键执行: 量化 + FP16 PPL + W8A8 PPL
bash torchao_eval.sh all /home/models/Qwen3-0.6B

# 分步执行
bash torchao_eval.sh quantize  /home/models/Qwen3-0.6B   # 量化
bash torchao_eval.sh eval_fp16 /home/models/Qwen3-0.6B   # FP16 PPL
bash torchao_eval.sh eval_w8a8 /home/models/Qwen3-0.6B   # W8A8 PPL
```

环境变量:
```bash
SERVE_PORT=8080 bash torchao_eval.sh all /home/models/Qwen3-0.6B
TP_SIZE=4 bash torchao_eval.sh all /home/models/Large-92B
EXTRA_SERVE_ARGS="--max-model-len 4096" bash torchao_eval.sh eval_fp16 /home/models/MyModel
```

## 方式二: llmcompressor (昇腾 910)

适用于昇腾 910 + omni-infer 容器环境。

### 1. 启动 omni-infer v1.0.0 容器

```bash
docker run -d --name int8-eval \
  --device /dev/davinci0 \
  --device /dev/davinci_manager \
  --device /dev/devmm_svm \
  --device /dev/hisi_hdc \
  -v /path/to/your/models:/models:ro \
  -v /path/to/this/repo:/data/int8-ppl-eval \
  -v /usr/local/dcmi:/usr/local/dcmi \
  -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
  -v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
  -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
  -v /etc/ascend_install.info:/etc/ascend_install.info \
  --entrypoint /bin/sh \
  swr.cn-east-4.myhuaweicloud.com/omni-ci/omniinfer-a2-arm:v1.0.0-vllm \
  -c 'while true; do sleep 3600; done'
```

### 2. 一键执行

```bash
docker exec -it int8-eval bash
bash /data/int8-ppl-eval/llmcompressor_setup.sh all /models/YourModel
```

### 3. 分步执行

```bash
bash llmcompressor_setup.sh install                                          # 安装依赖
bash llmcompressor_setup.sh quantize  /models/YourModel                      # W8A8 量化
bash llmcompressor_setup.sh eval_fp16 /models/YourModel 2>&1 | tee fp16.log  # FP16 PPL
bash llmcompressor_setup.sh eval_w8a8 /models/YourModel 2>&1 | tee w8a8.log  # W8A8 PPL
```

## 已知问题

| 问题 | 原因 | 处理方式 |
|------|------|---------|
| SM120 compressed-tensors `Int8 not supported` | cuBLAS 不支持 | 使用 torchao 流程 |
| llmcompressor 降级 torch | pip 依赖解析 | llmcompressor_setup.sh 自动恢复 |
| graph capture `stream is captured` | CANN 不兼容 | enforce_eager=True |
| engine 关闭 core dump | vllm 已知问题 | 结果不受影响 |

## 参考结果 (Qwen3-0.6B)

| 模型 | 平台 | word_perplexity | 劣化 |
|------|------|----------------|------|
| FP16 (vllm, NVIDIA) | RTX 5090 | 60.0549 | — |
| W8A16 (llmcompressor, NVIDIA) | RTX 5090 | 60.0586 | +0.006% |
| **FP16 (torchao, NVIDIA)** | RTX 5090 | 60.0424 | — |
| **W8A8 (torchao, NVIDIA)** | RTX 5090 | 60.9461 | +1.50% |
| FP16 (omni-infer v1.0.0, 昇腾) | 910B2 | 60.0255 | — |
| W8A8 (llmcompressor, 昇腾) | 910B2 | 61.1840 | +1.93% |
