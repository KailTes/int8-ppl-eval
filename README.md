# W8A8 INT8 真量化 PPL 评测工具

对任意模型进行 FP16 baseline + W8A8 INT8 真量化 WikiText-2 PPL 评测。

**评测方式**: 先用 `vllm serve` 拉起模型服务，再通过 OpenAI API 调用 lm-eval 评测 PPL。

提供三套量化流程:
- **RTN safetensors** — 纯 CPU，无需额外依赖，适用于所有平台
- **torchao** — NVIDIA GPU (RTX 5090 / SM120+)
- **llmcompressor** — 昇腾 910 (omni-infer)

## 文件清单

```
├── setup.sh                         主评测脚本 (量化 + serve + eval，统一入口)
├── quantize_safetensors_int8.py     RTN 纯 safetensors INT8 量化 (CPU，无需 GPU/NPU)
├── llmcompressor_quantize_w8a8.py   llmcompressor W8A8 量化脚本
├── torchao_eval.sh                  torchao 评测脚本 (NVIDIA GPU)
├── torchao_quantize_w8a8.py         torchao W8A8 量化脚本
├── wikitext_local.yaml              lm-eval 任务配置
├── preprocess_wikitext_local.py     word_perplexity 计算逻辑
├── run_pangu_reference.sh           pangu 参考脚本
└── data/
    └── wikitext2_doc_level/         WikiText-2 测试数据 (7MB)
```

## 方式一: RTN safetensors (推荐，全平台)

纯 CPU 离线量化，不依赖 GPU/NPU/模型代码，直接对 safetensors 权重做 RTN per-channel INT8 量化。
输出 compressed-tensors 格式，兼容 vllm / omni-infer INT8 推理。

所有操作通过 `setup.sh` 完成（量化、拉起服务、评测 PPL）：

```bash
# 第 1 步: 量化 (纯 CPU，昇腾容器内自动禁用 torch_npu)
#   /models 只读时用 QUANT_OUTPUT_BASE 指定输出目录
QUANT_OUTPUT_BASE=/data/models \
bash setup.sh quantize_rtn /models/Qwen3-0.6B
# → 输出: /data/models/Qwen3-0.6B-RTN-W8A8

# 第 2 步: 拉起 INT8 模型服务 + PPL 评测 (一条命令完成 serve + eval)
QUANT_OUTPUT_BASE=/data/models \
bash setup.sh eval_rtn /models/Qwen3-0.6B

# 第 3 步: 停止服务
bash setup.sh stop
```

也可以同时跑 FP16 baseline 对比：

```bash
# FP16 baseline
bash setup.sh eval_fp16 /models/Qwen3-0.6B
bash setup.sh stop

# W8A8 RTN
QUANT_OUTPUT_BASE=/data/models \
bash setup.sh eval_rtn /models/Qwen3-0.6B
bash setup.sh stop
```

> `quantize_safetensors_int8.py` 也可以独立使用:
> `python3 quantize_safetensors_int8.py --model /path/to/fp16 --output /path/to/int8`
> 但后续 serve + eval 仍需通过 `setup.sh` 的 `serve` / `eval` 命令完成。

### 注意事项

1. **CANN 环境**: 脚本自动查找 `/usr/local/Ascend/cann-*/set_env.sh`，如果不存在则回退到 `ascend-toolkit/set_env.sh`。确保容器内 CANN 版本与主机驱动兼容。

2. **NPU 设备**: `ASCEND_RT_VISIBLE_DEVICES` 必须与容器实际映射的 `/dev/davinci*` 设备对应。查看：
   ```bash
   docker inspect <container> --format '{{range .HostConfig.Devices}}{{.PathOnHost}} {{end}}'
   ```

3. **只读挂载**: 模型目录 `/models` 通常只读，用 `QUANT_OUTPUT_BASE` 指向可写目录。

## 方式二: torchao (NVIDIA GPU)

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

## 方式三: llmcompressor (昇腾 910)

适用于昇腾 910 + omni-infer 容器环境，需要安装 llmcompressor。

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
bash /data/int8-ppl-eval/setup.sh all /models/YourModel
```

### 3. 分步执行

```bash
bash setup.sh install                                          # 安装依赖
bash setup.sh quantize  /models/YourModel                      # W8A8 量化 (llmcompressor)
bash setup.sh quantize_rtn /models/YourModel                   # W8A8 量化 (RTN, 无需 install)
bash setup.sh eval_fp16 /models/YourModel 2>&1 | tee fp16.log  # FP16 PPL
bash setup.sh eval_w8a8 /models/YourModel 2>&1 | tee w8a8.log  # W8A8 PPL (llmcompressor)
bash setup.sh eval_rtn  /models/YourModel 2>&1 | tee rtn.log   # W8A8 PPL (RTN)
```

## 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `SERVE_PORT` | `8000` | vllm serve 端口 |
| `TP_SIZE` | `1` | tensor parallel size |
| `QUANT_OUTPUT_BASE` | 模型所在目录 | 量化输出的父目录 (解决只读挂载) |
| `VLLM_PLUGINS` | 自动检测 | vllm 插件列表 |
| `ASCEND_RT_VISIBLE_DEVICES` | `0` | NPU 设备 |
| `VLLM_USE_V1` | `0` | vllm v1 引擎开关 |
| `EXTRA_SERVE_ARGS` | 空 | 额外 vllm serve 参数 |

## 已知问题

| 问题 | 原因 | 处理方式 |
|------|------|---------|
| SM120 compressed-tensors `Int8 not supported` | cuBLAS 不支持 | 使用 torchao 或 RTN 流程 |
| llmcompressor 降级 torch | pip 依赖解析 | setup.sh 自动恢复 |
| graph capture `stream is captured` | CANN 不兼容 | enforce_eager=True |
| `libhccl.so` not found | CANN 版本路径不对 | 脚本已自动检测 cann-*/set_env.sh |
| `/models` read-only 写入失败 | 只读挂载 | 设 `QUANT_OUTPUT_BASE=/data/models` |
| engine 关闭 core dump | vllm 已知问题 | 结果不受影响 |

## 参考结果 (Qwen3-0.6B, WikiText-2)

| 模型 | 平台 | word_perplexity | 劣化 |
|------|------|----------------|------|
| FP16 (vllm, NVIDIA) | RTX 5090 | 60.0549 | — |
| W8A16 (llmcompressor, NVIDIA) | RTX 5090 | 60.0586 | +0.006% |
| **FP16 (torchao, NVIDIA)** | RTX 5090 | 60.0424 | — |
| **W8A8 (torchao, NVIDIA)** | RTX 5090 | 60.9461 | +1.50% |
| FP16 (omni-infer v1.0.0, 昇腾) | 910B2 | 60.0255 | — |
| W8A8 (llmcompressor, 昇腾) | 910B2 | 61.1840 | +1.93% |
| **W8A8 (RTN safetensors, 昇腾)** | 910B2 | **60.8364** | +1.35%* |

> *RTN 劣化基于 FP16=60.0255 计算
