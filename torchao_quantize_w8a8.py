"""
W8A8 INT8 量化脚本 (torchao, GPU)
使用 transformers TorchAoConfig + torchao Int8DynamicActivationInt8WeightConfig

用法:
  python3 torchao_quantize_w8a8.py --model /path/to/model --output /path/to/output

量化配置:
  权重: INT8, symmetric, per-channel static
  激活: INT8, dynamic per-token
"""
import argparse
import os

os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"


def main():
    parser = argparse.ArgumentParser(description="W8A8 INT8 quantization via torchao")
    parser.add_argument("--model", required=True, help="FP16/BF16 model path")
    parser.add_argument("--output", required=True, help="Output directory for quantized model")
    args = parser.parse_args()

    from transformers import AutoTokenizer, AutoModelForCausalLM, TorchAoConfig
    from torchao.quantization import Int8DynamicActivationInt8WeightConfig

    print(f"Loading model from {args.model} ...")
    quantization_config = TorchAoConfig(Int8DynamicActivationInt8WeightConfig())
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype="auto",
        device_map="auto",
        quantization_config=quantization_config,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    print(f"Saving to {args.output} ...")
    os.makedirs(args.output, exist_ok=True)
    model.save_pretrained(args.output, safe_serialization=False)
    tokenizer.save_pretrained(args.output)
    print("Done!")


if __name__ == "__main__":
    main()
