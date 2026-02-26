# Deploy, Inference, Quantization & Evaluation Templates

## Inference (Local)

```bash
swift infer \
    --model Qwen/Qwen3-8B-Instruct \
    --infer_backend vllm \
    --max_length 4096
```

## Inference with LoRA Adapter

```bash
swift infer \
    --model Qwen/Qwen3-8B-Instruct \
    --adapters output/checkpoint-xxx \
    --infer_backend vllm
```

## Deploy as OpenAI-Compatible API

```bash
swift deploy \
    --model Qwen/Qwen3-8B-Instruct \
    --infer_backend vllm \
    --port 8000
```

Then call: `curl http://localhost:8000/v1/chat/completions -H "Content-Type: application/json" -d '{"model":"default","messages":[{"role":"user","content":"Hello"}]}'`

## Deploy with LoRA

```bash
swift deploy \
    --model Qwen/Qwen3-8B-Instruct \
    --adapters output/checkpoint-xxx \
    --infer_backend vllm \
    --port 8000
```

## Quantization — AWQ

```bash
swift export \
    --model Qwen/Qwen3-8B-Instruct \
    --quant_method awq \
    --quant_bits 4 \
    --dataset 'AI-ModelScope/alpaca-gpt4-data-zh#200' \
    --output_dir output/qwen3-8b-awq-int4
```

## Quantization — GPTQ

```bash
swift export \
    --model Qwen/Qwen3-8B-Instruct \
    --quant_method gptq \
    --quant_bits 4 \
    --dataset 'AI-ModelScope/alpaca-gpt4-data-zh#200' \
    --output_dir output/qwen3-8b-gptq-int4
```

## Merge LoRA

```bash
swift merge_lora \
    --model Qwen/Qwen3-8B-Instruct \
    --adapters output/checkpoint-xxx \
    --output_dir output/merged
```

## Evaluation

```bash
swift eval \
    --model Qwen/Qwen3-8B-Instruct \
    --eval_dataset ceval mmlu arc \
    --infer_backend vllm
```

## Evaluate Fine-Tuned Model

```bash
swift eval \
    --model output/checkpoint-xxx \
    --eval_dataset ceval mmlu \
    --infer_backend vllm
```

## Inference Backends

| Backend | `--infer_backend` | Best For |
|---------|-------------------|----------|
| vLLM | `vllm` | High throughput, PagedAttention |
| SGLang | `sglang` | Fast structured generation |
| LMDeploy | `lmdeploy` | TurboMind engine, quantized models |
| Transformers | `pt` | Default, no extra deps |
