# Deploy, Inference, Quantization, Evaluation & Sampling Templates

## Inference Backends

| Backend | `--infer_backend` | Best For | Multi-LoRA | QLoRA | TP/PP |
|---------|-------------------|----------|-----------|-------|-------|
| Transformers | `transformers` | Default, no extra deps, QLoRA inference | Yes | Yes | device_map |
| vLLM | `vllm` | High throughput, PagedAttention | Yes | No | TP/PP/DP |
| SGLang | `sglang` | Fast structured generation | No | No | TP/PP/DP/EP |
| LMDeploy | `lmdeploy` | TurboMind engine, quantized models | No | No | TP/DP |

## Inference (Interactive)

```bash
CUDA_VISIBLE_DEVICES=0 \
~/swift-env/bin/swift infer \
    --model Qwen/Qwen3-4B-Instruct-2507 \
    --use_hf \
    --infer_backend vllm \
    --max_new_tokens 2048 \
    --temperature 0.7 \
    --stream true
```

Interactive commands: `multi-line`, `single-line`, `reset-system`, `clear`, `quit`/`exit`.

## Inference with LoRA Adapter

```bash
CUDA_VISIBLE_DEVICES=0 \
~/swift-env/bin/swift infer \
    --adapters output/v0-xxx/checkpoint-xxx \
    --infer_backend vllm \
    --stream true \
    --temperature 0 \
    --max_new_tokens 2048
```

Note: `--adapters` auto-loads the base model and adapter. No need to specify `--model`.

## Inference with Merged LoRA

```bash
CUDA_VISIBLE_DEVICES=0 \
~/swift-env/bin/swift infer \
    --model output/merged \
    --infer_backend vllm \
    --stream true
```

## Deploy as OpenAI-Compatible API

### vLLM (recommended for throughput)
```bash
CUDA_VISIBLE_DEVICES=0 \
~/swift-env/bin/swift deploy \
    --model Qwen/Qwen3-4B-Instruct-2507 \
    --use_hf \
    --infer_backend vllm \
    --port 8000 \
    --vllm_gpu_memory_utilization 0.9 \
    --vllm_max_model_len 8192
```

### SGLang
```bash
CUDA_VISIBLE_DEVICES=0 \
~/swift-env/bin/swift deploy \
    --model Qwen/Qwen3-4B-Instruct-2507 \
    --use_hf \
    --infer_backend sglang \
    --port 8000
```

### LMDeploy
```bash
CUDA_VISIBLE_DEVICES=0 \
~/swift-env/bin/swift deploy \
    --model Qwen/Qwen3-4B-Instruct-2507 \
    --use_hf \
    --infer_backend lmdeploy \
    --port 8000
```

### Multi-GPU Deployment (Tensor Parallel)
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
~/swift-env/bin/swift deploy \
    --model Qwen/Qwen3-72B-Instruct \
    --use_hf \
    --infer_backend vllm \
    --vllm_tensor_parallel_size 4 \
    --port 8000
```

### Deploy with LoRA Adapter
```bash
CUDA_VISIBLE_DEVICES=0 \
~/swift-env/bin/swift deploy \
    --adapters output/v0-xxx/checkpoint-xxx \
    --infer_backend vllm \
    --port 8000
```

### Deploy with Custom Model Name
```bash
~/swift-env/bin/swift deploy \
    --model Qwen/Qwen3-4B-Instruct-2507 \
    --use_hf \
    --infer_backend vllm \
    --served_model_name my-model \
    --port 8000
```

### Client Access
All backends expose OpenAI-compatible API at `http://localhost:8000/v1`.

```bash
# cURL
curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"model": "default", "messages": [{"role": "user", "content": "Hello!"}]}'

# Python (openai SDK)
# pip install openai
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")
response = client.chat.completions.create(
    model="default",
    messages=[{"role": "user", "content": "Hello!"}]
)
print(response.choices[0].message.content)
```

## Quantization

### AWQ (recommended, requires calibration data)
```bash
CUDA_VISIBLE_DEVICES=0 \
~/swift-env/bin/swift export \
    --model Qwen/Qwen3-4B-Instruct-2507 \
    --use_hf \
    --quant_method awq \
    --quant_bits 4 \
    --dataset 'AI-ModelScope/alpaca-gpt4-data-zh#200' \
    --output_dir output/qwen3-8b-awq-int4
```

### GPTQ
```bash
CUDA_VISIBLE_DEVICES=0 \
~/swift-env/bin/swift export \
    --model Qwen/Qwen3-4B-Instruct-2507 \
    --use_hf \
    --quant_method gptq \
    --quant_bits 4 \
    --dataset 'AI-ModelScope/alpaca-gpt4-data-zh#200' \
    --output_dir output/qwen3-8b-gptq-int4
```

### BNB (no calibration data needed)
```bash
CUDA_VISIBLE_DEVICES=0 \
~/swift-env/bin/swift export \
    --model Qwen/Qwen3-4B-Instruct-2507 \
    --use_hf \
    --quant_method bnb \
    --quant_bits 4 \
    --output_dir output/qwen3-8b-bnb-int4
```

### FP8
```bash
CUDA_VISIBLE_DEVICES=0 \
~/swift-env/bin/swift export \
    --model Qwen/Qwen3-4B-Instruct-2507 \
    --use_hf \
    --quant_method fp8 \
    --output_dir output/qwen3-8b-fp8
```

### Quantization Dependencies
- AWQ: `pip install autoawq`
- GPTQ: `pip install auto_gptq optimum`
- BNB: `pip install bitsandbytes`

## Merge LoRA

```bash
~/swift-env/bin/swift merge-lora \
    --model Qwen/Qwen3-4B-Instruct-2507 \
    --use_hf \
    --adapters output/v0-xxx/checkpoint-xxx \
    --output_dir output/merged
```

## Export to Ollama

```bash
~/swift-env/bin/swift export \
    --model Qwen/Qwen3-4B-Instruct-2507 \
    --use_hf \
    --to_ollama true \
    --output_dir output/ollama
```

## Push to Hub

```bash
~/swift-env/bin/swift export \
    --model output/merged \
    --push_to_hub true \
    --hub_model_id your-username/model-name \
    --hub_token your_token
```

Add `--use_hf` to push to HuggingFace instead of ModelScope.

## Evaluation

### Basic Evaluation
```bash
CUDA_VISIBLE_DEVICES=0 \
~/swift-env/bin/swift eval \
    --model Qwen/Qwen3-4B-Instruct-2507 \
    --use_hf \
    --eval_dataset mmlu gsm8k arc \
    --infer_backend vllm
```

### Evaluate Fine-Tuned Model
```bash
CUDA_VISIBLE_DEVICES=0 \
~/swift-env/bin/swift eval \
    --adapters output/v0-xxx/checkpoint-xxx \
    --eval_dataset mmlu gsm8k \
    --infer_backend vllm
```

### Evaluate with Sample Limit
```bash
~/swift-env/bin/swift eval \
    --model Qwen/Qwen3-4B-Instruct-2507 \
    --use_hf \
    --eval_dataset mmlu \
    --eval_limit 100 \
    --infer_backend vllm
```

### Available Evaluation Benchmarks

**Native backend (default):**
`arc`, `bbh`, `ceval`, `cmmlu`, `competition_math`, `general_qa`, `gpqa`, `gsm8k`, `hellaswag`, `humaneval`, `ifeval`, `mmlu`, `mmlu_pro`, `race`, `trivia_qa`, `truthful_qa`

**OpenCompass backend** (`--eval_backend OpenCompass`):
`ARC_c`, `ARC_e`, `mmlu`, `gsm8k`, `math`, `humaneval`, `mbpp`, `bbh`, `hellaswag`, `winogrande`, `BoolQ`, `piqa`, `triviaqa`, `ceval`, `cmmlu`, `agieval`, etc.

**VLMEvalKit backend** (multimodal, `--eval_backend VLMEvalKit`):
`MME`, `MMBench`, `MMMU`, `MathVista`, `OCRBench`, `ChartQA`, `DocVQA`, `TextVQA`, `ScienceQA`, `HallusionBench`, `SEEDBench_IMG`, etc.

### Evaluate During Training
```bash
~/swift-env/bin/swift sft \
    ... \
    --eval_use_evalscope true \
    --eval_dataset gsm8k \
    --eval_steps 500
```

## Sampling / Distillation

### Sample with model (RFT data generation)
```bash
CUDA_VISIBLE_DEVICES=0 \
~/swift-env/bin/swift sample \
    --model Qwen/Qwen3-4B-Instruct-2507 \
    --use_hf \
    --dataset /path/to/prompts.jsonl \
    --num_return_sequences 16 \
    --n_best_to_keep 4 \
    --temperature 0.9 \
    --infer_backend vllm \
    --output_dir output/samples
```

### Distill from API (e.g., GPT-4, Claude)
```bash
OPENAI_API_KEY=sk-xxx \
~/swift-env/bin/swift sample \
    --sampler_type distill \
    --sampler_engine client \
    --model gpt-4o \
    --dataset /path/to/prompts.jsonl \
    --output_dir output/distilled
```

## Gradio Web Interface

```bash
~/swift-env/bin/swift app --model Qwen/Qwen3-4B-Instruct-2507 --use_hf
```

## Full Web UI (Training + Inference + Eval)

```bash
SWIFT_UI_LANG=en ~/swift-env/bin/swift web-ui
```
