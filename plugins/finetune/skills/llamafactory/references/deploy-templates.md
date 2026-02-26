# Deploy, Inference, Export, Quantization & Evaluation Templates

## Inference Backends

| Backend | `infer_backend` | Best For | LoRA | QLoRA |
|---------|-----------------|----------|------|-------|
| HuggingFace | `huggingface` | Default, all features | Yes | Yes |
| vLLM | `vllm` | High throughput | Yes | No |
| SGLang | `sglang` | Fast generation, tensor parallel | Yes | No |
| KTransformers | `ktransformers` | CPU+GPU hybrid for huge models | Yes | No |

## CLI Chat (Interactive)

### Base model
```yaml
# chat_config.yaml
model_name_or_path: Qwen/Qwen3-4B-Instruct-2507
template: qwen3_nothink
trust_remote_code: true
```

```bash
~/lmf-env/bin/llamafactory-cli chat chat_config.yaml
```

### With LoRA adapter
```yaml
model_name_or_path: Qwen/Qwen3-4B-Instruct-2507
adapter_name_or_path: saves/qwen3-8b/lora/sft
template: qwen3_nothink
finetuning_type: lora
trust_remote_code: true
```

### With vLLM backend
```yaml
model_name_or_path: Qwen/Qwen3-4B-Instruct-2507
template: qwen3_nothink
infer_backend: vllm
vllm_maxlen: 4096
vllm_gpu_util: 0.9
trust_remote_code: true
```

## Web Chat

```bash
~/lmf-env/bin/llamafactory-cli webchat chat_config.yaml
```

## Deploy as OpenAI-Compatible API

### HuggingFace backend
```yaml
# api_config.yaml
model_name_or_path: Qwen/Qwen3-4B-Instruct-2507
template: qwen3_nothink
infer_backend: huggingface
trust_remote_code: true
```

```bash
API_PORT=8000 ~/lmf-env/bin/llamafactory-cli api api_config.yaml
```

### vLLM backend (recommended for throughput)
```yaml
model_name_or_path: Qwen/Qwen3-4B-Instruct-2507
template: qwen3_nothink
infer_backend: vllm
vllm_maxlen: 8192
vllm_gpu_util: 0.9
trust_remote_code: true
```

### SGLang backend
```yaml
model_name_or_path: Qwen/Qwen3-4B-Instruct-2507
template: qwen3_nothink
infer_backend: sglang
sglang_maxlen: 8192
sglang_mem_fraction: 0.9
trust_remote_code: true
```

### Deploy with LoRA adapter
```yaml
model_name_or_path: Qwen/Qwen3-4B-Instruct-2507
adapter_name_or_path: saves/qwen3-8b/lora/sft
template: qwen3_nothink
finetuning_type: lora
infer_backend: vllm
trust_remote_code: true
```

### Client Access
All backends expose OpenAI-compatible API at `http://localhost:8000/v1`.

```bash
curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"model": "Qwen/Qwen3-4B-Instruct-2507", "messages": [{"role": "user", "content": "Hello!"}]}'
```

```python
from openai import OpenAI
client = OpenAI(api_key="0", base_url="http://localhost:8000/v1")
response = client.chat.completions.create(
    model="Qwen/Qwen3-4B-Instruct-2507",
    messages=[{"role": "user", "content": "Hello!"}]
)
print(response.choices[0].message.content)
```

### vLLM Parameters
| Param | Description | Default |
|-------|-------------|---------|
| `vllm_maxlen` | Max sequence length | 4096 |
| `vllm_gpu_util` | GPU memory fraction (0-1) | 0.7 |
| `vllm_enforce_eager` | Disable CUDA graph | false |
| `vllm_max_lora_rank` | Max LoRA rank | 32 |

### SGLang Parameters
| Param | Description | Default |
|-------|-------------|---------|
| `sglang_maxlen` | Max sequence length | 4096 |
| `sglang_mem_fraction` | Memory fraction (0-1) | 0.7 |
| `sglang_tp_size` | Tensor parallel size | -1 (auto) |

## Merge LoRA Adapters

```yaml
# merge_config.yaml
model_name_or_path: Qwen/Qwen3-4B-Instruct-2507
adapter_name_or_path: saves/qwen3-8b/lora/sft
template: qwen3_nothink
finetuning_type: lora
trust_remote_code: true

export_dir: saves/qwen3-8b-merged
export_size: 5                  # Shard size in GB
export_device: cpu              # or "auto" to use GPU
export_legacy_format: false     # .safetensors format
```

```bash
~/lmf-env/bin/llamafactory-cli export merge_config.yaml
```

## Push to HuggingFace Hub

```yaml
model_name_or_path: saves/qwen3-8b-merged
template: qwen3_nothink
trust_remote_code: true

export_dir: saves/qwen3-8b-hub
export_hub_model_id: your-username/qwen3-8b-sft
```

## Quantization -- GPTQ (Post-Training)

```yaml
# gptq_config.yaml
model_name_or_path: Qwen/Qwen3-4B-Instruct-2507
template: qwen3_nothink
trust_remote_code: true

export_dir: saves/qwen3-8b-gptq-int4
export_quantization_bit: 4
export_quantization_dataset: data/c4_demo.jsonl
export_quantization_nsamples: 128
export_quantization_maxlen: 1024
export_size: 5
export_device: cpu
export_legacy_format: false
```

```bash
~/lmf-env/bin/llamafactory-cli export gptq_config.yaml
```

## Quantization -- Using Pre-Quantized Models

Pre-quantized GPTQ/AWQ models can be used directly for LoRA fine-tuning:

```yaml
# No quantization_bit needed -- model is already quantized
model_name_or_path: TechxGenus/Qwen3-4B-Instruct-2507-GPTQ-Int4
finetuning_type: lora
# ... rest same as LoRA SFT
```

## Export Ollama Modelfile

```yaml
model_name_or_path: saves/qwen3-8b-merged   # Must be merged first
template: qwen3_nothink
trust_remote_code: true
export_dir: saves/qwen3-8b-ollama
```

## Evaluation -- NLG (BLEU/ROUGE)

Use `do_predict: true` in the SFT training config:

```yaml
model_name_or_path: Qwen/Qwen3-4B-Instruct-2507
adapter_name_or_path: saves/qwen3-8b/lora/sft
template: qwen3_nothink
finetuning_type: lora
trust_remote_code: true

stage: sft
do_predict: true
predict_with_generate: true

eval_dataset: alpaca_en_demo
per_device_eval_batch_size: 1
output_dir: saves/qwen3-8b/predict
```

```bash
~/lmf-env/bin/llamafactory-cli train predict_config.yaml
# Then evaluate:
python scripts/eval_bleu_rouge.py saves/qwen3-8b/predict/generated_predictions.jsonl
```

## Evaluation -- Benchmarks (MMLU, C-Eval)

Note: `llamafactory-cli eval` is deprecated. Use vLLM batch inference instead:

```bash
python scripts/vllm_infer.py \
    --model_name_or_path Qwen/Qwen3-4B-Instruct-2507 \
    --template qwen3_nothink \
    --dataset alpaca_en_demo
```

## Batch Inference with vLLM

```bash
python scripts/vllm_infer.py \
    --model_name_or_path saves/qwen3-8b-merged \
    --template qwen3_nothink \
    --dataset my_test_data \
    --dataset_dir /path/to/data

python scripts/eval_bleu_rouge.py generated_predictions.jsonl
```

## LlamaBoard (Full Web UI)

Launch the full GUI for training, inference, evaluation, and export:

```bash
~/lmf-env/bin/llamafactory-cli webui
```

Set language: `LANG=en ~/lmf-env/bin/llamafactory-cli webui`

## KTransformers (CPU+GPU Hybrid for 671B+ Models)

```yaml
model_name_or_path: deepseek-ai/DeepSeek-V3
use_kt: true
kt_optimize_rule: /path/to/optimize_rule
cpu_infer: 32
chunk_size: 8192
kt_maxlen: 4096
kt_use_cuda_graph: true
trust_remote_code: true
```

Enables fine-tuning DeepSeek-V3 (671B) with ~70GB GPU + 1.3TB host RAM.
