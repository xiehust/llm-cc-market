---
name: ms-swift
description: "Fine-tune and deploy large language models using ModelScope ms-swift. Use when: (1) fine-tuning/training LLMs or multimodal models (SFT, LoRA, QLoRA, full), (2) RLHF training (GRPO, DPO, KTO, etc.), (3) model inference/deployment with vLLM/SGLang, (4) model quantization (GPTQ, AWQ, FP8), (5) model evaluation. Supports 900+ models including Qwen3, Llama4, DeepSeek-R1, InternLM3, GLM4, Mistral, and multimodal models like Qwen3-VL, InternVL3."
---

# ms-swift: LLM Fine-Tuning Skill

Fine-tune, evaluate, quantize, and deploy large models via `swift` CLI.

## Prerequisites

Ensure ms-swift is installed. If not, run `scripts/setup.sh` from the skill directory.
This script uses **uv** to create a virtual environment (default: `~/swift-env`) and install ms-swift.

Requirements: Python ≥3.9, PyTorch ≥2.0, CUDA 12 (for GPU training).

After setup, invoke swift via full path (do NOT rely on PATH):
- `~/swift-env/bin/swift sft ...`

## Quick Reference

| Task | Command |
|------|---------|
| LoRA SFT | `swift sft --model X --train_type lora --dataset Y` |
| Full SFT | `swift sft --model X --train_type full --dataset Y` |
| Pre-train | `swift pt --model X --dataset Y` |
| GRPO | `swift rlhf --rlhf_type grpo --model X --dataset Y` |
| DPO | `swift rlhf --rlhf_type dpo --model X --dataset Y` |
| Inference | `swift infer --model X --infer_backend vllm` |
| Deploy | `swift deploy --model X --infer_backend vllm` |
| Evaluate | `swift eval --model X --eval_dataset Y` |
| Quantize | `swift export --model X --quant_method awq` |
| Merge LoRA | `swift merge_lora --model X --adapters Y` |
| Web UI | `swift app` |

## Workflow

### 1. Determine Task Type

Ask the user for:
- **Task**: SFT / pre-train / RLHF (which algorithm?) / eval / quantize / deploy
- **Model**: Which model? (e.g., Qwen/Qwen3-4B-Instruct-2507)
- **Dataset**: Custom dataset path or HuggingFace dataset ID
- **Hardware**: How many GPUs? Memory per GPU?

### 2. Generate Training Command

Use the templates in `references/` for common configurations:
- `references/training-templates.md` — LoRA, QLoRA, full-param, multi-GPU examples
- `references/rlhf-templates.md` — GRPO, DPO, KTO examples
- `references/deploy-templates.md` — Inference, deployment, quantization, evaluation

Adapt parameters based on user's hardware and requirements.

### 3. Execute

Run training in background (training is long-running):

```bash
CUDA_VISIBLE_DEVICES=0 ~/swift-env/bin/swift sft --model ... --dataset ... <params> 2>&1 &
```

Monitor with `tail -f <logfile>` or `nvidia-smi`.

### 4. Post-Training

After training completes:
- **Merge LoRA** (if applicable): `swift merge_lora --model X --adapters output/checkpoint-xxx`
- **Evaluate**: `swift eval --model output/checkpoint-xxx --eval_dataset <benchmarks>`
- **Quantize**: `swift export --model X --quant_method awq --quant_bits 4`
- **Deploy**: `swift deploy --model X --infer_backend vllm --port 8000`

## Key Parameters

### Training
| Param | Description | Default |
|-------|-------------|---------|
| `--model` | HuggingFace/ModelScope model ID | required |
| `--train_type` | `lora` / `full` / `adalora` / `dora` | `lora` |
| `--dataset` | Dataset name or path (`name#count` for subset) | required |
| `--use_hf` | Download model & dataset from HuggingFace (omit for ModelScope) | false |
| `--torch_dtype` | `bfloat16` / `float16` / `auto` | `auto` |
| `--num_train_epochs` | Training epochs | 1 |
| `--per_device_train_batch_size` | Batch size per GPU | 1 |
| `--learning_rate` | Learning rate | 1e-4 |
| `--lora_rank` | LoRA rank | 8 |
| `--lora_alpha` | LoRA alpha | 32 |
| `--target_modules` | LoRA target | `all-linear` |
| `--max_length` | Max sequence length | 2048 |
| `--gradient_accumulation_steps` | Gradient accumulation | 16 |
| `--output_dir` | Output directory | `output` |
| `--deepspeed` | DeepSpeed config (zero2/zero3) | none |

### Hardware Sizing Guide
| Model Size | Method | Min VRAM | Recommended |
|-----------|--------|----------|-------------|
| 4B | LoRA | 12GB | 1×A10 |
| 7B | QLoRA | 9GB | 1×A10 |
| 7B | LoRA | 18GB | 1×A100-40G |
| 7B | Full | 60GB | 2×A100-40G |
| 14B | LoRA | 32GB | 1×A100-40G |
| 72B | LoRA | 80GB+ | 4×A100-80G + DeepSpeed ZeRO3 |

## Dataset Format

ms-swift accepts multiple formats. The simplest for SFT:

```json
{"messages": [{"role": "user", "content": "question"}, {"role": "assistant", "content": "answer"}]}
```

Save as `.jsonl` (one JSON per line) or `.json` (array).

For custom datasets, pass the file path directly: `--dataset /path/to/data.jsonl`

## Hub Selection

By default ms-swift downloads from **ModelScope**. Add `--use_hf` to use **HuggingFace** for both model and dataset.
- ModelScope datasets: `AI-ModelScope/alpaca-gpt4-data-zh#500`
- HuggingFace datasets: `yahma/alpaca-cleaned#500`
- These are different namespaces — always match dataset ID to the hub being used.

## Notes

- For Qwen3 thinking models, see `references/training-templates.md` for special config
- Multi-GPU: prepend `CUDA_VISIBLE_DEVICES=0,1,2,3` or use `--deepspeed zero2/zero3`
- Monitor GPU usage: `nvidia-smi` or `watch -n1 nvidia-smi`
- Output is versioned automatically: `output/v0-<timestamp>/`
