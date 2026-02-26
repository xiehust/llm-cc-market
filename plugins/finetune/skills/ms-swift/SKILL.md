---
name: ms-swift
description: "This skill should be used when the user asks to fine-tune, train, or adapt large language models or multimodal models using ms-swift or ModelScope SWIFT. Triggers include: 'fine-tune a model', 'LoRA training', 'QLoRA', 'GRPO training', 'RLHF', 'DPO training', 'deploy model with vLLM', 'quantize model', 'evaluate model', 'train embedding model', 'agent training', 'multimodal fine-tuning', 'swift sft', 'swift rlhf', 'swift deploy', 'Megatron training', 'megatron sft', 'MoE training', 'tensor parallel', 'pipeline parallel'. Supports 600+ LLMs and 300+ multimodal models including Qwen3, Llama4, DeepSeek-R1/V3, InternLM3, GLM4/5, Gemma3, Phi-4, and vision models like Qwen3-VL, InternVL3, Qwen3-Omni. Native Megatron integration for 10x MoE training speedup."
---

# ms-swift: LLM Fine-Tuning & Deployment

Fine-tune, align, evaluate, quantize, and deploy large models via the `swift` CLI. Published at AAAI 2025.

**Stable version**: v3.12.x (branch `release/3.12`). **Dev version**: v4.0.0.dev0 (main branch).

## Prerequisites

Ensure ms-swift is installed. If not, run `scripts/setup.sh` from the skill directory. This uses **uv** to create a virtual environment (default: `~/swift-env`) and install ms-swift.

Requirements: Python >=3.9, PyTorch >=2.0, CUDA 12 (for GPU training).

Invoke swift via full path (do NOT rely on PATH): `~/swift-env/bin/swift sft ...`

## CLI Commands

| Command | Purpose |
|---------|---------|
| `swift sft` | Supervised fine-tuning |
| `swift pt` | Pre-training (continual) |
| `swift rlhf` | RLHF/preference training (DPO, GRPO, KTO, PPO, etc.) |
| `swift infer` | Interactive or batch inference |
| `swift deploy` | OpenAI-compatible API server |
| `swift eval` | Benchmark evaluation (EvalScope) |
| `swift export` | Quantize, merge LoRA, push to hub |
| `swift sample` | Data sampling / distillation / RFT generation |
| `swift merge-lora` | Merge LoRA adapters into base model |
| `swift app` | Gradio inference interface |
| `swift web-ui` | Full Web UI (train, infer, eval, quantize) |
| `swift rollout` | vLLM rollout server for GRPO |
| `megatron sft` | Megatron-parallel SFT (10x MoE speedup) |
| `megatron pt` | Megatron-parallel pre-training |
| `megatron rlhf` | Megatron-parallel RLHF (GRPO/DPO/KTO/GKD) |
| `megatron export` | HF<->Megatron weight conversion |

All commands accept `--config path/to/config.yaml` to load parameters from YAML.

## Quick Reference

| Task | Command |
|------|---------|
| LoRA SFT | `swift sft --model X --tuner_type lora --dataset Y` |
| Full SFT | `swift sft --model X --tuner_type full --dataset Y` |
| QLoRA SFT | `swift sft --model X --tuner_type lora --quant_method bnb --quant_bits 4 --dataset Y` |
| Pre-train | `swift pt --model X --dataset Y` |
| GRPO | `swift rlhf --rlhf_type grpo --model X --dataset Y --reward_funcs accuracy format` |
| DPO | `swift rlhf --rlhf_type dpo --model X --dataset Y` |
| KTO | `swift rlhf --rlhf_type kto --model X --dataset Y` |
| PPO | `swift rlhf --rlhf_type ppo --model X --reward_model Z --dataset Y` |
| GKD | `swift rlhf --rlhf_type gkd --model X --teacher_model Z --dataset Y` |
| Inference | `swift infer --model X --infer_backend vllm` |
| Deploy | `swift deploy --model X --infer_backend vllm --port 8000` |
| Evaluate | `swift eval --model X --eval_dataset mmlu gsm8k --infer_backend vllm` |
| Quantize AWQ | `swift export --model X --quant_method awq --quant_bits 4 --dataset Y` |
| Merge LoRA | `swift merge-lora --model X --adapters output/checkpoint-xxx` |
| Distill/Sample | `swift sample --model X --dataset Y --sampler_type distill` |

## Workflow

### 1. Determine Task Type

Gather from the user:
- **Task**: SFT / pre-train / RLHF (which algorithm?) / eval / quantize / deploy
- **Model**: Which model? (e.g., `Qwen/Qwen3-4B-Instruct-2507`)
- **Dataset**: Custom path or hub dataset ID
- **Hardware**: GPU count, memory per GPU
- **Hub**: HuggingFace (`--use_hf`) or ModelScope (default)

### 2. Generate Command

Consult the templates in `references/` for common configurations:
- **`references/training-templates.md`** -- LoRA, QLoRA, full-param, multi-GPU, multimodal, embedding
- **`references/grpo-guide.md`** -- GRPO, DAPO, SAPO, reward functions, vLLM acceleration
- **`references/rlhf-templates.md`** -- DPO, KTO, PPO, GKD, CPO, SimPO, reward model
- **`references/deploy-templates.md`** -- Inference, deployment, quantization, evaluation, sampling
- **`references/dataset-formats.md`** -- All dataset formats by task type
- **`references/megatron-guide.md`** -- Megatron-parallel training (MoE 10x speedup, TP/PP/EP/SP/CP)
- **`references/troubleshooting.md`** -- Common issues and solutions

### 3. Execute

Run training in background (long-running):
```bash
CUDA_VISIBLE_DEVICES=0 ~/swift-env/bin/swift sft --model ... --dataset ... <params> 2>&1 | tee train.log &
```

For multi-GPU, set env vars instead of `torchrun`:
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 NPROC_PER_NODE=4 ~/swift-env/bin/swift sft ...
```

Monitor with `tail -f train.log` or `nvidia-smi`.

### 4. Post-Training

- **Merge LoRA**: `~/swift-env/bin/swift merge-lora --model X --adapters output/checkpoint-xxx --output_dir output/merged`
- **Evaluate**: `~/swift-env/bin/swift eval --model output/checkpoint-xxx --eval_dataset mmlu gsm8k --infer_backend vllm`
- **Quantize**: `~/swift-env/bin/swift export --model output/merged --quant_method awq --quant_bits 4 --dataset Y`
- **Deploy**: `~/swift-env/bin/swift deploy --model output/merged --infer_backend vllm --port 8000`

## Megatron Training

Megatron-SWIFT integrates NVIDIA Megatron-LM parallel technologies for high-performance training. **Use `megatron` instead of `swift` as the command prefix.** Recommended for MoE models (10x speedup) and large-scale multi-GPU training.

### When to Use Megatron

| Scenario | Use | Reason |
|----------|-----|--------|
| MoE models (Qwen3-30B-A3B, etc.) | `megatron sft` | 10x faster than DeepSpeed ZeRO3 |
| Dense full-param on 8+ GPUs | `megatron sft` | Lower memory, ~12% faster |
| LoRA on 1-2 GPUs | `swift sft` | Simpler, no extra deps |
| QLoRA | `swift sft` | Megatron doesn't support QLoRA |

### Megatron Quick Reference

| Task | Command |
|------|---------|
| Megatron SFT (Mcore-Bridge) | `megatron sft --model X --save_safetensors true --tensor_model_parallel_size N --dataset Y` |
| Megatron LoRA | `megatron sft --model X --tuner_type lora --save_safetensors true --merge_lora false --dataset Y` |
| Megatron MoE SFT | `megatron sft --model X --expert_model_parallel_size N --moe_grouped_gemm true --dataset Y` |
| Megatron GRPO | `megatron rlhf --rlhf_type grpo --model X --reward_funcs accuracy format --dataset Y` |
| Megatron DPO | `megatron rlhf --rlhf_type dpo --model X --dataset Y` |
| Megatron Pre-train | `megatron pt --model X --dataset Y` |
| HF to Megatron | `swift export --model X --to_mcore true --output_dir X-mcore` |
| Megatron to HF | `swift export --mcore_model X-mcore --to_hf true --output_dir X-hf` |

**Environment**: Requires extra deps (transformer_engine, megatron-core, apex). See **`references/megatron-guide.md`** for setup and full templates.

**Important**: Megatron uses `--micro_batch_size` and `--global_batch_size` instead of `--per_device_train_batch_size` and `--gradient_accumulation_steps`. Set `PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'` for memory efficiency.

## Key Parameters

### Training
| Param | Description | Default |
|-------|-------------|---------|
| `--model` | HuggingFace/ModelScope model ID or local path | required |
| `--tuner_type` | `lora` / `full` / `adalora` / `dora` / `longlora` / `llamapro` / `vera` / `boft` / `reft` | `lora` |
| `--dataset` | Dataset(s), supports `name#count` for subset sampling | required |
| `--use_hf` | Use HuggingFace hub (omit for ModelScope) | false |
| `--torch_dtype` | `bfloat16` / `float16` / `auto` | `auto` |
| `--num_train_epochs` | Training epochs | 1 |
| `--per_device_train_batch_size` | Batch size per GPU | 1 |
| `--learning_rate` | LR (1e-4 for LoRA, 1e-5 for full) | auto |
| `--lora_rank` | LoRA rank | 8 |
| `--lora_alpha` | LoRA alpha | 32 |
| `--target_modules` | LoRA target modules | `all-linear` |
| `--max_length` | Max sequence length | 2048 |
| `--gradient_accumulation_steps` | Gradient accumulation | 16 |
| `--output_dir` | Output directory | `output` |
| `--deepspeed` | `zero0`/`zero1`/`zero2`/`zero3`/`zero2_offload`/`zero3_offload` | none |
| `--fsdp` | `fsdp2` for built-in FSDP2 | none |
| `--task_type` | `causal_lm` / `seq_cls` / `embedding` / `reranker` | `causal_lm` |

### Memory Optimization
| Param | Description |
|-------|-------------|
| `--quant_method bnb --quant_bits 4` | QLoRA (4-bit quantized LoRA) |
| `--packing true` | Pack samples to fill sequences (requires flash_attn) |
| `--padding_free true` | Flatten batch to avoid padding (requires flash_attn) |
| `--use_liger_kernel true` | Liger kernel for speed/memory optimization |
| `--attn_impl flash_attn` | Flash Attention 2 (use on A100+) |
| `--sequence_parallel_size N` | Ulysses + ring-attention for long sequences |
| `--gradient_checkpointing true` | Trade compute for memory (default: true) |

### Multimodal-Specific
| Param | Description |
|-------|-------------|
| `--freeze_vit true` | Freeze vision encoder (default: true) |
| `--freeze_llm false` | Freeze LLM backbone |
| `--freeze_aligner true` | Freeze alignment module (default: true) |
| `--vit_lr` | Separate learning rate for ViT |
| `--max_pixels` | Max image pixels (H*W) |

### Distributed Training (Environment Variables)
| Variable | Purpose |
|----------|---------|
| `CUDA_VISIBLE_DEVICES` | GPU selection |
| `NPROC_PER_NODE` | GPUs per node (auto-wraps with torchrun) |
| `NNODES` | Number of nodes |
| `NODE_RANK` | This node's rank |
| `MASTER_ADDR` | Master node address |
| `MASTER_PORT` | Master node port |

## Hardware Sizing Guide

| Model Size | Method | Min VRAM | Recommended |
|-----------|--------|----------|-------------|
| 4B | LoRA | 12GB | 1x A10 |
| 7-8B | QLoRA | 9GB | 1x A10 / 3090 |
| 7-8B | LoRA | 18GB | 1x A100-40G |
| 7-8B | Full | 60GB | 2x A100-40G |
| 14B | LoRA | 32GB | 1x A100-40G |
| 14B | Full | 120GB+ | 4x A100-40G + DeepSpeed ZeRO2 |
| 72B | LoRA | 80GB+ | 4x A100-80G + DeepSpeed ZeRO3 |
| 72B | GRPO | 4x 80GB | 4x A100-80G (hybrid mode) |
| MoE (30B-A3B) | Full (Megatron) | 8x 60GB | 8x A800 + Megatron-SWIFT |
| MoE (30B-A3B) | LoRA (Megatron) | 2x 50GB | 2x A100/A800 + Megatron-SWIFT |
| MoE (30B-A3B) | Full (ZeRO3) | 16x 80GB | 16x A800 (10x slower than Megatron) |
| Dense 14B | Full (Megatron) | 8x 64GB | 8x A800 + Megatron-SWIFT |

## Dataset Format

ms-swift auto-detects four input formats. The simplest for SFT:

```json
{"messages": [{"role": "user", "content": "question"}, {"role": "assistant", "content": "answer"}]}
```

Save as `.jsonl` (one JSON per line). Pass the file path: `--dataset /path/to/data.jsonl`

For all formats (ShareGPT, Alpaca, Query-Response) and task-specific formats (DPO, KTO, GRPO, multimodal, agent), see **`references/dataset-formats.md`**.

## Hub Selection

By default ms-swift downloads from **ModelScope**. Add `--use_hf` for **HuggingFace**.
- ModelScope: `AI-ModelScope/alpaca-gpt4-data-zh#500`
- HuggingFace: `yahma/alpaca-cleaned#500`
- Always match dataset ID to the hub being used.

## Plugin System

Extend ms-swift with custom reward functions, loss functions, metrics, and more via `--external_plugins path/to/plugin.py`. See GRPO guide for custom reward function examples.

## Notes

- **Qwen3.5 / Qwen-VL models** require extra deps: `uv pip install qwen_vl_utils torchvision --python ~/swift-env/bin/python`. Qwen3.5 is multimodal even for text tasks.
- For v4.x (dev), `--train_type` is renamed to `--tuner_type` (backward compatible in 3.x)
- For Qwen3 thinking models, use `--enable_thinking true/false` to control reasoning
- Multi-GPU: set `NPROC_PER_NODE=N` instead of manual torchrun
- DeepSpeed requires `CUDA_HOME` to be set (e.g., `export CUDA_HOME=/usr/local/cuda`)
- Output versioned automatically: `output/v0-<timestamp>/`
- Monitor GPU: `watch -n1 nvidia-smi`
- Avoid QLoRA for production (cannot merge weights); use LoRA, merge, then quantize with AWQ/GPTQ
