# Megatron-SWIFT Training Guide

Megatron-SWIFT integrates NVIDIA Megatron-LM's parallel training technologies into ms-swift, enabling highly efficient large-scale model training. It supports data parallelism (DP), tensor parallelism (TP), pipeline parallelism (PP), sequence parallelism (SP), context parallelism (CP), and expert parallelism (EP). **Recommended for MoE model training where it typically achieves 10x speedup over DeepSpeed.**

## When to Use Megatron vs Standard swift

| Scenario | Use | Reason |
|----------|-----|--------|
| MoE model training (e.g., Qwen3-30B-A3B) | `megatron sft` | 10x faster than DeepSpeed ZeRO3 |
| Dense model full-param training on 8+ GPUs | `megatron sft` | Lower memory, faster than ZeRO2 |
| LoRA on single GPU | `swift sft` | Simpler, no extra deps |
| QLoRA | `swift sft` | Megatron doesn't support QLoRA |
| Quick prototyping | `swift sft` | No environment setup needed |

## Supported Tasks

| Task | Full Param | LoRA | MoE | Multimodal | FP8 |
|------|-----------|------|-----|------------|-----|
| Pre-training (`megatron pt`) | Yes | Yes | Yes | Yes | Yes |
| SFT (`megatron sft`) | Yes | Yes | Yes | Yes | Yes |
| GRPO (`megatron rlhf --rlhf_type grpo`) | Yes | Yes | Yes | Yes | Yes |
| DPO (`megatron rlhf --rlhf_type dpo`) | Yes | Yes | Yes | Yes | Yes |
| KTO (`megatron rlhf --rlhf_type kto`) | Yes | Yes | Yes | Yes | Yes |
| GKD (`megatron rlhf --rlhf_type gkd`) | Yes | Yes | Yes | Yes | Yes |
| RM (`megatron rlhf --rlhf_type rm`) | Yes | Yes | Yes | Yes | Yes |
| Embedding | Yes | Yes | Yes | Yes | Yes |
| Reranker | Yes | Yes | Yes | Yes | Yes |
| Seq Classification | Yes | Yes | Yes | Yes | Yes |

## Supported Models

Qwen3, Qwen3-MoE, Qwen3-VL, Qwen3-Omni, Qwen2.5, Llama3, DeepSeek-R1, GLM4.5, InternVL3.5, Kimi-VL, and more. Full list at the ms-swift supported models documentation.

## Environment Setup

Megatron-SWIFT requires additional dependencies beyond ms-swift:

### Option A: Install into swift-env (Recommended for EC2 / no system cuDNN)

On EC2 or environments where cuDNN/NCCL dev headers are not installed globally, you must point the build at the pip-installed nvidia packages inside the venv:

```bash
# Set environment variables for compilation
# Adjust python version (3.11) to match your swift-env
export CUDA_HOME=/usr
export CUDNN_HOME=~/swift-env/lib/python3.11/site-packages/nvidia/cudnn
export CUDNN_PATH=~/swift-env/lib/python3.11/site-packages/nvidia/cudnn
export CPLUS_INCLUDE_PATH="~/swift-env/lib/python3.11/site-packages/nvidia/cudnn/include:~/swift-env/lib/python3.11/site-packages/nvidia/nccl/include"
export LIBRARY_PATH="~/swift-env/lib/python3.11/site-packages/nvidia/cudnn/lib:~/swift-env/lib/python3.11/site-packages/nvidia/nccl/lib"

# 1. Install pybind11
uv pip install pybind11 --python ~/swift-env/bin/python

# 2. Install Transformer Engine + megatron-core
# If errors occur, see: https://github.com/modelscope/ms-swift/issues/3793
CUDA_HOME=/usr \
CUDNN_HOME=~/swift-env/lib/python3.11/site-packages/nvidia/cudnn \
CUDNN_PATH=~/swift-env/lib/python3.11/site-packages/nvidia/cudnn \
CPLUS_INCLUDE_PATH="~/swift-env/lib/python3.11/site-packages/nvidia/cudnn/include:~/swift-env/lib/python3.11/site-packages/nvidia/nccl/include" \
LIBRARY_PATH="~/swift-env/lib/python3.11/site-packages/nvidia/cudnn/lib:~/swift-env/lib/python3.11/site-packages/nvidia/nccl/lib" \
uv pip install "transformer-engine[pytorch]" megatron-core --python ~/swift-env/bin/python

# 3. Install NVIDIA apex (optional but recommended)
# Megatron-SWIFT can run without apex by setting --gradient_accumulation_fusion false
git clone https://github.com/NVIDIA/apex
cd apex
CUDA_HOME=/usr \
uv pip install -v --no-build-isolation \
    --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" \
    . --python ~/swift-env/bin/python
cd ..

# 4. Clone Megatron-LM repo (needed for training modules)
# swift will auto-clone, or set MEGATRON_LM_PATH to an existing clone
git clone --branch core_r0.15.0 https://github.com/NVIDIA/Megatron-LM.git
export MEGATRON_LM_PATH='/path/to/Megatron-LM'

# 5. Install flash-attn
# Do NOT install a version higher than transformer_engine's limit
CUDA_HOME=/usr MAX_JOBS=8 \
uv pip install "flash-attn==2.8.3" --no-build-isolation --python ~/swift-env/bin/python

# 6. For multi-node training, set shared cache path (critical!)
export MODELSCOPE_CACHE='/shared/storage/path'
```

**Why the extra env vars?** Transformer Engine compiles CUDA kernels at install time and needs cuDNN/NCCL headers. On EC2 GPU instances, `nvcc` is at `/usr/bin/nvcc` (so `CUDA_HOME=/usr`) but cuDNN dev headers are not installed system-wide. The pip packages `nvidia-cudnn-cu12` and `nvidia-nccl-cu12` (installed as dependencies) contain the needed headers inside the venv.

### Option B: Standard install (system with cuDNN dev headers)

If your system has cuDNN and NCCL development headers installed globally (e.g., Docker images, DGX):

```bash
# 1. Install pybind11
pip install pybind11

# 2. Install Transformer Engine + megatron-core
pip install --no-build-isolation transformer_engine[pytorch]
pip install megatron-core

# 3. Install NVIDIA apex (optional but recommended)
# Megatron-SWIFT can run without apex by setting --gradient_accumulation_fusion false
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation \
    --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./

# 4. Clone Megatron-LM repo (needed for training modules)
# swift will auto-clone, or set MEGATRON_LM_PATH to an existing clone
git clone --branch core_r0.15.0 https://github.com/NVIDIA/Megatron-LM.git
export MEGATRON_LM_PATH='/path/to/Megatron-LM'

# 5. Install flash-attn
# Do NOT install a version higher than transformer_engine's limit
MAX_JOBS=8 pip install "flash-attn==2.8.3" --no-build-isolation

# 6. For multi-node training, set shared cache path (critical!)
export MODELSCOPE_CACHE='/shared/storage/path'
```

### Verify Installation

After installing, run the environment check script to confirm everything works:

```bash
~/swift-env/bin/python scripts/check_megatron_env.py
```

This checks all required (PyTorch, CUDA, transformer_engine, megatron-core, swift) and optional (apex, flash_attn, MEGATRON_LM_PATH) dependencies, reports versions, and provides fix instructions for any failures.

### Recommended Versions

| Package | Range | Recommended |
|---------|-------|-------------|
| Python | >=3.9 | 3.10/3.11 |
| CUDA | 12.x | CUDA 12.8 |
| PyTorch | >=2.0 | 2.8.0 |
| transformer_engine | >=2.3 | 2.10.0 |
| megatron_core | >=0.12,<0.16 | 0.15 |
| flash_attn | | 2.8.3 |

### Docker Image (Pre-built)

```
modelscope-registry.us-west-1.cr.aliyuncs.com/modelscope-repo/modelscope:ubuntu22.04-cuda12.8.1-py311-torch2.9.0-vllm0.13.0-modelscope1.33.0-swift3.12.5
```

## Quick Start: Mcore-Bridge (Recommended)

Mcore-Bridge eliminates the need for manual HF-to-Megatron weight conversion. It directly loads safetensors weights and saves back to safetensors format.

### Full Parameter SFT (Dense Model)

```bash
PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
NPROC_PER_NODE=2 \
CUDA_VISIBLE_DEVICES=0,1 \
megatron sft \
    --model Qwen/Qwen2.5-7B-Instruct \
    --use_hf \
    --save_safetensors true \
    --dataset 'AI-ModelScope/alpaca-gpt4-data-zh#500' \
              'AI-ModelScope/alpaca-gpt4-data-en#500' \
    --tensor_model_parallel_size 2 \
    --sequence_parallel true \
    --micro_batch_size 16 \
    --global_batch_size 16 \
    --recompute_granularity full \
    --recompute_method uniform \
    --recompute_num_layers 1 \
    --finetune true \
    --cross_entropy_loss_fusion true \
    --lr 1e-5 \
    --lr_warmup_fraction 0.05 \
    --min_lr 1e-6 \
    --num_train_epochs 1 \
    --output_dir megatron_output/Qwen2.5-7B-Instruct \
    --save_steps 100 \
    --max_length 2048 \
    --dataloader_num_workers 4 \
    --no_save_optim true \
    --no_save_rng true \
    --dataset_num_proc 4
```

### LoRA SFT (Dense Model)

```bash
PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
NPROC_PER_NODE=2 \
CUDA_VISIBLE_DEVICES=0,1 \
megatron sft \
    --model Qwen/Qwen2.5-7B-Instruct \
    --use_hf \
    --save_safetensors true \
    --merge_lora false \
    --tuner_type lora \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --dataset 'AI-ModelScope/alpaca-gpt4-data-zh#500' \
              'AI-ModelScope/alpaca-gpt4-data-en#500' \
    --tensor_model_parallel_size 2 \
    --sequence_parallel true \
    --micro_batch_size 16 \
    --global_batch_size 16 \
    --recompute_granularity full \
    --recompute_method uniform \
    --recompute_num_layers 1 \
    --finetune true \
    --cross_entropy_loss_fusion true \
    --lr 1e-4 \
    --lr_warmup_fraction 0.05 \
    --min_lr 1e-5 \
    --num_train_epochs 1 \
    --output_dir megatron_output/Qwen2.5-7B-Instruct \
    --save_steps 100 \
    --max_length 2048 \
    --dataloader_num_workers 4 \
    --no_save_optim true \
    --no_save_rng true \
    --dataset_num_proc 4
```

### MoE Model SFT (e.g., Qwen3-30B-A3B)

```bash
# 8 GPUs, ~76GB each, ~3s/it
PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
NPROC_PER_NODE=8 \
megatron sft \
    --model Qwen/Qwen3-30B-A3B-Instruct-2507 \
    --use_hf \
    --save_safetensors true \
    --dataset 'your_dataset#20000' \
    --load_from_cache_file true \
    --moe_permute_fusion true \
    --pipeline_model_parallel_size 2 \
    --decoder_first_pipeline_num_layers 25 \
    --tensor_model_parallel_size 4 \
    --expert_model_parallel_size 4 \
    --moe_grouped_gemm true \
    --moe_shared_expert_overlap true \
    --moe_aux_loss_coeff 1e-6 \
    --micro_batch_size 1 \
    --global_batch_size 4 \
    --recompute_granularity full \
    --recompute_method uniform \
    --recompute_num_layers 1 \
    --num_train_epochs 1 \
    --finetune true \
    --cross_entropy_loss_fusion true \
    --lr 1e-5 \
    --lr_warmup_fraction 0.05 \
    --min_lr 1e-6 \
    --output_dir megatron_output/Qwen3-30B-A3B \
    --save_steps 500 \
    --max_length 8192 \
    --packing true \
    --dataloader_num_workers 8 \
    --dataset_num_proc 8 \
    --no_save_optim true \
    --no_save_rng true \
    --sequence_parallel true \
    --moe_expert_capacity_factor 2 \
    --attention_backend flash
```

### MoE LoRA Training

```bash
# 2 GPUs, ~50GB each
PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
NPROC_PER_NODE=2 \
CUDA_VISIBLE_DEVICES=0,1 \
megatron sft \
    --model Qwen/Qwen3-30B-A3B-Instruct-2507 \
    --use_hf \
    --save_safetensors true \
    --merge_lora false \
    --tuner_type lora \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --dataset 'your_dataset#2000' \
    --load_from_cache_file true \
    --expert_model_parallel_size 2 \
    --moe_permute_fusion true \
    --moe_grouped_gemm true \
    --moe_shared_expert_overlap true \
    --moe_aux_loss_coeff 1e-3 \
    --micro_batch_size 8 \
    --global_batch_size 16 \
    --recompute_granularity full \
    --recompute_method uniform \
    --recompute_num_layers 1 \
    --num_train_epochs 1 \
    --finetune true \
    --cross_entropy_loss_fusion true \
    --lr 1e-4 \
    --lr_warmup_fraction 0.05 \
    --min_lr 1e-5 \
    --output_dir megatron_output/Qwen3-30B-A3B \
    --save_steps 200 \
    --max_length 2048 \
    --dataloader_num_workers 8 \
    --dataset_num_proc 8 \
    --no_save_optim true \
    --no_save_rng true \
    --sequence_parallel true \
    --moe_expert_capacity_factor 2 \
    --attention_backend flash
```

### Multimodal SFT (e.g., Qwen3-VL)

```bash
# 2 GPUs, ~76GB each
PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
NPROC_PER_NODE=2 \
IMAGE_MAX_TOKEN_NUM=1024 \
VIDEO_MAX_TOKEN_NUM=128 \
FPS_MAX_FRAMES=16 \
CUDA_VISIBLE_DEVICES=0,1 \
megatron sft \
    --model Qwen/Qwen3-VL-8B-Instruct \
    --use_hf \
    --save_safetensors true \
    --dataset /path/to/multimodal_data.jsonl \
    --tensor_model_parallel_size 2 \
    --sequence_parallel true \
    --packing true \
    --freeze_llm false \
    --freeze_vit true \
    --freeze_aligner true \
    --micro_batch_size 1 \
    --global_batch_size 4 \
    --recompute_granularity full \
    --recompute_method uniform \
    --recompute_num_layers 1 \
    --finetune true \
    --cross_entropy_loss_fusion true \
    --lr 1e-5 \
    --lr_warmup_fraction 0.05 \
    --min_lr 1e-6 \
    --num_train_epochs 1 \
    --output_dir megatron_output/Qwen3-VL-8B \
    --save_steps 200 \
    --max_length 2048 \
    --dataloader_num_workers 4 \
    --no_save_optim true \
    --no_save_rng true \
    --dataset_num_proc 8
```

### Pre-training

Replace `megatron sft` with `megatron pt` to use generative (non-chat) template:

```bash
NPROC_PER_NODE=8 \
megatron pt \
    --model Qwen/Qwen2.5-7B \
    --use_hf \
    --save_safetensors true \
    --dataset /path/to/pretrain_data.jsonl \
    --tensor_model_parallel_size 2 \
    --sequence_parallel true \
    --micro_batch_size 4 \
    --global_batch_size 32 \
    --recompute_granularity selective \
    --finetune true \
    --cross_entropy_loss_fusion true \
    --lr 1e-5 \
    --num_train_epochs 1 \
    --max_length 4096 \
    --packing true \
    --output_dir megatron_output/pretrain \
    --no_save_optim true \
    --no_save_rng true
```

## Megatron GRPO

Requires ms-swift >= 3.11. Supports full-param and LoRA, all parallel strategies (CP/PP/TP/EP), vLLM colocate and server modes.

```bash
PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
NPROC_PER_NODE=4 \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
megatron rlhf \
    --rlhf_type grpo \
    --model Qwen/Qwen2.5-7B-Instruct \
    --use_hf \
    --save_safetensors true \
    --dataset /path/to/grpo_data.jsonl \
    --reward_funcs accuracy format \
    --num_generations 8 \
    --max_completion_length 2048 \
    --tensor_model_parallel_size 2 \
    --sequence_parallel true \
    --micro_batch_size 1 \
    --global_batch_size 8 \
    --recompute_granularity full \
    --recompute_method uniform \
    --recompute_num_layers 1 \
    --finetune true \
    --cross_entropy_loss_fusion true \
    --lr 5e-7 \
    --beta 0.04 \
    --num_train_epochs 1 \
    --output_dir megatron_output/grpo \
    --no_save_optim true \
    --no_save_rng true
```

### Megatron GRPO with vLLM (Server Mode)

Terminal 1 -- rollout server:
```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 \
swift rollout \
    --model Qwen/Qwen2.5-7B-Instruct \
    --use_hf \
    --infer_backend vllm \
    --vllm_tensor_parallel_size 4
```

Terminal 2 -- training:
```bash
NPROC_PER_NODE=4 \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
megatron rlhf \
    --rlhf_type grpo \
    --model Qwen/Qwen2.5-7B-Instruct \
    --use_hf \
    --save_safetensors true \
    --vllm_mode server \
    --dataset /path/to/data.jsonl \
    --reward_funcs accuracy format \
    --num_generations 8 \
    --tensor_model_parallel_size 2 \
    --sequence_parallel true \
    --micro_batch_size 1 \
    --global_batch_size 8 \
    --output_dir megatron_output/grpo \
    --no_save_optim true \
    --no_save_rng true
```

## Megatron DPO

```bash
NPROC_PER_NODE=2 \
CUDA_VISIBLE_DEVICES=0,1 \
megatron rlhf \
    --rlhf_type dpo \
    --model Qwen/Qwen2.5-7B-Instruct \
    --use_hf \
    --save_safetensors true \
    --dataset /path/to/dpo_data.jsonl \
    --beta 0.1 \
    --tensor_model_parallel_size 2 \
    --sequence_parallel true \
    --micro_batch_size 4 \
    --global_batch_size 16 \
    --recompute_granularity full \
    --recompute_method uniform \
    --recompute_num_layers 1 \
    --finetune true \
    --cross_entropy_loss_fusion true \
    --lr 1e-5 \
    --num_train_epochs 1 \
    --output_dir megatron_output/dpo \
    --no_save_optim true \
    --no_save_rng true
```

## Megatron GKD (Knowledge Distillation)

Requires ms-swift >= 3.12.

```bash
NPROC_PER_NODE=2 \
CUDA_VISIBLE_DEVICES=0,1 \
megatron rlhf \
    --rlhf_type gkd \
    --model Qwen/Qwen2.5-7B-Instruct \
    --teacher_model Qwen/Qwen2.5-72B-Instruct \
    --use_hf \
    --save_safetensors true \
    --dataset /path/to/sft_data.jsonl \
    --beta 0.5 \
    --lmbda 0.5 \
    --temperature 0.9 \
    --max_completion_length 512 \
    --tensor_model_parallel_size 2 \
    --sequence_parallel true \
    --micro_batch_size 4 \
    --global_batch_size 16 \
    --finetune true \
    --cross_entropy_loss_fusion true \
    --lr 1e-5 \
    --num_train_epochs 1 \
    --output_dir megatron_output/gkd \
    --no_save_optim true \
    --no_save_rng true
```

## Post-Training: Inference After Megatron Training

With Mcore-Bridge (`--save_safetensors true`), checkpoints are saved in HF-compatible format. Inference is the same as standard swift:

```bash
# Full-param checkpoint
CUDA_VISIBLE_DEVICES=0 \
swift infer \
    --model megatron_output/model-name/vx-xxx/checkpoint-xxx \
    --stream true \
    --temperature 0 \
    --max_new_tokens 2048

# LoRA checkpoint
CUDA_VISIBLE_DEVICES=0 \
swift infer \
    --adapters megatron_output/model-name/vx-xxx/checkpoint-xxx \
    --stream true \
    --temperature 0 \
    --max_new_tokens 2048
```

## Traditional Workflow (Without Mcore-Bridge)

If you prefer explicit weight conversion:

### Step 1: HF to Megatron

```bash
CUDA_VISIBLE_DEVICES=0 \
swift export \
    --model Qwen/Qwen2.5-7B-Instruct \
    --use_hf \
    --to_mcore true \
    --torch_dtype bfloat16 \
    --output_dir Qwen2.5-7B-Instruct-mcore \
    --test_convert_precision true
```

### Step 2: Train with mcore_model

```bash
megatron sft \
    --mcore_model Qwen2.5-7B-Instruct-mcore \
    --save_safetensors false \
    ...
```

### Step 3: Megatron to HF

```bash
CUDA_VISIBLE_DEVICES=0 \
swift export \
    --mcore_model megatron_output/model/vx-xxx/checkpoint-xxx \
    --to_hf true \
    --torch_dtype bfloat16 \
    --output_dir megatron_output/model/vx-xxx/checkpoint-xxx-hf \
    --test_convert_precision true
```

## Key Parameters

### Parallelism

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--tensor_model_parallel_size` | TP degree | 1 |
| `--pipeline_model_parallel_size` | PP degree | 1 |
| `--expert_model_parallel_size` | EP degree (MoE) | 1 |
| `--expert_tensor_parallel_size` | Expert TP degree (MoE) | 1 |
| `--context_parallel_size` | CP degree | 1 |
| `--sequence_parallel` | Enable SP (needs TP) | false |
| `--use_distributed_optimizer` | ZeRO-1 | true |

### Batch Size

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--micro_batch_size` | Per-GPU batch size | 1 |
| `--global_batch_size` | Total batch = micro_bs * DP * grad_accum | 16 |

Formula: `DP = total_GPUs / (TP * PP * CP)`

### Recompute (Activation Checkpointing)

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--recompute_granularity` | `full` / `selective` / `none` | `selective` |
| `--recompute_method` | `uniform` / `block` (for `full` granularity) | None |
| `--recompute_num_layers` | Layers per recompute unit (for `full`) | None |
| `--recompute_modules` | Modules to recompute (for `selective`) | `["core_attn"]` |

### MoE-Specific

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--moe_grouped_gemm` | Use grouped GEMM for speed | true |
| `--moe_permute_fusion` | Fuse token permutation ops | false |
| `--moe_shared_expert_overlap` | Overlap shared expert with routing | false |
| `--moe_aux_loss_coeff` | Auxiliary loss coefficient (higher = more balanced, lower quality) | 0 |
| `--moe_expert_capacity_factor` | Capacity factor (drop tokens over capacity, speeds up training) | None |
| `--moe_router_dtype` | Router computation dtype | `fp32` |

### Learning Rate

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--lr` | Initial learning rate | 1e-5 (full) / 1e-4 (LoRA) |
| `--lr_decay_style` | Decay strategy | `cosine` |
| `--lr_warmup_fraction` | Warmup ratio | None |
| `--min_lr` | Minimum learning rate | 0 |

### Memory Optimization

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--optimizer_cpu_offload` | Offload optimizer to CPU | false |
| `--optimizer_offload_fraction` | Fraction of optimizer to offload | 1.0 |
| `--packing` | Pack sequences for balanced load | false |
| `--padding_free` | Flatten batch to avoid padding | true |

### Checkpoint

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--save_safetensors` | Save as safetensors (Mcore-Bridge) | true |
| `--save_steps` | Save interval | 500 |
| `--no_save_optim` | Don't save optimizer states | false |
| `--no_save_rng` | Don't save RNG states | false |
| `--save_total_limit` | Max checkpoints to keep | None (all) |
| `--finetune` | Reset iteration counter, skip optimizer/RNG loading | true |

### LoRA (Megatron)

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--tuner_type` | `lora` / `full` | `full` |
| `--lora_rank` | LoRA rank | 8 |
| `--lora_alpha` | LoRA alpha | 32 |
| `--target_modules` | Target modules (`all-linear`, or specific: `linear_qkv linear_proj linear_fc1 linear_fc2`) | `all-linear` |
| `--modules_to_save` | Extra modules to train/save (e.g., `word_embeddings output_layer`) | `[]` |
| `--merge_lora` | Merge LoRA on save | auto |

**Note on Megatron linear names**: `linear_proj` = `o_proj`, `linear_qkv` = `q/k/v_proj` combined, `linear_fc1` = `gate_proj + up_proj` combined, `linear_fc2` = `down_proj`.

## Performance Benchmarks

### Dense Model (Qwen2.5-14B, 8x A800, 8K context, full-param)

| Backend | Speed | Memory |
|---------|-------|--------|
| Megatron-LM | 9.04 s/it | 8x 64GB |
| DeepSpeed-ZeRO2 | 10.32 s/it | 8x 80GB |
| DeepSpeed-ZeRO3 | 10.56 s/it | 8x 58GB |

### MoE Model (Qwen3-30B-A3B, 16x A800, 8K context, full-param)

| Backend | Speed | Memory |
|---------|-------|--------|
| Megatron-LM | 9.6 s/it | 16x 60GB |
| DeepSpeed-ZeRO2 | OOM | OOM |
| DeepSpeed-ZeRO3 | 91.2 s/it | 16x 80GB |

## Parallelism Strategy Guide

- **DP** is fastest but uses more memory. Use other parallelism to reduce memory.
- **TP/EP**: High communication -- keep within NVLink domain (same node). Don't cross nodes.
- **PP/DP**: Better for cross-node communication.
- For **MoE**, use EP (not ETP) for speed. ETP saves more memory but is slower.
- Use `--overlap_grad_reduce true --overlap_param_gather true` to overlap DP communication.
- Use `--packing true` to improve GPU utilization (disable streaming when using packing).

## Troubleshooting

### Missing apex
Set `--gradient_accumulation_fusion false` to run without apex.

### Multi-node data inconsistency
Set `MODELSCOPE_CACHE` to a shared storage path. This ensures all nodes use the same dataset cache.

### OOM during weight conversion
Remove `CUDA_VISIBLE_DEVICES=0` to use multi-GPU conversion. Remove `--test_convert_precision true` if memory is insufficient.

### Attention backend compatibility
Some models (Llama4, GPT-OSS) don't support flash attention. Set `--attention_backend unfused --padding_free false`.

### transformer_engine install fails: missing cudnn.h / nccl.h
**Problem**: `pip install transformer_engine[pytorch]` fails with errors like `fatal error: cudnn.h: No such file or directory` or `nccl.h: No such file or directory`. Common on EC2 GPU instances that have NVIDIA drivers but no cuDNN/NCCL dev headers installed system-wide.
**Solution**: Point the build at the pip-installed nvidia packages inside the venv:
```bash
CUDA_HOME=/usr \
CUDNN_HOME=~/swift-env/lib/python3.11/site-packages/nvidia/cudnn \
CUDNN_PATH=~/swift-env/lib/python3.11/site-packages/nvidia/cudnn \
CPLUS_INCLUDE_PATH="~/swift-env/lib/python3.11/site-packages/nvidia/cudnn/include:~/swift-env/lib/python3.11/site-packages/nvidia/nccl/include" \
LIBRARY_PATH="~/swift-env/lib/python3.11/site-packages/nvidia/cudnn/lib:~/swift-env/lib/python3.11/site-packages/nvidia/nccl/lib" \
uv pip install "transformer-engine[pytorch]" megatron-core --python ~/swift-env/bin/python
```
Adjust `python3.11` to match your actual Python version. The `nvidia-cudnn-cu12` and `nvidia-nccl-cu12` pip packages contain the needed headers.

### LoRA + transformers 5.0 MoE issue
transformers 5.0 restructured MoE models, which may cause LoRA inference issues. Use `--merge_lora true` for MoE models (vLLM is not affected).
