# Training Templates

## LoRA SFT (Single GPU, ~18GB VRAM) -- HuggingFace

```bash
CUDA_VISIBLE_DEVICES=0 \
~/swift-env/bin/swift sft \
    --model Qwen/Qwen3-4B-Instruct-2507 \
    --use_hf \
    --tuner_type lora \
    --dataset 'yahma/alpaca-cleaned#500' \
    --torch_dtype bfloat16 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --learning_rate 1e-4 \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --gradient_accumulation_steps 16 \
    --eval_steps 50 \
    --save_steps 50 \
    --save_total_limit 2 \
    --logging_steps 5 \
    --max_length 2048 \
    --output_dir output \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4
```

## LoRA SFT -- ModelScope

```bash
CUDA_VISIBLE_DEVICES=0 \
~/swift-env/bin/swift sft \
    --model Qwen/Qwen3-4B-Instruct-2507 \
    --tuner_type lora \
    --dataset 'AI-ModelScope/alpaca-gpt4-data-zh#500' \
              'AI-ModelScope/alpaca-gpt4-data-en#500' \
    --torch_dtype bfloat16 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --learning_rate 1e-4 \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --gradient_accumulation_steps 16 \
    --eval_steps 50 \
    --save_steps 50 \
    --save_total_limit 2 \
    --logging_steps 5 \
    --max_length 2048 \
    --output_dir output \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4
```

## QLoRA SFT (Single GPU, ~9GB VRAM)

```bash
CUDA_VISIBLE_DEVICES=0 \
~/swift-env/bin/swift sft \
    --model Qwen/Qwen3-4B-Instruct-2507 \
    --use_hf \
    --tuner_type lora \
    --quant_method bnb \
    --quant_bits 4 \
    --dataset 'yahma/alpaca-cleaned#500' \
    --torch_dtype bfloat16 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --learning_rate 1e-4 \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --gradient_accumulation_steps 16 \
    --max_length 2048 \
    --output_dir output
```

**Note**: QLoRA cannot merge weights -- use for prototyping only. For production, use LoRA, merge, then quantize with AWQ/GPTQ.

## Full Parameter SFT (Multi-GPU + DeepSpeed)

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 NPROC_PER_NODE=4 \
~/swift-env/bin/swift sft \
    --model Qwen/Qwen3-4B-Instruct-2507 \
    --use_hf \
    --tuner_type full \
    --dataset 'yahma/alpaca-cleaned#2000' \
    --torch_dtype bfloat16 \
    --deepspeed zero2 \
    --num_train_epochs 2 \
    --per_device_train_batch_size 2 \
    --learning_rate 5e-6 \
    --gradient_accumulation_steps 8 \
    --max_length 4096 \
    --output_dir output
```

## Multi-GPU LoRA (DeepSpeed ZeRO3)

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 NPROC_PER_NODE=4 \
~/swift-env/bin/swift sft \
    --model Qwen/Qwen3-72B-Instruct \
    --use_hf \
    --tuner_type lora \
    --deepspeed zero3 \
    --dataset 'yahma/alpaca-cleaned#1000' \
    --torch_dtype bfloat16 \
    --lora_rank 16 \
    --lora_alpha 64 \
    --target_modules all-linear \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 32 \
    --max_length 4096 \
    --output_dir output
```

## LoRA with Packing + Flash Attention (High Throughput)

```bash
CUDA_VISIBLE_DEVICES=0 \
~/swift-env/bin/swift sft \
    --model Qwen/Qwen3-4B-Instruct-2507 \
    --use_hf \
    --tuner_type lora \
    --dataset 'yahma/alpaca-cleaned#1000' \
    --torch_dtype bfloat16 \
    --packing true \
    --attn_impl flash_attn \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --max_length 4096 \
    --output_dir output
```

## LoRA with Liger Kernel (Speed + Memory Optimization)

```bash
CUDA_VISIBLE_DEVICES=0 \
~/swift-env/bin/swift sft \
    --model Qwen/Qwen3-4B-Instruct-2507 \
    --use_hf \
    --tuner_type lora \
    --use_liger_kernel true \
    --dataset 'yahma/alpaca-cleaned#1000' \
    --torch_dtype bfloat16 \
    --output_dir output
```

## FSDP2 Training

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 NPROC_PER_NODE=4 \
~/swift-env/bin/swift sft \
    --model Qwen/Qwen3-4B-Instruct-2507 \
    --use_hf \
    --tuner_type full \
    --fsdp fsdp2 \
    --dataset 'yahma/alpaca-cleaned#2000' \
    --torch_dtype bfloat16 \
    --output_dir output
```

## Qwen3 Thinking Model SFT

```bash
CUDA_VISIBLE_DEVICES=0 \
~/swift-env/bin/swift sft \
    --model Qwen/Qwen3-4B-Instruct-2507 \
    --use_hf \
    --tuner_type lora \
    --enable_thinking true \
    --dataset 'yahma/alpaca-cleaned#500' \
    --torch_dtype bfloat16 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --learning_rate 1e-4 \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --gradient_accumulation_steps 16 \
    --max_length 4096 \
    --output_dir output
```

## Self-Cognition Training (Custom Identity)

```bash
~/swift-env/bin/swift sft \
    --model Qwen/Qwen3-4B-Instruct-2507 \
    --use_hf \
    --tuner_type lora \
    --dataset 'yahma/alpaca-cleaned#500' \
    --torch_dtype bfloat16 \
    --num_train_epochs 2 \
    --learning_rate 1e-4 \
    --lora_rank 8 \
    --max_length 2048 \
    --output_dir output \
    --model_author "YourName" \
    --model_name "YourBot"
```

## Custom Dataset

Prepare data in messages format:
```json
{"messages": [{"role": "system", "content": "You are a helpful assistant"}, {"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hello! How can I help you?"}]}
```

```bash
~/swift-env/bin/swift sft \
    --model Qwen/Qwen3-4B-Instruct-2507 \
    --use_hf \
    --tuner_type lora \
    --dataset /path/to/train.jsonl \
    --torch_dtype bfloat16 \
    --num_train_epochs 3 \
    --learning_rate 1e-4 \
    --output_dir output
```

## Multimodal (Vision) SFT

```bash
~/swift-env/bin/swift sft \
    --model Qwen/Qwen3-VL-8B-Instruct \
    --use_hf \
    --tuner_type lora \
    --dataset /path/to/multimodal_data.jsonl \
    --torch_dtype bfloat16 \
    --freeze_vit true \
    --freeze_aligner true \
    --num_train_epochs 1 \
    --learning_rate 1e-4 \
    --max_length 4096 \
    --max_pixels 1003520 \
    --output_dir output
```

## Embedding Model Training

```bash
~/swift-env/bin/swift sft \
    --model BAAI/bge-large-en-v1.5 \
    --use_hf \
    --task_type embedding \
    --tuner_type lora \
    --dataset /path/to/embedding_data.jsonl \
    --output_dir output
```

## Sequence Classification Training

```bash
~/swift-env/bin/swift sft \
    --model Qwen/Qwen3-4B-Instruct-2507 \
    --use_hf \
    --task_type seq_cls \
    --num_labels 3 \
    --tuner_type lora \
    --dataset /path/to/classification_data.jsonl \
    --output_dir output
```

## Long Sequence Training with Sequence Parallelism

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 NPROC_PER_NODE=4 \
~/swift-env/bin/swift sft \
    --model Qwen/Qwen3-4B-Instruct-2507 \
    --use_hf \
    --tuner_type lora \
    --sequence_parallel_size 4 \
    --attn_impl flash_attn \
    --dataset /path/to/long_text.jsonl \
    --max_length 65536 \
    --output_dir output
```

## Pre-training (Continual)

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 NPROC_PER_NODE=4 \
~/swift-env/bin/swift pt \
    --model Qwen/Qwen3-4B-Instruct-2507 \
    --use_hf \
    --tuner_type lora \
    --dataset /path/to/pretrain_data.jsonl \
    --torch_dtype bfloat16 \
    --deepspeed zero2 \
    --max_length 4096 \
    --truncation_strategy split \
    --streaming true \
    --max_steps 10000 \
    --output_dir output
```

## YAML Config File

Any command can use a YAML config instead of CLI args:

```yaml
# config.yaml
model: Qwen/Qwen3-4B-Instruct-2507
use_hf: true
tuner_type: lora
dataset:
  - 'yahma/alpaca-cleaned#500'
torch_dtype: bfloat16
num_train_epochs: 1
per_device_train_batch_size: 1
learning_rate: 1e-4
lora_rank: 8
lora_alpha: 32
target_modules: all-linear
gradient_accumulation_steps: 16
max_length: 2048
output_dir: output
```

Usage: `~/swift-env/bin/swift sft --config config.yaml`
