# Training Templates

## LoRA SFT (Single GPU, ~22GB VRAM) — HuggingFace

```bash
CUDA_VISIBLE_DEVICES=0 \
~/swift-env/bin/swift sft \
    --model Qwen/Qwen3-4B-Instruct-2507 \
    --use_hf \
    --train_type lora \
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

## LoRA SFT — ModelScope

```bash
CUDA_VISIBLE_DEVICES=0 \
~/swift-env/bin/swift sft \
    --model Qwen/Qwen3-8B-Instruct \
    --train_type lora \
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
    --model Qwen/Qwen3-8B-Instruct \
    --use_hf \
    --train_type lora \
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

## Full Parameter SFT (Multi-GPU + DeepSpeed)

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
~/swift-env/bin/swift sft \
    --model Qwen/Qwen3-8B-Instruct \
    --use_hf \
    --train_type full \
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
CUDA_VISIBLE_DEVICES=0,1,2,3 \
~/swift-env/bin/swift sft \
    --model Qwen/Qwen3-72B-Instruct \
    --use_hf \
    --train_type lora \
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

## Qwen3 Thinking Model SFT

```bash
CUDA_VISIBLE_DEVICES=0 \
~/swift-env/bin/swift sft \
    --model Qwen/Qwen3-8B \
    --use_hf \
    --train_type lora \
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
    --model Qwen/Qwen3-8B-Instruct \
    --use_hf \
    --train_type lora \
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
    --model Qwen/Qwen3-8B-Instruct \
    --use_hf \
    --train_type lora \
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
    --train_type lora \
    --dataset /path/to/multimodal_data.jsonl \
    --torch_dtype bfloat16 \
    --num_train_epochs 1 \
    --learning_rate 1e-4 \
    --max_length 4096 \
    --output_dir output
```
