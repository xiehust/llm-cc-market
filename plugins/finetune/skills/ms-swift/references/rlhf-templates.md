# RLHF Templates

## GRPO (Group Relative Policy Optimization)

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
swift rlhf \
    --rlhf_type grpo \
    --model Qwen/Qwen3-8B-Instruct \
    --dataset /path/to/grpo_data.jsonl \
    --torch_dtype bfloat16 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --learning_rate 5e-7 \
    --gradient_accumulation_steps 16 \
    --max_length 4096 \
    --output_dir output
```

GRPO data format (query + solution for reward):
```json
{"messages": [{"role": "user", "content": "Solve: 2+3=?"}], "solution": "5"}
```

## GRPO with vLLM Acceleration

```bash
swift rlhf \
    --rlhf_type grpo \
    --model Qwen/Qwen3-8B-Instruct \
    --infer_backend vllm \
    --dataset /path/to/data.jsonl \
    --torch_dtype bfloat16 \
    --num_train_epochs 1 \
    --output_dir output
```

## DPO (Direct Preference Optimization)

```bash
swift rlhf \
    --rlhf_type dpo \
    --model Qwen/Qwen3-8B-Instruct \
    --dataset /path/to/dpo_data.jsonl \
    --torch_dtype bfloat16 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --learning_rate 5e-7 \
    --gradient_accumulation_steps 8 \
    --max_length 4096 \
    --output_dir output
```

DPO data format:
```json
{"messages": [{"role": "user", "content": "问题"}], "chosen": [{"role": "assistant", "content": "好回答"}], "rejected": [{"role": "assistant", "content": "差回答"}]}
```

## KTO (Kahneman-Tversky Optimization)

```bash
swift rlhf \
    --rlhf_type kto \
    --model Qwen/Qwen3-8B-Instruct \
    --dataset /path/to/kto_data.jsonl \
    --torch_dtype bfloat16 \
    --num_train_epochs 1 \
    --output_dir output
```

KTO data format:
```json
{"messages": [{"role": "user", "content": "问题"}, {"role": "assistant", "content": "回答"}], "label": true}
```

## Reward Model Training

```bash
swift rlhf \
    --rlhf_type rm \
    --model Qwen/Qwen3-8B-Instruct \
    --dataset /path/to/rm_data.jsonl \
    --torch_dtype bfloat16 \
    --num_train_epochs 1 \
    --output_dir output
```

## Supported RLHF Algorithms

| Algorithm | `--rlhf_type` | Description |
|-----------|---------------|-------------|
| GRPO | `grpo` | Group Relative Policy Optimization |
| DAPO | `dapo` | Dynamic Advantage Policy Optimization |
| DPO | `dpo` | Direct Preference Optimization |
| KTO | `kto` | Kahneman-Tversky Optimization |
| CPO | `cpo` | Contrastive Preference Optimization |
| SimPO | `simpo` | Simple Preference Optimization |
| ORPO | `orpo` | Odds Ratio Preference Optimization |
| RM | `rm` | Reward Model |
| RLOO | `rloo` | REINFORCE Leave-One-Out |
