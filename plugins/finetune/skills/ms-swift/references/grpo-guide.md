# GRPO & Reinforcement Learning Guide

Comprehensive guide for GRPO (Group Relative Policy Optimization) and related RL algorithms in ms-swift.

## GRPO Overview

GRPO generates multiple completions per prompt, scores them with reward functions, computes group-relative advantages, and updates the policy. No separate value model or reward model is required (unlike PPO).

## Basic GRPO Command

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 NPROC_PER_NODE=4 \
~/swift-env/bin/swift rlhf \
    --rlhf_type grpo \
    --model Qwen/Qwen3-8B-Instruct \
    --use_hf \
    --tuner_type lora \
    --dataset /path/to/grpo_data.jsonl \
    --reward_funcs accuracy format \
    --num_generations 8 \
    --max_completion_length 2048 \
    --torch_dtype bfloat16 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --learning_rate 5e-7 \
    --beta 0.04 \
    --gradient_accumulation_steps 16 \
    --output_dir output
```

## GRPO Data Format

Query-only format (model generates completions). Extra fields are passed to reward functions:

```jsonl
{"messages": [{"role": "user", "content": "Solve: what is 15 * 23?"}], "solution": "345"}
```

The `solution` field is used by the `accuracy` reward function. Any extra fields can be passed.

## GRPO with vLLM Acceleration

### Colocate Mode (single machine, default)
Generation and training share GPUs. Model weights are moved between training and inference.

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 NPROC_PER_NODE=4 \
~/swift-env/bin/swift rlhf \
    --rlhf_type grpo \
    --model Qwen/Qwen3-8B-Instruct \
    --use_hf \
    --use_vllm true \
    --vllm_mode colocate \
    --dataset /path/to/data.jsonl \
    --reward_funcs accuracy format \
    --num_generations 8 \
    --torch_dtype bfloat16 \
    --output_dir output
```

### Server Mode (separate generation and training)
Run a vLLM rollout server on dedicated GPUs, train on others.

Terminal 1 — rollout server:
```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 \
~/swift-env/bin/swift rollout \
    --model Qwen/Qwen3-8B-Instruct \
    --use_hf \
    --infer_backend vllm \
    --vllm_tensor_parallel_size 4
```

Terminal 2 — GRPO training:
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 NPROC_PER_NODE=4 \
~/swift-env/bin/swift rlhf \
    --rlhf_type grpo \
    --model Qwen/Qwen3-8B-Instruct \
    --use_hf \
    --use_vllm true \
    --vllm_mode server \
    --dataset /path/to/data.jsonl \
    --reward_funcs accuracy \
    --output_dir output
```

### Async Generation (overlaps generation with training)
```bash
~/swift-env/bin/swift rlhf \
    --rlhf_type grpo \
    --model Qwen/Qwen3-8B-Instruct \
    --use_vllm true --vllm_mode server \
    --async_generate true \
    --dataset /path/to/data.jsonl \
    --output_dir output
```

## Key GRPO Parameters

| Param | Description | Default |
|-------|-------------|---------|
| `--beta` | KL penalty coefficient | 0.04 |
| `--num_generations` | Completions per prompt (G) | 8 |
| `--num_iterations` | Update steps per batch (K) | 1 |
| `--epsilon` | Lower clipping coefficient | 0.2 |
| `--epsilon_high` | Upper clipping coefficient | none |
| `--max_completion_length` | Max generation tokens | 512 |
| `--temperature` | Sampling temperature | 0.9 |
| `--reward_funcs` | Space-separated reward function names | required |
| `--reward_weights` | Weights per reward function | equal |
| `--reward_model` | External reward model path(s) | none |
| `--ref_model` | Reference model (needed for full-param GRPO) | none |
| `--use_vllm` | Use vLLM for generation | false |
| `--vllm_mode` | `colocate` or `server` | colocate |
| `--async_generate` | Overlap generation/training (server mode) | false |
| `--steps_per_generation` | Optimization steps per generation | auto |
| `--advantage_estimator` | `grpo` / `rloo` / `reinforce_plus_plus` | `grpo` |
| `--scale_rewards` | `group` / `batch` / `none` / `gdpo` | `group` |
| `--log_completions` | Log completions to W&B | false |

## Built-in Reward Functions

Available via `--reward_funcs`:

| Function | Description | Extra Data Field |
|----------|-------------|-----------------|
| `accuracy` | DeepSeek-R1 style correctness check | `solution` |
| `format` | Format compliance checking | none |
| `cosine` | Length-aware cosine reward (from DAPO) | none |
| `repetition` | N-gram repetition penalty | none |
| `soft_overlong` | Soft penalty for overlong generations | none |

Example combining multiple:
```bash
--reward_funcs accuracy format repetition \
--reward_weights 1.0 0.5 0.3
```

## Custom Reward Functions

Create a plugin file (e.g., `my_rewards.py`):

```python
def my_reward_func(completions, solution=None, **kwargs):
    """Custom reward function.
    Args:
        completions: list of generated strings
        solution: from dataset's 'solution' field (if present)
        **kwargs: any other extra dataset fields
    Returns:
        list of float rewards (same length as completions)
    """
    rewards = []
    for completion in completions:
        if solution and solution in completion:
            rewards.append(1.0)
        else:
            rewards.append(0.0)
    return rewards
```

Register and use:
```bash
~/swift-env/bin/swift rlhf \
    --rlhf_type grpo \
    --external_plugins my_rewards.py \
    --reward_funcs my_reward_func accuracy \
    --reward_weights 1.0 0.5 \
    ...
```

## GRPO Algorithm Variants

### DAPO (Dynamic Advantage Policy Optimization)
From ByteDance. Adds dynamic sampling, overlong filtering, and soft cache length.

```bash
~/swift-env/bin/swift rlhf \
    --rlhf_type grpo \
    --loss_type bnpo \
    --dynamic_sample true \
    --overlong_filter true \
    --epsilon 0.2 --epsilon_high 0.28 \
    --reward_funcs accuracy soft_overlong \
    ...
```

### SAPO (Soft Adaptive Policy Optimization)
Replaces hard clipping with smooth temperature-controlled gating.

```bash
~/swift-env/bin/swift rlhf \
    --rlhf_type grpo \
    --loss_type sapo \
    --tau_pos 1.0 --tau_neg 1.05 \
    ...
```

### CISPO (Clipped Importance Sampling Policy Optimization)
Cross-instance sampling with gradient detachment.

```bash
~/swift-env/bin/swift rlhf \
    --rlhf_type grpo \
    --loss_type cispo \
    --epsilon 0.2 --epsilon_high 0.28 \
    ...
```

### RLOO (REINFORCE Leave-One-Out)
```bash
~/swift-env/bin/swift rlhf \
    --rlhf_type grpo \
    --advantage_estimator rloo \
    ...
```

### Reinforce++
```bash
~/swift-env/bin/swift rlhf \
    --rlhf_type grpo \
    --advantage_estimator reinforce_plus_plus \
    ...
```

### CHORD (with SFT expert data)
Mixes GRPO with SFT data from expert demonstrations.

```bash
~/swift-env/bin/swift rlhf \
    --rlhf_type grpo \
    --chord_sft_dataset /path/to/sft_data.jsonl \
    ...
```

## GRPO Loss Types

| `--loss_type` | Description |
|---------------|-------------|
| `grpo` | Standard (sample-level normalization) |
| `bnpo` | Batch Normalized (token-level, used by DAPO) |
| `dr_grpo` | Fixed dimension normalization (batch x max_length) |
| `dapo` | Global token normalization (multi-process) |
| `cispo` | CISPO loss with gradient detachment |
| `sapo` | Soft gating with temperature sigmoid |

## Multi-Turn GRPO (Agent Training)

For training agents that interact with environments/tools over multiple turns. Uses `MultiTurnScheduler`:

```bash
~/swift-env/bin/swift rlhf \
    --rlhf_type grpo \
    --multi_turn_scheduler my_scheduler \
    --external_plugins my_scheduler.py \
    --agent_template hermes \
    ...
```

The scheduler manages environment interaction, tool calls, and per-turn loss masking.

## GRPO Memory Optimization

| Technique | Param | Notes |
|-----------|-------|-------|
| LoRA training | `--tuner_type lora` | Much less memory than full |
| vLLM colocate | `--use_vllm true --vllm_mode colocate` | Shares GPU memory |
| Sleep levels | `--sleep_level 0/1/2` | Release GPU memory during phases |
| Optimizer offload | `--offload_optimizer true` | Offload to CPU |
| Model offload | `--offload_model true` | Offload to CPU |
| DeepSpeed ZeRO | `--deepspeed zero2` | Shard optimizer states |
| Reduce num_generations | `--num_generations 4` | Less memory for rollouts |

## GRPO Hybrid Mode (Large Models on Limited GPUs)

Train 72B model on 4x80GB GPUs using tensor parallelism with vLLM:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 NPROC_PER_NODE=4 \
~/swift-env/bin/swift rlhf \
    --rlhf_type grpo \
    --model Qwen/Qwen3-72B-Instruct \
    --use_hf \
    --tuner_type lora \
    --use_vllm true --vllm_mode colocate \
    --deepspeed zero3 \
    --dataset /path/to/data.jsonl \
    --reward_funcs accuracy format \
    --num_generations 8 \
    --max_completion_length 1024 \
    --output_dir output
```
