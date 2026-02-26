# RLHF / Preference Training Templates

For GRPO and its variants (DAPO, SAPO, CISPO, etc.), see `references/grpo-guide.md`.

## Supported RLHF Algorithms

| Algorithm | `--rlhf_type` | Data Format | Description |
|-----------|---------------|-------------|-------------|
| DPO | `dpo` | chosen + rejected | Direct Preference Optimization |
| KTO | `kto` | response + label | Kahneman-Tversky Optimization |
| CPO | `cpo` | chosen + rejected | Contrastive Preference Optimization |
| SimPO | `simpo` | chosen + rejected | Simple Preference Optimization |
| ORPO | `orpo` | chosen + rejected | Odds Ratio Preference Optimization |
| RM | `rm` | chosen + rejected | Reward Model training |
| PPO | `ppo` | query only | Proximal Policy Optimization (4-model system) |
| GKD | `gkd` | SFT or query-only | Generalized Knowledge Distillation |
| GRPO | `grpo` | query only + rewards | Group Relative Policy Optimization (see grpo-guide.md) |

## DPO (Direct Preference Optimization)

### Data Format
```jsonl
{"messages": [{"role": "user", "content": "What is AI?"}], "rejected_response": "I don't know."}
```
The last assistant message in `messages` is the chosen response. `rejected_response` is the rejected one.

### Single GPU
```bash
CUDA_VISIBLE_DEVICES=0 \
~/swift-env/bin/swift rlhf \
    --rlhf_type dpo \
    --model Qwen/Qwen3-8B-Instruct \
    --use_hf \
    --tuner_type lora \
    --dataset /path/to/dpo_data.jsonl \
    --torch_dtype bfloat16 \
    --beta 0.1 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --learning_rate 5e-7 \
    --gradient_accumulation_steps 8 \
    --max_length 4096 \
    --output_dir output
```

### Multi-GPU with DeepSpeed
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 NPROC_PER_NODE=4 \
~/swift-env/bin/swift rlhf \
    --rlhf_type dpo \
    --model Qwen/Qwen3-8B-Instruct \
    --use_hf \
    --tuner_type lora \
    --deepspeed zero2 \
    --dataset /path/to/dpo_data.jsonl \
    --torch_dtype bfloat16 \
    --beta 0.1 \
    --output_dir output
```

### DPO Parameters
| Param | Description | Default |
|-------|-------------|---------|
| `--beta` | KL regularization coefficient | 0.1 |
| `--loss_type` | Variant: sigmoid, hinge, ipo, etc. | sigmoid |
| `--label_smoothing` | Label smoothing factor | 0 |
| `--rpo_alpha` | RPO NLL term weight | 0 |
| `--ref_model` | Reference model (needed for full-param DPO) | auto (base model) |

## KTO (Kahneman-Tversky Optimization)

### Data Format
```jsonl
{"messages": [{"role": "user", "content": "Question"}, {"role": "assistant", "content": "Answer"}], "label": true}
```
`label: true` = desirable, `label: false` = undesirable. Does not need paired data.

### Command
```bash
CUDA_VISIBLE_DEVICES=0 \
~/swift-env/bin/swift rlhf \
    --rlhf_type kto \
    --model Qwen/Qwen3-8B-Instruct \
    --use_hf \
    --tuner_type lora \
    --dataset /path/to/kto_data.jsonl \
    --torch_dtype bfloat16 \
    --beta 0.1 \
    --num_train_epochs 1 \
    --output_dir output
```

### KTO Parameters
| Param | Description | Default |
|-------|-------------|---------|
| `--beta` | KL divergence weight | 0.1 |
| `--desirable_weight` | Weight for positive examples | 1.0 |
| `--undesirable_weight` | Weight for negative examples | 1.0 |

## CPO (Contrastive Preference Optimization)

Uses same data format as DPO (chosen + rejected). No reference model needed.

```bash
CUDA_VISIBLE_DEVICES=0 \
~/swift-env/bin/swift rlhf \
    --rlhf_type cpo \
    --model Qwen/Qwen3-8B-Instruct \
    --use_hf \
    --tuner_type lora \
    --dataset /path/to/dpo_data.jsonl \
    --torch_dtype bfloat16 \
    --cpo_alpha 1.0 \
    --output_dir output
```

## SimPO (Simple Preference Optimization)

Uses same data format as DPO. No reference model needed.

```bash
CUDA_VISIBLE_DEVICES=0 \
~/swift-env/bin/swift rlhf \
    --rlhf_type simpo \
    --model Qwen/Qwen3-8B-Instruct \
    --use_hf \
    --tuner_type lora \
    --dataset /path/to/dpo_data.jsonl \
    --torch_dtype bfloat16 \
    --simpo_gamma 1.0 \
    --beta 2.0 \
    --output_dir output
```

## ORPO (Odds Ratio Preference Optimization)

Uses same data format as DPO. No reference model needed.

```bash
CUDA_VISIBLE_DEVICES=0 \
~/swift-env/bin/swift rlhf \
    --rlhf_type orpo \
    --model Qwen/Qwen3-8B-Instruct \
    --use_hf \
    --tuner_type lora \
    --dataset /path/to/dpo_data.jsonl \
    --torch_dtype bfloat16 \
    --beta 0.1 \
    --output_dir output
```

## Reward Model Training

### Data Format
Same as DPO (chosen + rejected pairs).

### Command
```bash
CUDA_VISIBLE_DEVICES=0 \
~/swift-env/bin/swift rlhf \
    --rlhf_type rm \
    --model Qwen/Qwen3-8B-Instruct \
    --use_hf \
    --tuner_type lora \
    --dataset /path/to/rm_data.jsonl \
    --torch_dtype bfloat16 \
    --num_train_epochs 1 \
    --output_dir output
```

## PPO (Proximal Policy Optimization)

4-model system: policy model, reference model, reward model, and value model.

### Data Format
Query-only (model generates completions):
```jsonl
{"messages": [{"role": "user", "content": "Write a poem about AI"}]}
```

### Command
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 NPROC_PER_NODE=4 \
~/swift-env/bin/swift rlhf \
    --rlhf_type ppo \
    --model Qwen/Qwen3-8B-Instruct \
    --use_hf \
    --tuner_type lora \
    --reward_model /path/to/reward_model \
    --dataset /path/to/ppo_data.jsonl \
    --torch_dtype bfloat16 \
    --num_ppo_epochs 4 \
    --kl_coef 0.05 \
    --cliprange 0.2 \
    --max_completion_length 512 \
    --output_dir output
```

### PPO Parameters
| Param | Description | Default |
|-------|-------------|---------|
| `--reward_model` | Path to trained reward model | required |
| `--num_ppo_epochs` | PPO update epochs per batch | 4 |
| `--kl_coef` | KL coefficient | 0.05 |
| `--cliprange` | Policy clip range | 0.2 |
| `--vf_coef` | Value function coefficient | 0.1 |
| `--cliprange_value` | Value clip range | 0.2 |
| `--gamma` | Discount factor | 1.0 |
| `--lam` | GAE lambda | 0.95 |
| `--whiten_rewards` | Reward normalization | false |
| `--max_completion_length` | Max generation length | 512 |

## GKD (Generalized Knowledge Distillation)

Distills a teacher model into a student model. Supports text and multimodal.

### Data Format
Same as SFT format. With `--seq_kd true`, can use prompts-only format.

### Command
```bash
CUDA_VISIBLE_DEVICES=0 \
~/swift-env/bin/swift rlhf \
    --rlhf_type gkd \
    --model Qwen/Qwen3-4B-Instruct \
    --use_hf \
    --teacher_model Qwen/Qwen3-72B-Instruct \
    --tuner_type lora \
    --dataset /path/to/sft_data.jsonl \
    --torch_dtype bfloat16 \
    --beta 0.5 \
    --lmbda 0.5 \
    --max_completion_length 512 \
    --output_dir output
```

### GKD Parameters
| Param | Description | Default |
|-------|-------------|---------|
| `--teacher_model` | Teacher model path | required |
| `--beta` | Divergence interpolation (0=ForwardKL, 0.5=JSD, 1=ReverseKL) | 0.5 |
| `--lmbda` | On-policy probability (0=dataset, 1=student-generated) | 0.5 |
| `--seq_kd` | Use teacher-generated sequences | false |
| `--sft_alpha` | Mix SFT loss proportion | 0 |
| `--max_completion_length` | Max generation tokens | 512 |

## Multimodal RLHF

All preference algorithms (DPO, KTO, CPO, SimPO, ORPO, RM, GRPO) support multimodal models. Add media fields to the data:

```jsonl
{"messages": [{"role": "user", "content": "<image>Which caption is better?"}, {"role": "assistant", "content": "Caption A is more accurate."}], "rejected_response": "Caption B is better.", "images": ["/path/to/img.jpg"]}
```

```bash
~/swift-env/bin/swift rlhf \
    --rlhf_type dpo \
    --model Qwen/Qwen3-VL-8B-Instruct \
    --use_hf \
    --tuner_type lora \
    --freeze_vit true \
    --dataset /path/to/multimodal_dpo.jsonl \
    --output_dir output
```
