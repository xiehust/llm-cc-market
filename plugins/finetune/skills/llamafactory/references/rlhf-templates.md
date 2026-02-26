# RLHF / Preference Training Templates

## Supported Alignment Algorithms

| Algorithm | `stage` | `pref_loss` | Data Format | Ref Model |
|-----------|---------|-------------|-------------|-----------|
| DPO | `dpo` | `sigmoid` | chosen + rejected | Yes (auto) |
| Hinge DPO | `dpo` | `hinge` | chosen + rejected | Yes |
| IPO | `dpo` | `ipo` | chosen + rejected | Yes |
| ORPO | `dpo` | `orpo` | chosen + rejected | No |
| SimPO | `dpo` | `simpo` | chosen + rejected | No |
| KTO (paired) | `dpo` | `kto_pair` | chosen + rejected | Yes |
| KTO | `kto` | -- | response + kto_tag | Yes (auto) |
| PPO | `ppo` | -- | query only | Yes + reward model |
| Reward Model | `rm` | -- | chosen + rejected | -- |

Note: ORPO and SimPO do not require a reference model. All others use the base model as reference automatically (or specify `ref_model`).

## DPO (Direct Preference Optimization)

### Data Format
Preference data with chosen and rejected responses. Register with `ranking: true` in `dataset_info.json`.

**Alpaca format:**
```json
[{"instruction": "Q", "input": "", "chosen": "good answer", "rejected": "bad answer"}]
```

**ShareGPT format:**
```json
[{"conversations": [{"from": "human", "value": "Q"}], "chosen": {"from": "gpt", "value": "good"}, "rejected": {"from": "gpt", "value": "bad"}}]
```

### Standard DPO Config
```yaml
model_name_or_path: Qwen/Qwen3-8B-Instruct
trust_remote_code: true

stage: dpo
do_train: true
finetuning_type: lora
lora_rank: 8
lora_target: all

pref_beta: 0.1
pref_loss: sigmoid

dataset: dpo_en_demo
template: qwen3_nothink
cutoff_len: 2048

output_dir: saves/qwen3-8b/lora/dpo
per_device_train_batch_size: 1
gradient_accumulation_steps: 8
learning_rate: 5.0e-6
num_train_epochs: 3.0
lr_scheduler_type: cosine
warmup_ratio: 0.1
bf16: true
logging_steps: 10
save_steps: 500
plot_loss: true
```

### DPO Parameters
| Param | Description | Default |
|-------|-------------|---------|
| `pref_beta` | KL regularization coefficient | 0.1 |
| `pref_loss` | Loss variant: `sigmoid`/`hinge`/`ipo`/`kto_pair`/`orpo`/`simpo` | `sigmoid` |
| `pref_ftx` | SFT loss coefficient mixed into DPO | 0.0 |
| `dpo_label_smoothing` | Label smoothing (cDPO, 0 to 0.5) | 0.0 |
| `ld_alpha` | LD-DPO alpha for verbose token weighting | none |
| `ref_model` | Explicit reference model path (auto if omitted) | none |

## ORPO (Odds Ratio Preference Optimization)

No reference model needed. Same data format as DPO.

```yaml
stage: dpo
pref_loss: orpo
pref_beta: 0.1
# ... rest same as DPO config
```

## SimPO (Simple Preference Optimization)

No reference model needed. Same data format as DPO.

```yaml
stage: dpo
pref_loss: simpo
pref_beta: 2.0
simpo_gamma: 0.5           # Target reward margin
# ... rest same as DPO config
```

## KTO (Kahneman-Tversky Optimization)

Uses binary feedback (desirable/undesirable) instead of paired preferences.

### Data Format
**ShareGPT format with kto_tag:**
```json
[{"conversations": [{"from": "human", "value": "Q"}, {"from": "gpt", "value": "A"}], "kto_tag": true}]
```
`kto_tag: true` = desirable, `kto_tag: false` = undesirable.

### KTO Config
```yaml
model_name_or_path: Qwen/Qwen3-8B-Instruct
trust_remote_code: true

stage: kto
do_train: true
finetuning_type: lora
lora_rank: 8
lora_target: all

pref_beta: 0.1
kto_chosen_weight: 1.0
kto_rejected_weight: 1.0

dataset: kto_en_demo
template: qwen3_nothink
cutoff_len: 2048

output_dir: saves/qwen3-8b/lora/kto
per_device_train_batch_size: 1
gradient_accumulation_steps: 8
learning_rate: 5.0e-6
num_train_epochs: 3.0
bf16: true
```

### KTO Parameters
| Param | Description | Default |
|-------|-------------|---------|
| `pref_beta` | KL divergence weight | 0.1 |
| `kto_chosen_weight` | Weight for desirable examples | 1.0 |
| `kto_rejected_weight` | Weight for undesirable examples | 1.0 |

## Reward Model Training

Train a reward model from preference data. Used as input for PPO.

```yaml
model_name_or_path: Qwen/Qwen3-8B-Instruct
trust_remote_code: true

stage: rm
do_train: true
finetuning_type: lora
lora_rank: 8
lora_target: all

dataset: dpo_en_demo
template: qwen3_nothink
cutoff_len: 2048

output_dir: saves/qwen3-8b/lora/rm
per_device_train_batch_size: 1
gradient_accumulation_steps: 8
learning_rate: 5.0e-6
num_train_epochs: 1.0
bf16: true
```

## PPO (Proximal Policy Optimization)

Requires a trained reward model. Uses a 4-model system: policy, reference, reward, and value model.

### Data Format
Query-only (model generates completions during training):
```json
[{"instruction": "Write a poem about AI", "input": "", "output": ""}]
```

### PPO Config
```yaml
model_name_or_path: Qwen/Qwen3-8B-Instruct
trust_remote_code: true

stage: ppo
do_train: true
finetuning_type: lora
lora_rank: 8
lora_target: all

reward_model: saves/qwen3-8b/lora/rm
reward_model_type: lora

dataset: alpaca_en_demo
template: qwen3_nothink
cutoff_len: 2048

output_dir: saves/qwen3-8b/lora/ppo
per_device_train_batch_size: 1
gradient_accumulation_steps: 8
learning_rate: 1.0e-6
num_train_epochs: 1.0
bf16: true
report_to: wandb            # PPO only supports wandb or tensorboard
```

### PPO Parameters
| Param | Description | Default |
|-------|-------------|---------|
| `reward_model` | Path to trained reward model | required |
| `reward_model_type` | `lora` / `full` / `api` | `lora` |
| `reward_model_adapters` | Reward model adapter path | none |
| `ppo_epochs` | PPO update epochs per batch | 4 |
| `ppo_buffer_size` | Mini-batches for experience buffer | 1 |
| `ppo_target` | Target KL for adaptive control | 6.0 |
| `ppo_score_norm` | Score normalization | false |
| `ppo_whiten_rewards` | Whiten rewards before advantages | false |

## Multimodal DPO

Same DPO config but with multimodal model and data containing images:

```yaml
model_name_or_path: Qwen/Qwen3-VL-8B-Instruct
trust_remote_code: true

stage: dpo
do_train: true
finetuning_type: lora
lora_rank: 8
lora_target: all
freeze_vision_tower: true

pref_beta: 0.1
pref_loss: sigmoid

dataset: mllm_dpo_demo
template: qwen3_vl_nothink
image_max_pixels: 262144

output_dir: saves/qwen3-vl-8b/lora/dpo
per_device_train_batch_size: 1
gradient_accumulation_steps: 8
learning_rate: 5.0e-6
num_train_epochs: 3.0
bf16: true
```

## DPO with DeepSpeed (Multi-GPU)

```yaml
model_name_or_path: Qwen/Qwen3-8B-Instruct
trust_remote_code: true

stage: dpo
do_train: true
finetuning_type: lora
lora_rank: 8
lora_target: all
deepspeed: examples/deepspeed/ds_z2_config.json

pref_beta: 0.1
pref_loss: sigmoid

dataset: dpo_en_demo
template: qwen3_nothink

output_dir: saves/qwen3-8b/lora/dpo-ds
per_device_train_batch_size: 1
gradient_accumulation_steps: 4
learning_rate: 5.0e-6
num_train_epochs: 3.0
bf16: true
```

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 FORCE_TORCHRUN=1 \
    ~/lmf-env/bin/llamafactory-cli train dpo_config.yaml
```
