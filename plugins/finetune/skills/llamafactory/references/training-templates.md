# Training Templates

All training in LLaMA-Factory uses YAML config files passed to `llamafactory-cli train`.

## LoRA SFT (Single GPU, ~16GB VRAM)

```yaml
# qwen3_lora_sft.yaml
model_name_or_path: Qwen/Qwen3-4B-Instruct-2507
trust_remote_code: true

stage: sft
do_train: true
finetuning_type: lora
lora_rank: 8
lora_target: all

dataset: alpaca_en_demo
template: qwen3_nothink
cutoff_len: 2048
max_samples: 1000
preprocessing_num_workers: 16

output_dir: saves/qwen3-8b/lora/sft
logging_steps: 10
save_steps: 500
save_total_limit: 2
plot_loss: true
overwrite_output_dir: true

per_device_train_batch_size: 1
gradient_accumulation_steps: 8
learning_rate: 1.0e-4
num_train_epochs: 3.0
lr_scheduler_type: cosine
warmup_ratio: 0.1
bf16: true
```

```bash
CUDA_VISIBLE_DEVICES=0 ~/lmf-env/bin/llamafactory-cli train qwen3_lora_sft.yaml
```

## QLoRA SFT (Single GPU, ~6GB VRAM)

```yaml
# qwen3_qlora_sft.yaml
model_name_or_path: Qwen/Qwen3-4B-Instruct-2507
trust_remote_code: true

stage: sft
do_train: true
finetuning_type: lora
lora_rank: 8
lora_target: all
quantization_bit: 4
quantization_method: bnb
quantization_type: nf4
double_quantization: true

dataset: alpaca_en_demo
template: qwen3_nothink
cutoff_len: 2048

output_dir: saves/qwen3-8b/qlora/sft
per_device_train_batch_size: 1
gradient_accumulation_steps: 8
learning_rate: 1.0e-4
num_train_epochs: 3.0
lr_scheduler_type: cosine
warmup_ratio: 0.1
bf16: true
logging_steps: 10
save_steps: 500
plot_loss: true
```

## Full Parameter SFT (Multi-GPU + DeepSpeed)

```yaml
# qwen3_full_sft.yaml
model_name_or_path: Qwen/Qwen3-4B-Instruct-2507
trust_remote_code: true

stage: sft
do_train: true
finetuning_type: full
deepspeed: examples/deepspeed/ds_z3_config.json

dataset: alpaca_en_demo
template: qwen3_nothink
cutoff_len: 4096

output_dir: saves/qwen3-8b/full/sft
per_device_train_batch_size: 2
gradient_accumulation_steps: 4
learning_rate: 1.0e-5
num_train_epochs: 2.0
lr_scheduler_type: cosine
warmup_ratio: 0.1
bf16: true
logging_steps: 10
save_steps: 500
```

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 FORCE_TORCHRUN=1 \
    ~/lmf-env/bin/llamafactory-cli train qwen3_full_sft.yaml
```

## LoRA SFT with Unsloth (170% Speed)

```yaml
model_name_or_path: Qwen/Qwen3-4B-Instruct-2507
trust_remote_code: true
use_unsloth: true

stage: sft
do_train: true
finetuning_type: lora
lora_rank: 8
lora_target: all

dataset: alpaca_en_demo
template: qwen3_nothink
cutoff_len: 2048

output_dir: saves/qwen3-8b/lora/sft-unsloth
per_device_train_batch_size: 2
gradient_accumulation_steps: 4
learning_rate: 1.0e-4
num_train_epochs: 3.0
bf16: true
```

Note: Unsloth is incompatible with DeepSpeed ZeRO-3. Use ZeRO-2 or single-GPU.

## LoRA with Sequence Packing (High Throughput)

```yaml
model_name_or_path: Qwen/Qwen3-4B-Instruct-2507
trust_remote_code: true
flash_attn: fa2

stage: sft
do_train: true
finetuning_type: lora
lora_rank: 8
lora_target: all

dataset: alpaca_en_demo
template: qwen3_nothink
cutoff_len: 4096
neat_packing: true

output_dir: saves/qwen3-8b/lora/sft-packed
per_device_train_batch_size: 1
gradient_accumulation_steps: 8
learning_rate: 1.0e-4
num_train_epochs: 3.0
bf16: true
```

## LoRA with Liger Kernel

```yaml
model_name_or_path: Qwen/Qwen3-4B-Instruct-2507
trust_remote_code: true
enable_liger_kernel: true

stage: sft
do_train: true
finetuning_type: lora
lora_rank: 8
lora_target: all

dataset: alpaca_en_demo
template: qwen3_nothink
output_dir: saves/qwen3-8b/lora/sft-liger
per_device_train_batch_size: 1
gradient_accumulation_steps: 8
learning_rate: 1.0e-4
num_train_epochs: 3.0
bf16: true
```

## DoRA (Weight-Decomposed LoRA)

```yaml
finetuning_type: lora
lora_rank: 8
lora_target: all
use_dora: true
# ... rest same as LoRA SFT
```

## LoRA+ (Different LR for A/B Matrices)

```yaml
finetuning_type: lora
lora_rank: 8
lora_target: all
loraplus_lr_ratio: 16.0
# ... rest same as LoRA SFT
```

## PiSSA (Principal Singular Value Adaptation)

```yaml
finetuning_type: lora
lora_rank: 8
lora_target: all
pissa_init: true
pissa_iter: 16
# ... rest same as LoRA SFT
```

## OFT (Orthogonal Fine-Tuning, v0.9.4+)

```yaml
model_name_or_path: Qwen/Qwen3-4B-Instruct-2507
trust_remote_code: true

stage: sft
do_train: true
finetuning_type: oft
oft_rank: 8
oft_block_size: 32
oft_target: all

dataset: alpaca_en_demo
template: qwen3_nothink
output_dir: saves/qwen3-8b/oft/sft
per_device_train_batch_size: 1
gradient_accumulation_steps: 8
learning_rate: 1.0e-4
num_train_epochs: 3.0
bf16: true
```

## Freeze (Partial-Parameter) SFT

```yaml
model_name_or_path: Qwen/Qwen3-4B-Instruct-2507
trust_remote_code: true

stage: sft
do_train: true
finetuning_type: freeze
freeze_trainable_layers: 2      # Last 2 layers trainable
freeze_trainable_modules: all

dataset: alpaca_en_demo
template: qwen3_nothink
output_dir: saves/qwen3-8b/freeze/sft
per_device_train_batch_size: 1
gradient_accumulation_steps: 8
learning_rate: 1.0e-5
num_train_epochs: 3.0
bf16: true
```

## Pre-training (Continual)

```yaml
model_name_or_path: Qwen/Qwen3-4B-Instruct-2507
trust_remote_code: true

stage: pt
do_train: true
finetuning_type: lora
lora_rank: 8
lora_target: all

dataset: c4_demo
template: qwen3
cutoff_len: 4096

output_dir: saves/qwen3-8b/lora/pt
per_device_train_batch_size: 1
gradient_accumulation_steps: 16
learning_rate: 1.0e-4
num_train_epochs: 1.0
bf16: true
```

## Multimodal (Vision-Language) SFT

```yaml
model_name_or_path: Qwen/Qwen3-VL-8B-Instruct
trust_remote_code: true

stage: sft
do_train: true
finetuning_type: lora
lora_rank: 8
lora_target: all
freeze_vision_tower: true
freeze_multi_modal_projector: true

dataset: mllm_demo
template: qwen3_vl_nothink
cutoff_len: 4096
image_max_pixels: 262144
video_max_pixels: 16384

output_dir: saves/qwen3-vl-8b/lora/sft
per_device_train_batch_size: 1
gradient_accumulation_steps: 8
learning_rate: 1.0e-4
num_train_epochs: 3.0
bf16: true
```

## Multi-GPU LoRA with DeepSpeed ZeRO-2

```yaml
model_name_or_path: Qwen/Qwen3-72B-Instruct
trust_remote_code: true

stage: sft
do_train: true
finetuning_type: lora
lora_rank: 16
lora_alpha: 32
lora_target: all
deepspeed: examples/deepspeed/ds_z2_config.json

dataset: alpaca_en_demo
template: qwen3_nothink
cutoff_len: 4096

output_dir: saves/qwen3-72b/lora/sft
per_device_train_batch_size: 1
gradient_accumulation_steps: 16
learning_rate: 1.0e-4
num_train_epochs: 1.0
bf16: true
```

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 FORCE_TORCHRUN=1 \
    ~/lmf-env/bin/llamafactory-cli train config.yaml
```

## GaLore Full Fine-Tuning (Memory-Efficient Full Params)

```yaml
model_name_or_path: Qwen/Qwen3-4B-Instruct-2507
trust_remote_code: true

stage: sft
do_train: true
finetuning_type: full
use_galore: true
galore_layerwise: true
galore_target: all
galore_rank: 128
galore_scale: 2.0
pure_bf16: true

dataset: alpaca_en_demo
template: qwen3_nothink
output_dir: saves/qwen3-8b/full/galore
per_device_train_batch_size: 1
gradient_accumulation_steps: 1
learning_rate: 1.0e-5
num_train_epochs: 2.0
```

## Custom Dataset with Evaluation

```yaml
model_name_or_path: Qwen/Qwen3-4B-Instruct-2507
trust_remote_code: true

stage: sft
do_train: true
finetuning_type: lora
lora_rank: 8
lora_target: all

dataset: my_custom_data
dataset_dir: /path/to/data/folder
template: qwen3_nothink
cutoff_len: 2048

val_size: 0.1
eval_strategy: steps
eval_steps: 500
per_device_eval_batch_size: 1

output_dir: saves/qwen3-8b/lora/custom
per_device_train_batch_size: 1
gradient_accumulation_steps: 8
learning_rate: 1.0e-4
num_train_epochs: 3.0
bf16: true
logging_steps: 10
save_steps: 500
plot_loss: true
load_best_model_at_end: true
metric_for_best_model: eval_loss
```

## Qwen3 Thinking Model SFT

```yaml
model_name_or_path: Qwen/Qwen3-4B-Instruct-2507
trust_remote_code: true

stage: sft
do_train: true
finetuning_type: lora
lora_rank: 8
lora_target: all

dataset: alpaca_en_demo
template: qwen3              # NOT qwen3_nothink for thinking models
enable_thinking: true         # Enable CoT reasoning
cutoff_len: 4096

output_dir: saves/qwen3-8b/lora/sft-thinking
per_device_train_batch_size: 1
gradient_accumulation_steps: 8
learning_rate: 1.0e-4
num_train_epochs: 3.0
bf16: true
```

## Common Templates

| Model Family | Template (standard) | Template (no-think) |
|-------------|--------------------|--------------------|
| Qwen3 | `qwen3` | `qwen3_nothink` |
| Qwen3-VL | `qwen3_vl` | `qwen3_vl_nothink` |
| Llama 3/3.1/3.2/3.3 | `llama3` | -- |
| Llama 4 | `llama4` | -- |
| DeepSeek-R1 | `deepseek3` | -- |
| DeepSeek-V3 | `deepseek3` | -- |
| Gemma 3 | `gemma3` | -- |
| GLM-4 | `glm4` | -- |
| Phi-4 | `phi4` | -- |
| InternLM 3 | `intern2` | -- |
| Mistral | `mistral` | -- |
