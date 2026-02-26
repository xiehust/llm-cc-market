---
name: llamafactory
description: "This skill should be used when the user asks to fine-tune, train, or adapt LLMs using LLaMA-Factory or LlamaFactory. Triggers include: 'fine-tune with llamafactory', 'llamafactory train', 'llamafactory-cli', 'lmf train', 'LlamaBoard', 'LoRA SFT', 'QLoRA training', 'DPO training', 'KTO', 'PPO training', 'ORPO', 'SimPO', 'deploy model API', 'merge LoRA', 'quantize model GPTQ', 'llamafactory webui'. Supports 100+ model families including Qwen3, Llama4, DeepSeek-R1/V3, Gemma3, GLM4/5, Phi-4, InternVL3, and multimodal VL models."
---

# LLaMA-Factory: Unified LLM Fine-Tuning

Fine-tune, align, evaluate, and deploy 100+ LLM families via YAML configs and the `llamafactory-cli` CLI. Published at ACL 2024.

**Stable version**: v0.9.4 (requires Python 3.11+, uses **uv** package manager).

## Prerequisites

Ensure LLaMA-Factory is installed. If not, run `scripts/setup.sh` from the skill directory. This uses **uv** to create a virtual environment (default: `~/lmf-env`) and install LLaMA-Factory.

Requirements: Python >=3.11, PyTorch >=2.4, CUDA 12 (for GPU training).

Invoke via full path: `~/lmf-env/bin/llamafactory-cli train config.yaml`

## CLI Commands

| Command | Purpose |
|---------|---------|
| `llamafactory-cli train config.yaml` | Train (SFT, DPO, KTO, PPO, pre-train, reward model) |
| `llamafactory-cli chat config.yaml` | CLI chat interface |
| `llamafactory-cli webchat config.yaml` | Web chat interface |
| `llamafactory-cli api config.yaml` | OpenAI-compatible API server |
| `llamafactory-cli export config.yaml` | Merge LoRA, quantize, push to hub |
| `llamafactory-cli webui` | LlamaBoard (full GUI for training/inference) |
| `llamafactory-cli env` | Show environment info |

Override YAML params from CLI: `llamafactory-cli train config.yaml learning_rate=1e-5`

Shortcut alias: `lmf` (e.g., `lmf train config.yaml`).

## Quick Reference

| Task | Key YAML Fields |
|------|-----------------|
| LoRA SFT | `stage: sft`, `finetuning_type: lora` |
| QLoRA SFT | `stage: sft`, `finetuning_type: lora`, `quantization_bit: 4` |
| Full SFT | `stage: sft`, `finetuning_type: full` |
| Pre-train | `stage: pt`, `finetuning_type: lora` |
| DPO | `stage: dpo`, `pref_loss: sigmoid` |
| ORPO | `stage: dpo`, `pref_loss: orpo` |
| SimPO | `stage: dpo`, `pref_loss: simpo` |
| KTO | `stage: kto` |
| PPO | `stage: ppo`, `reward_model: /path/to/rm` |
| Reward Model | `stage: rm` |
| OFT (new) | `finetuning_type: oft`, `oft_rank: 8` |

## Workflow

### 0. Environment Selection

Before any training task, determine the execution environment.

Use `AskUserQuestion` to ask: "Where will training run?" with options:
- **Local** — "Run on this machine"
- **Remote EC2** — "Run on remote EC2 instance(s) via SSH"

#### If Local
Proceed directly to Step 1. All commands run on this machine.

#### If Remote EC2

1. **Collect connection info**: Use `AskUserQuestion` to ask for the EC2 instance **IP address** and **PEM key file path** (e.g., "What is the EC2 instance IP address and PEM key file path?"). Allow the user to provide both in a single response.
2. **Validate connectivity**:
   ```bash
   chmod 400 /path/to/key.pem
   ssh -i /path/to/key.pem -o ConnectTimeout=10 -o StrictHostKeyChecking=no ubuntu@IP "echo 'Connection OK' && nvidia-smi --query-gpu=name,memory.total --format=csv,noheader"
   ```
   Report GPU info to user. If fails, use `AskUserQuestion` to ask the user to verify IP, PEM path, and security group (port 22).

3. Use `AskUserQuestion` to ask: "Do you want to add more machines?" with options **Yes** / **No**. If yes, repeat collection + validation for each additional machine. Loop until user says no.

4. **If multiple machines**: Use `AskUserQuestion` to ask: "You have N machines. Do you want to run distributed training across all of them?" with options **Yes (distributed)** / **No (use first machine only)**. If yes, designate the first machine as master node (`NODE_RANK=0`).

5. **Record execution context** for all subsequent steps:
   - **Single remote**: Wrap all commands with `ssh -i PEM ubuntu@IP "COMMAND"`
   - **Multi-machine distributed**: Run on all nodes via SSH with different `NODE_RANK`

#### SSH Command Patterns

```bash
# Run a command on remote machine
ssh -i PEM ubuntu@IP "COMMAND"

# Long-running training -- use nohup to survive SSH disconnect
ssh -i PEM ubuntu@IP "nohup bash -c 'COMMAND' > ~/train.log 2>&1 &"

# Check training progress
ssh -i PEM ubuntu@IP "tail -50 ~/train.log"

# Monitor GPU
ssh -i PEM ubuntu@IP "nvidia-smi"

# Upload files (dataset, scripts, YAML config)
scp -i PEM local_file ubuntu@IP:~/remote_path
scp -i PEM -r local_dir ubuntu@IP:~/remote_dir

# Download results
scp -i PEM -r ubuntu@IP:~/output ./local_output
```

#### Remote Environment Setup

If the framework is not installed on the remote machine:
```bash
scp -i PEM scripts/setup.sh ubuntu@IP:~/setup.sh
ssh -i PEM ubuntu@IP "bash ~/setup.sh"
```

#### Distributed Training (Multi-Node)

When distributed training across N machines:

1. **Run setup on ALL nodes** (upload and execute setup.sh on each)
2. **Upload dataset and YAML config to ALL nodes** (or use shared storage EFS/S3):
   ```bash
   scp -i PEM /path/to/dataset.jsonl ubuntu@IP:~/dataset.jsonl
   scp -i PEM config.yaml ubuntu@IP:~/config.yaml
   ```
3. **Launch training on each node** with different NODE_RANK:
   ```bash
   # Master node (machine 1):
   ssh -i PEM1 ubuntu@MASTER_IP "nohup bash -c 'FORCE_TORCHRUN=1 NNODES=N NODE_RANK=0 MASTER_ADDR=MASTER_IP MASTER_PORT=29500 ~/lmf-env/bin/llamafactory-cli train ~/config.yaml' > ~/train.log 2>&1 &"

   # Worker node K:
   ssh -i PEMK ubuntu@WORKERK_IP "nohup bash -c 'FORCE_TORCHRUN=1 NNODES=N NODE_RANK=K MASTER_ADDR=MASTER_IP MASTER_PORT=29500 ~/lmf-env/bin/llamafactory-cli train ~/config.yaml' > ~/train.log 2>&1 &"
   ```
4. **Monitor all nodes**: Check `tail ~/train.log` and `nvidia-smi` on each via SSH
5. **Important**: All nodes need identical environments, security groups must allow TCP 29500-29600 between nodes, start master first then workers, checkpoints saved on master node (NODE_RANK=0)

### 1. Determine Task Type

> **Remote mode**: If remote execution is active (from Step 0), detect hardware from the remote machine:
> ```bash
> ssh -i PEM ubuntu@IP "nvidia-smi --query-gpu=name,memory.total,count --format=csv,noheader"
> ```

Gather from the user:
- **Task**: SFT / pre-train / DPO / KTO / PPO / reward model / eval / export
- **Model**: Which model? (e.g., `Qwen/Qwen3-4B-Instruct-2507`)
- **Dataset**: Custom path or registered dataset name
- **Hardware**: GPU count, memory per GPU
- **Method**: LoRA / QLoRA / full / freeze / OFT

### 2. Create YAML Config

LLaMA-Factory is entirely YAML-driven. Consult the templates in `references/`:
- **`references/training-templates.md`** -- LoRA, QLoRA, full, freeze, OFT, multimodal, pre-train
- **`references/rlhf-templates.md`** -- DPO, KTO, PPO, ORPO, SimPO, reward model
- **`references/deploy-templates.md`** -- Inference, API, export, quantization, evaluation
- **`references/dataset-formats.md`** -- Alpaca, ShareGPT, OpenAI formats, dataset_info.json
- **`references/troubleshooting.md`** -- Common issues and solutions

### 3. Execute

> **Remote mode**: If remote execution is active, upload the YAML config file first, then wrap training commands with SSH using nohup (see Step 0 patterns). For single remote, run directly. For distributed, launch on each node with appropriate `NODE_RANK` and `FORCE_TORCHRUN=1`.
> ```bash
> scp -i PEM config.yaml ubuntu@IP:~/config.yaml
> ssh -i PEM ubuntu@IP "nohup bash -c 'CUDA_VISIBLE_DEVICES=0 ~/lmf-env/bin/llamafactory-cli train ~/config.yaml' > ~/train.log 2>&1 &"
> ```

```bash
# Single GPU
CUDA_VISIBLE_DEVICES=0 ~/lmf-env/bin/llamafactory-cli train config.yaml

# Multi-GPU (auto-detected)
CUDA_VISIBLE_DEVICES=0,1,2,3 ~/lmf-env/bin/llamafactory-cli train config.yaml

# Multi-GPU with DeepSpeed
CUDA_VISIBLE_DEVICES=0,1,2,3 FORCE_TORCHRUN=1 \
    ~/lmf-env/bin/llamafactory-cli train config.yaml
```

Multi-node:
```bash
FORCE_TORCHRUN=1 NNODES=2 NODE_RANK=0 MASTER_ADDR=192.168.0.1 MASTER_PORT=29500 \
    ~/lmf-env/bin/llamafactory-cli train config.yaml
```

### 4. Post-Training

- **Chat test**: `llamafactory-cli chat config.yaml` (add `adapter_name_or_path`)
- **Merge LoRA**: `llamafactory-cli export merge_config.yaml`
- **Quantize**: `llamafactory-cli export quant_config.yaml` (GPTQ)
- **Deploy**: `API_PORT=8000 llamafactory-cli api config.yaml`
- **WebUI**: `llamafactory-cli webui`

> **Remote mode**: If remote execution is active, run post-training commands via SSH. Download results to local machine:
> ```bash
> scp -i PEM -r ubuntu@IP:~/output ./local_output
> ```

## Key YAML Parameters

### Model
| Param | Description | Default |
|-------|-------------|---------|
| `model_name_or_path` | HuggingFace/ModelScope model ID or local path | required |
| `adapter_name_or_path` | LoRA adapter path (for inference/merge) | none |
| `trust_remote_code` | Trust remote code | false |
| `flash_attn` | `auto` / `disabled` / `sdpa` / `fa2` / `fa3` | `auto` |
| `use_unsloth` | Unsloth optimization (170% speed for LoRA) | false |
| `enable_liger_kernel` | Liger kernel acceleration | false |

### Training
| Param | Description | Default |
|-------|-------------|---------|
| `stage` | `pt` / `sft` / `rm` / `ppo` / `dpo` / `kto` | `sft` |
| `finetuning_type` | `lora` / `full` / `freeze` / `oft` | `lora` |
| `do_train` | Enable training | false |
| `output_dir` | Output directory | required |
| `num_train_epochs` | Training epochs | 3.0 |
| `per_device_train_batch_size` | Batch size per GPU | 8 |
| `gradient_accumulation_steps` | Gradient accumulation | 1 |
| `learning_rate` | Learning rate | 5e-5 |
| `lr_scheduler_type` | `cosine` / `linear` / `constant` | `linear` |
| `warmup_ratio` | Warmup ratio | 0.0 |
| `bf16` | Use bfloat16 | false |
| `deepspeed` | Path to DeepSpeed config JSON | none |
| `report_to` | `none` / `wandb` / `tensorboard` / `swanlab` | `none` |
| `plot_loss` | Save loss curves | false |

### Dataset
| Param | Description | Default |
|-------|-------------|---------|
| `dataset` | Comma-separated dataset names | required |
| `dataset_dir` | Dataset folder path | `data` |
| `template` | Chat template (must match model) | required |
| `cutoff_len` | Max token length | 2048 |
| `max_samples` | Limit samples (for debugging) | none |
| `packing` | Sequence packing | false |
| `neat_packing` | Packing without cross-attention | false |
| `streaming` | Dataset streaming | false |
| `enable_thinking` | Reasoning model control: `true`/`false`/`null` | true |

### LoRA
| Param | Description | Default |
|-------|-------------|---------|
| `lora_rank` | LoRA rank | 8 |
| `lora_alpha` | Scale factor | rank * 2 |
| `lora_dropout` | Dropout | 0.0 |
| `lora_target` | Target modules (`all` = all linear) | `all` |
| `use_dora` | Weight-decomposed LoRA (DoRA) | false |
| `use_rslora` | Rank stabilization scaling | false |
| `loraplus_lr_ratio` | LoRA+ learning rate ratio | none |
| `pissa_init` | PiSSA initialization | false |

### Quantization (QLoRA)
| Param | Description | Default |
|-------|-------------|---------|
| `quantization_bit` | `4` or `8` | none |
| `quantization_method` | `bnb` / `hqq` / `eetq` | `bnb` |
| `quantization_type` | `nf4` / `fp4` | `nf4` |
| `double_quantization` | Double quantization | true |

### Multimodal
| Param | Description | Default |
|-------|-------------|---------|
| `freeze_vision_tower` | Freeze vision encoder | true |
| `freeze_multi_modal_projector` | Freeze projector | true |
| `freeze_language_model` | Freeze LLM backbone | false |
| `image_max_pixels` | Max image pixels | 589824 |
| `video_max_pixels` | Max video pixels | 65536 |
| `video_fps` | Video frame rate | 2.0 |

## Hardware Sizing Guide

| Model Size | Method | Min VRAM | Recommended |
|-----------|--------|----------|-------------|
| 7-8B | QLoRA 4-bit | 6GB | 1x RTX 4090 |
| 7-8B | LoRA 16-bit | 16GB | 1x RTX 4090 / A100-40G |
| 7-8B | Full | 120GB | 2x A100-80G |
| 14B | QLoRA 4-bit | 12GB | 1x RTX 4090 |
| 14B | LoRA 16-bit | 32GB | 1x A100-40G |
| 30B | QLoRA 4-bit | 24GB | 1x RTX 4090 |
| 70B | QLoRA 4-bit | 48GB | 2x RTX 4090 |
| 70B | LoRA 16-bit | 160GB | 4x A100-80G |
| 671B (MoE) | KTransformers LoRA | 70GB GPU + 1.3TB RAM | 2-4x RTX 4090 + CPU |

## Dataset Format

LLaMA-Factory supports Alpaca and ShareGPT formats. The simplest for SFT:

**Alpaca format** (default):
```json
[{"instruction": "question", "input": "", "output": "answer"}]
```

**ShareGPT format**:
```json
[{"conversations": [{"from": "human", "value": "Q"}, {"from": "gpt", "value": "A"}]}]
```

Register custom datasets in `dataset_info.json`. See **`references/dataset-formats.md`** for all formats and registration details.

## Distributed Training

| Method | Config |
|--------|--------|
| DDP (auto) | Just set `CUDA_VISIBLE_DEVICES` |
| DeepSpeed ZeRO-2 | `deepspeed: examples/deepspeed/ds_z2_config.json` |
| DeepSpeed ZeRO-3 | `deepspeed: examples/deepspeed/ds_z3_config.json` |
| ZeRO-3 + Offload | `deepspeed: examples/deepspeed/ds_z3_offload_config.json` |
| FSDP | Via Accelerate configs |
| Ray | `USE_RAY=1` env var |
| Megatron | `USE_MCA=1` env var (v0.9.4+) |

Always set `FORCE_TORCHRUN=1` when using DeepSpeed.

## Notes

- LLaMA-Factory is YAML-driven -- all training configs are YAML files, not CLI flags
- Template must match model (e.g., `qwen3`, `llama3`, `gemma3`, `qwen3_nothink` for non-thinking)
- For reasoning models: `enable_thinking: true` (slow/CoT), `false` (fast), `null` (mixed)
- QLoRA cannot be combined with full fine-tuning or freeze tuning
- `llamafactory-cli eval` is deprecated; use `do_predict: true` with vLLM batch inference instead
- Day-0/Day-1 model support: often the first framework to support new model releases
- Unsloth integration gives 170% speed + 50% memory reduction for LoRA
