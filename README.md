# River CC Market

A Claude Code plugin marketplace for LLM fine-tuning and deployment workflows.

## Overview

River CC Market provides reusable skills for [Claude Code](https://docs.anthropic.com/en/docs/claude-code) that simplify model fine-tuning, RLHF training, and deployment. It wraps production-grade frameworks behind guided workflows so you can go from raw data to a deployed model with natural language commands.

## Available Plugins

### finetune

Fine-tune and deploy LLMs using [ms-swift](https://github.com/modelscope/ms-swift) and [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory).

**Supported fine-tuning methods:**

| Method | VRAM (7B model) | Use case |
|--------|-----------------|----------|
| LoRA | ~18 GB | Standard fine-tuning |
| QLoRA | ~9 GB | Memory-constrained environments |
| Full parameter | ~60 GB | Maximum quality |

**Skills:**

| Skill | Framework | Models | Strengths |
|-------|-----------|--------|-----------|
| **ms-swift** | [ModelScope SWIFT](https://github.com/modelscope/ms-swift) | 600+ LLM, 300+ MLLM | GRPO/DAPO/SAPO, native Megatron, broadest model support |
| **llamafactory** | [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) | 100+ families | Zero-code WebUI, day-0 model support, YAML-driven |

**Supported fine-tuning methods:** LoRA, QLoRA, full parameter, freeze, OFT, DoRA, LoRA+, PiSSA, GaLore

**RLHF algorithms:** GRPO (+ DAPO, SAPO, CISPO, CHORD), DPO, KTO, PPO, CPO, SimPO, ORPO, GKD, RLOO, Reward Model training

**Megatron-SWIFT (ms-swift):** Native NVIDIA Megatron-LM integration with tensor/pipeline/expert/sequence/context parallelism. MoE models get ~10x training speedup vs DeepSpeed. Supports Mcore-Bridge for seamless HF weight loading. Full-param, LoRA, and FP8 training.

**Model operations:** Inference, OpenAI-compatible API deployment (vLLM/SGLang/LMDeploy), quantization (AWQ/GPTQ/BNB/FP8), evaluation, LoRA merging, data sampling/distillation

**Supported models:** Qwen3, Llama4, DeepSeek-R1/V3, InternLM3, GLM4/5, Gemma3, Phi-4, Mistral, and vision models like Qwen3-VL, InternVL3, Qwen3-Omni via HuggingFace/ModelScope.

## Requirements

- Python >= 3.11 (LLaMA-Factory) / >= 3.9 (ms-swift)
- PyTorch >= 2.4 (LLaMA-Factory) / >= 2.0 (ms-swift)
- CUDA 12.x
- GPU with sufficient VRAM for your target model/method

## Installation

Install the finetune plugin directly from GitHub:

```bash
/plugin marketplace add https://github.com/xiehust/river-cc-market.git
/plugin install finetune@river-cc-market
```

Or clone the repo and install locally:

```bash
git clone https://github.com/xiehust/river-cc-market.git
claude plugin add --from ./river-cc-market/plugins/finetune
```

Then set up the training environment (choose one or both):

```bash
# ms-swift
cd river-cc-market/plugins/finetune/skills/ms-swift
bash scripts/setup.sh

# LLaMA-Factory
cd river-cc-market/plugins/finetune/skills/llamafactory
bash scripts/setup.sh
```

Each setup script installs [uv](https://github.com/astral-sh/uv), creates an isolated virtual environment, and installs the framework with optional extras (vLLM, DeepSpeed, Flash Attention, etc.).

## Repository Structure

```
river-cc-market/
├── .claude-plugin/
│   └── marketplace.json              # Marketplace registry
└── plugins/
    └── finetune/
        ├── .claude-plugin/
        │   └── plugin.json           # Plugin metadata
        └── skills/
            ├── ms-swift/
            │   ├── SKILL.md           # ms-swift skill guide
            │   ├── scripts/setup.sh   # Environment setup
            │   └── references/        # Training, GRPO, RLHF, deploy, dataset, Megatron, troubleshooting
            └── llamafactory/
                ├── SKILL.md           # LLaMA-Factory skill guide
                ├── scripts/setup.sh   # Environment setup
                └── references/        # Training, RLHF, deploy, dataset, troubleshooting
```

## Usage

Once installed, use natural language in Claude Code to fine-tune and deploy models:

- **Train (ms-swift):** "Fine-tune Qwen3.5-35B-A3B on dataset yahma/alpaca-cleaned#1000 using LoRA with ms-swift"
- **Train (LLaMA-Factory):** "Fine-tune Qwen3-4B-Instruct-2507 on dataset yahma/alpaca-cleaned#1000 with llamafactory using QLoRA"
- **Megatron:** "Train Qwen3-30B-A3B MoE model on dataset yahma/alpaca-cleaned#1000 with Megatron"
- **RLHF:** "Run GRPO training on my math dataset with accuracy rewards"
- **DPO:** "Run DPO alignment on my preference data with llamafactory"
- **Deploy:** "Deploy my fine-tuned model as an OpenAI-compatible API with vLLM"
- **Evaluate:** "Run MMLU and GSM8K benchmarks on my model"
- **Quantize:** "Quantize my model to 4-bit AWQ"
- **Agent:** "Train a tool-calling agent with GRPO"

The skill handles environment setup, command generation, hardware sizing, and data format validation.

## Skill 简介

### ms-swift

**ms-swift**（ModelScope SWIFT）是ModelScope 团队开源的大模型微调与部署框架，该 skill 封装了 ms-swift 的完整工作流，支持通过自然语言指令完成模型训练、对齐、评估、量化和部署。

- 支持 600+ 文本大模型和 300+ 多模态模型（Qwen3、Llama4、DeepSeek-R1/V3、GLM4/5 等）
- 最全面的 RLHF/RL 算法支持：GRPO 及变体（DAPO、SAPO、CISPO、CHORD）、DPO、KTO、PPO、GKD 等
- 原生 Megatron-SWIFT 集成，支持数据并行（DP）、张量并行（TP）、流水线并行（PP）、序列并行（SP）、上下文并行（CP）、专家并行（EP）
- MoE 模型训练可获 10 倍加速（Qwen3-30B-A3B: Megatron 9.6s/it vs DeepSpeed ZeRO3 91.2s/it）
- Mcore-Bridge 无缝加载 HuggingFace safetensors 权重，无需手动格式转换
- 支持 Megatron 全参数训练、LoRA 微调、FP8 训练，以及 Megatron GRPO/DPO/KTO/GKD
- 通过 CLI 命令行参数驱动（`swift sft` / `megatron sft --model X --dataset Y`）
- 支持 vLLM/SGLang/LMDeploy 多种推理后端部署为 OpenAI 兼容 API

### llamafactory

**LLaMA-Factory**（LlamaFactory）是由 hiyouga 开源的统一大模型微调框架，发表于 ACL 2024，GitHub 67,000+ stars。该 skill 封装了 LLaMA-Factory 的 YAML 配置驱动工作流，支持零代码 Web UI 训练和自然语言指令操作。

- 支持 100+ 模型家族，Day-0/Day-1 极速适配新模型（Qwen3、Llama4、DeepSeek-R1/V3、Gemma3 等）
- 全 YAML 配置驱动，无需记忆命令行参数
- 内置 LlamaBoard Web UI，支持零代码训练和推理
- 丰富的微调方法：LoRA、QLoRA、Full、Freeze、OFT、DoRA、LoRA+、PiSSA、GaLore
- 偏好对齐算法：DPO、KTO、PPO、ORPO、SimPO
- Unsloth 加速集成（LoRA 训练提速 170%，内存减少 50%）
- KTransformers 支持：2 张 4090 即可微调 DeepSeek-V3（671B）

## License

See individual plugin directories for license details.
