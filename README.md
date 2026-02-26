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
            │   └── references/        # Training, GRPO, RLHF, deploy, dataset, troubleshooting
            └── llamafactory/
                ├── SKILL.md           # LLaMA-Factory skill guide
                ├── scripts/setup.sh   # Environment setup
                └── references/        # Training, RLHF, deploy, dataset, troubleshooting
```

## Usage

Once installed, use natural language in Claude Code to fine-tune and deploy models:

- **Train (ms-swift):** "Fine-tune Qwen3-4B-Instruct-2507 on dataset yahma/alpaca-cleaned#1000 using LoRA with ms-swift"
- **Train (LLaMA-Factory):** "Fine-tune Qwen3-4B-Instruct-2507 on dataset yahma/alpaca-cleaned#1000 with llamafactory using QLoRA"
- **RLHF:** "Run GRPO training on my math dataset with accuracy rewards"
- **DPO:** "Run DPO alignment on my preference data with llamafactory"
- **Deploy:** "Deploy my fine-tuned model as an OpenAI-compatible API with vLLM"
- **Evaluate:** "Run MMLU and GSM8K benchmarks on my model"
- **Quantize:** "Quantize my model to 4-bit AWQ"
- **Agent:** "Train a tool-calling agent with GRPO"

The skill handles environment setup, command generation, hardware sizing, and data format validation.

## License

See individual plugin directories for license details.
