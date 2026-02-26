# River CC Market

A Claude Code plugin marketplace for LLM fine-tuning and deployment workflows.

## Overview

River CC Market provides reusable skills for [Claude Code](https://docs.anthropic.com/en/docs/claude-code) that simplify model fine-tuning, RLHF training, and deployment. It wraps production-grade frameworks behind guided workflows so you can go from raw data to a deployed model with natural language commands.

## Available Plugins

### finetune

Fine-tune and deploy LLMs using [ms-swift](https://github.com/modelscope/ms-swift) (with llamafactory support planned).

**Supported fine-tuning methods:**

| Method | VRAM (7B model) | Use case |
|--------|-----------------|----------|
| LoRA | ~18 GB | Standard fine-tuning |
| QLoRA | ~9 GB | Memory-constrained environments |
| Full parameter | ~60 GB | Maximum quality |

**RLHF algorithms:** GRPO (+ DAPO, SAPO, CISPO, CHORD variants), DPO, KTO, PPO, CPO, SimPO, ORPO, GKD, RLOO, Reward Model training

**Model operations:** Inference, OpenAI-compatible API deployment (vLLM/SGLang/LMDeploy), quantization (AWQ/GPTQ/BNB/FP8), evaluation, LoRA merging, data sampling/distillation

**Supported models:** 600+ LLMs and 300+ multimodal models — Qwen3, Llama4, DeepSeek-R1/V3, InternLM3, GLM4/5, Gemma3, Phi-4, Mistral, and vision models like Qwen3-VL, InternVL3, Qwen3-Omni via HuggingFace/ModelScope.

## Requirements

- Python >= 3.9
- PyTorch >= 2.0
- CUDA 12.x
- GPU with sufficient VRAM for your target model/method

## Installation

Install the plugin in Claude Code:

```
claude install-plugin /path/to/river-cc-market/plugins/finetune
```

Or set up the ms-swift environment manually:

```bash
cd plugins/finetune/skills/ms-swift
bash scripts/setup.sh
```

The setup script installs [uv](https://github.com/astral-sh/uv), creates a Python 3.11 virtual environment, and installs ms-swift with optional extras (vLLM, LMDeploy, DeepSpeed, Flash Attention).

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
            └── ms-swift/
                ├── SKILL.md           # Skill workflow guide
                ├── scripts/
                │   └── setup.sh       # Environment setup
                └── references/
                    ├── training-templates.md   # SFT/LoRA/full training commands
                    ├── grpo-guide.md           # GRPO & RL algorithm guide
                    ├── rlhf-templates.md       # DPO/KTO/PPO/GKD commands
                    ├── deploy-templates.md     # Inference, deployment & quantization
                    ├── dataset-formats.md      # All dataset formats by task type
                    └── troubleshooting.md      # Common issues & solutions
```

## Usage

Once installed, use natural language in Claude Code to fine-tune and deploy models:

- **Train:** "Fine-tune Qwen3-8B on my dataset using LoRA"
- **RLHF:** "Run GRPO training on my math dataset with accuracy rewards"
- **Deploy:** "Deploy my fine-tuned model as an OpenAI-compatible API with vLLM"
- **Evaluate:** "Run MMLU and GSM8K benchmarks on my model"
- **Quantize:** "Quantize my model to 4-bit AWQ"
- **Agent:** "Train a tool-calling agent with GRPO"

The skill handles environment setup, command generation, hardware sizing, and data format validation.

## License

See individual plugin directories for license details.
