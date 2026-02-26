# Troubleshooting Guide

Common issues and solutions when using LLaMA-Factory.

## CUDA Out of Memory (Most Common)

**Problem**: `torch.cuda.OutOfMemoryError` during training.
**Solutions** (try in order):
1. Reduce `per_device_train_batch_size` to 1
2. Verify `gradient_checkpointing` is not disabled (enabled by default)
3. Use QLoRA: add `quantization_bit: 4` and `quantization_method: bnb`
4. Reduce `cutoff_len` (sequence length)
5. Use `neat_packing: true` with `flash_attn: fa2`
6. Enable Liger kernel: `enable_liger_kernel: true`
7. Use DeepSpeed ZeRO-3 with offload: `deepspeed: examples/deepspeed/ds_z3_offload_config.json`
8. Set `PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128`

## Quantization Issues

### Cannot Use Quantization with Full/Freeze Fine-Tuning
**Problem**: Error about quantization being incompatible.
**Solution**: `quantization_bit` only works with `finetuning_type: lora` or `finetuning_type: oft`. Cannot combine with `full` or `freeze`.

### Cannot Resize Embeddings of Quantized Model
**Problem**: Error when using `resize_vocab: true` with quantized model.
**Solution**: Resize vocab is incompatible with quantization. Remove one or the other.

### Quantized Model Only Accepts Single Adapter
**Problem**: Cannot load multiple LoRA adapters on a quantized model.
**Solution**: Merge adapters first, then load the merged model.

### QLoRA Training Tips
- Always enable `upcast_layernorm: true` for better stability
- Use `double_quantization: true` for additional memory savings
- QLoRA cannot be used with vLLM inference -- merge LoRA first, then quantize with GPTQ/AWQ

## DeepSpeed Issues

### "DeepSpeed needs CUDA_HOME set" / nvcc Not Found
**Problem**: DeepSpeed cannot find the CUDA toolkit compiler. Common on EC2 GPU instances that have NVIDIA drivers but no CUDA toolkit installed.
**Solution**:
1. Install CUDA toolkit:
```bash
sudo apt-get update && sudo apt-get install -y nvidia-cuda-toolkit
```
2. Set `CUDA_HOME`:
```bash
export CUDA_HOME=/usr/local/cuda
echo 'export CUDA_HOME=/usr/local/cuda' >> ~/.bashrc
```
Verify with: `nvcc --version && echo $CUDA_HOME`

The setup script (`scripts/setup.sh`) attempts to install the CUDA toolkit automatically if `nvcc` is missing.

### "Please use FORCE_TORCHRUN=1"
**Problem**: DeepSpeed requires torchrun launcher.
**Solution**: Set `FORCE_TORCHRUN=1` environment variable:
```bash
FORCE_TORCHRUN=1 llamafactory-cli train config.yaml
```

### predict_with_generate Incompatible with ZeRO-3
**Problem**: Cannot use generation during ZeRO-3 training.
**Solution**: Use ZeRO-2 for evaluation with generation, or run evaluation separately.

### Unsloth Incompatible with ZeRO-3
**Problem**: `use_unsloth: true` fails with DeepSpeed ZeRO-3.
**Solution**: Use ZeRO-2, or disable unsloth for ZeRO-3 training.

### GaLore/APOLLO Incompatible with DeepSpeed
**Problem**: Cannot combine GaLore or APOLLO with DeepSpeed.
**Solution**: Use GaLore/APOLLO on single GPU only, or switch to standard optimizer with DeepSpeed.

### pure_bf16 Incompatible with ZeRO-3
**Problem**: `pure_bf16: true` cannot be used with ZeRO-3.
**Solution**: Use `bf16: true` (standard AMP) instead of `pure_bf16: true` with ZeRO-3.

## Training Stage Issues

### predict_with_generate Only Works for SFT
**Problem**: `predict_with_generate: true` errors on non-SFT stages.
**Solution**: This option only works with `stage: sft`. For other stages, use different evaluation methods.

### neat_packing Only Works for SFT
**Problem**: `neat_packing: true` errors on non-SFT stages.
**Solution**: Sequence packing only works with `stage: sft`.

### PPO Requires Reward Model
**Problem**: "reward_model is necessary for PPO training".
**Solution**: Train a reward model first (`stage: rm`), then reference it: `reward_model: saves/path/to/rm`.

### PPO Logger Limitation
**Problem**: PPO logging only supports wandb or tensorboard.
**Solution**: Set `report_to: wandb` or `report_to: tensorboard` for PPO training.

## LoRA Issues

### Adapter Only Valid for LoRA/OFT
**Problem**: Cannot use `adapter_name_or_path` with `finetuning_type: full` or `freeze`.
**Solution**: Adapters only exist for LoRA/OFT. For full/freeze models, point `model_name_or_path` to the checkpoint.

### Cannot Combine LoRA with GaLore/APOLLO/BAdam
**Problem**: These optimizers are mutually exclusive with LoRA.
**Solution**: GaLore, APOLLO, and BAdam are designed for full-parameter training only.

### Missing Embedding Training Warning
**Problem**: Added tokens are not trainable with LoRA.
**Solution**: Add embedding layers to `additional_target`:
```yaml
additional_target: embed_tokens,lm_head
```

## Reasoning Model Issues

### Understanding enable_thinking
**Problem**: Confusion about thinking mode settings.
**Solution**:
- `enable_thinking: true` (default) -- Slow thinking: model uses `<think>...</think>` tags for chain-of-thought. Auto-adds empty CoT to data without it.
- `enable_thinking: false` -- Fast thinking: CoT moved to prompt (not in model output), loss not computed on CoT.
- `enable_thinking: null` -- Mixed mode: preserves data as-is (use with caution).

**Keep `enable_thinking` consistent between training and inference.**

### Template Selection for Thinking Models
- Qwen3 with thinking: `template: qwen3`
- Qwen3 without thinking: `template: qwen3_nothink`
- Always match template to whether the model should reason or not.

## Mixed Precision Issues

### "We recommend enable mixed precision training"
**Problem**: Warning about not using mixed precision.
**Solution**: Always set `bf16: true` (for A100/H100) or `fp16: true` (for V100/consumer GPUs).

### "This device does not support pure_bf16"
**Problem**: GPU doesn't support bfloat16.
**Solution**: Use `fp16: true` instead. V100 and older GPUs don't support bfloat16.

### FP8 Incompatible with Quantization
**Problem**: Cannot combine FP8 training with QLoRA.
**Solution**: Use either FP8 or quantization, not both.

## Dataset Issues

### Unknown Arguments Warning
**Problem**: "Some specified arguments are not used by DataArguments."
**Solution**: YAML contains unrecognized keys. Fix the YAML or set `ALLOW_EXTRA_ARGS=1`.

### Streaming Mode Requires max_steps
**Problem**: Error when using `streaming: true` with `num_train_epochs`.
**Solution**: Streaming mode cannot determine dataset size. Use `max_steps` instead of `num_train_epochs`.

### Dataset Not Found
**Problem**: Training fails to find dataset.
**Solution**: Verify:
1. Dataset name matches a key in `dataset_info.json`
2. `dataset_dir` points to the folder containing `dataset_info.json`
3. The referenced file/URL exists

## Version Compatibility

### Python Version
LLaMA-Factory v0.9.4+ requires Python 3.11-3.13. Python 3.9-3.10 are deprecated.

### Key Package Version Constraints
```
transformers>=4.51.0,<=5.0.0
datasets>=2.16.0,<=4.0.0
accelerate>=1.3.0,<=1.11.0
peft>=0.18.0,<=0.18.1
trl>=0.18.0,<=0.24.0
```

### neat_packing with transformers>=4.53.0
**Problem**: `neat_packing` is broken with transformers 4.53.0+.
**Solution**: Pin transformers below 4.53.0, or disable `neat_packing`.

## Environment Variables Quick Reference

| Variable | Purpose | Example |
|----------|---------|---------|
| `CUDA_VISIBLE_DEVICES` | GPU selection | `0,1,2,3` |
| `FORCE_TORCHRUN` | Force distributed training | `1` |
| `NNODES` | Number of nodes | `2` |
| `NODE_RANK` | This node's rank | `0` |
| `NPROC_PER_NODE` | Processes per node | `4` |
| `MASTER_ADDR` | Master address | `192.168.1.1` |
| `MASTER_PORT` | Master port | `29500` |
| `USE_RAY` | Enable Ray training | `1` |
| `USE_MCA` | Enable Megatron-Core | `1` |
| `USE_V1` | Use V1 engine | `1` |
| `API_PORT` | API server port | `8000` |
| `USE_MODELSCOPE_HUB` | Use ModelScope hub | `1` |
| `DISABLE_VERSION_CHECK` | Skip dep checks | `1` |
| `ALLOW_EXTRA_ARGS` | Allow unknown YAML args | `1` |
| `LLAMAFACTORY_VERBOSITY` | Log level | `DEBUG` |

## Security Note

Versions <= v0.9.3 had a remote code execution vulnerability (CVE-2025-53002). Update to v0.9.4+.
