# Troubleshooting Guide

Common issues and solutions when using ms-swift.

## Training Issues

### NaN or 0 Loss on V100 GPUs
**Problem**: Loss becomes NaN or stays at 0 during training on V100.
**Solution**: V100 does not support bfloat16. Use `--torch_dtype float32`.

### NCCL Shared Memory Error (Docker)
**Problem**: `RuntimeError: NCCL error` related to shared memory in Docker.
**Solution**: Increase Docker shared memory: `docker run --shm-size 16g ...`

### RuntimeError: Expected to mark a variable ready only once
**Problem**: DDP error when freezing certain model layers.
**Solution**: Add `--gradient_checkpointing_kwargs '{"use_reentrant": false}'` or use DeepSpeed instead of DDP.

### DDP Unused Parameters Error
**Problem**: Error about unused parameters when freezing layers.
**Solution**: Set `--ddp_find_unused_parameters true`.

### Slow Tokenization on Large Datasets
**Problem**: Preprocessing takes very long on large datasets.
**Solution**: Use `--lazy_tokenize true` (tokenize during training) or `--streaming true` (requires `--max_steps` instead of `--num_train_epochs`).

### Streaming Requires max_steps
**Problem**: Error when using `--streaming true` with `--num_train_epochs`.
**Solution**: Cannot use `num_train_epochs` with streaming. Set `--max_steps N` instead.

### device_map + DeepSpeed Conflict
**Problem**: Error when using `--device_map auto` with `--deepspeed`.
**Solution**: Cannot use both simultaneously. Remove `--device_map` when using DeepSpeed.

### GPTQ Models Cannot Full Fine-Tune
**Problem**: Error attempting full fine-tuning on a pre-quantized GPTQ model.
**Solution**: GPTQ models have integer weights and can only be fine-tuned with LoRA/QLoRA, not full fine-tuning.

### Checkpoint Larger Than Original Model
**Problem**: Saved checkpoint is larger than the original model.
**Solution**: V100 stores weights in FP32. Use `--torch_dtype bfloat16` on A100/H100 GPUs to keep original precision.

### Training Speed Drops in Multi-Machine Setup
**Problem**: Multi-node training is much slower than expected.
**Solution**: Check NCCL configuration and network bandwidth between nodes. Consider ZeRO2 instead of ZeRO3, or use Megatron-SWIFT for large-scale training.

### LongLoRA Compatibility
**Problem**: LongLoRA errors with non-Llama models.
**Solution**: LongLoRA (`--tuner_type longlora`) is only compatible with Llama-series models.

## GRPO Issues

### GRPO loss=0, grad_norm=0
**Problem**: Loss stays at 0 at the beginning of GRPO training.
**Solution**: This is common in early stages. Check that reward functions return non-zero values and that data quality is good. Increase `--num_generations` to get more diverse completions.

### GRPO Colocate OOM
**Problem**: Out-of-memory in GRPO colocate mode.
**Solution**: Options (try in order):
1. Reduce `--num_generations` (e.g., from 8 to 4)
2. Use `--sleep_level 1` or `--sleep_level 2` to release GPU memory between phases
3. Use `--offload_optimizer true`
4. Switch to server mode (`--vllm_mode server`) with dedicated generation GPUs
5. Reduce `--max_completion_length`

### GRPO colocate + async_generate Not Supported
**Problem**: Error combining colocate mode with async generation.
**Solution**: Async generation only works with server mode: `--vllm_mode server --async_generate true`.

## Inference Issues

### Transformers vs vLLM Output Differences
**Problem**: Same prompt produces different outputs between backends.
**Solution**: Expected behavior due to different sampling implementations. Not a bug.

### Max Length Error During Inference
**Problem**: Error about exceeding maximum sequence length.
**Solution**: Increase `--max_length` for transformers backend or `--vllm_max_model_len` for vLLM backend.

### CPU Inference
**Problem**: Want to run inference without GPU.
**Solution**: Set `CUDA_VISIBLE_DEVICES='-1'` before the command.

## Memory Issues

### VLM (Vision-Language Model) OOM
**Problem**: Out-of-memory when training multimodal models.
**Solution**: Combine strategies:
1. `--freeze_vit true` (default, freeze vision encoder)
2. `--max_pixels 1003520` (limit image resolution)
3. `--tuner_type lora` (use LoRA instead of full fine-tuning)
4. Reduce `--per_device_train_batch_size`

### General OOM During Training
**Problem**: CUDA out-of-memory during training.
**Solution**: Try in order:
1. Reduce `--per_device_train_batch_size` to 1
2. Add `--gradient_checkpointing true` (default, but verify)
3. Use QLoRA: `--quant_method bnb --quant_bits 4`
4. Reduce `--max_length`
5. Add `--packing true --attn_impl flash_attn` (requires flash-attn)
6. Add `--use_liger_kernel true`
7. Use DeepSpeed: `--deepspeed zero2` or `zero3`

### QLoRA Cannot Merge Weights
**Problem**: Need to deploy a QLoRA-trained model but cannot merge weights.
**Solution**: QLoRA weights cannot be merged. For production:
1. Train with standard LoRA (not QLoRA)
2. Merge: `~/swift-env/bin/swift merge-lora --model X --adapters output/checkpoint-xxx`
3. Quantize merged model: `~/swift-env/bin/swift export --model output/merged --quant_method awq --quant_bits 4`

## Evaluation Issues

### Evaluation Stops at Fixed Percentage
**Problem**: `swift eval` hangs or stops at a specific percentage.
**Solution**: Set `SWIFT_TIMEOUT=-1` environment variable to disable the timeout.

### NLTK Download Failure
**Problem**: `nltk.download('punkt_tab')` fails in restricted network environments.
**Solution**: Known issue. Manually download the punkt_tab package and place in nltk_data directory.

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

## Multimodal / VL Model Issues

### "Missing qwen_vl_utils" or "Missing torchvision" for Qwen3.5 / Qwen-VL Models
**Problem**: `ModuleNotFoundError: No module named 'qwen_vl_utils'` or `No module named 'torchvision'` when using Qwen3.5 or Qwen3-VL models. Qwen3.5 is a multimodal MoE model that requires vision utilities and torchvision even for text-only tasks.
**Solution**: Install both packages:
```bash
uv pip install qwen_vl_utils torchvision --python ~/swift-env/bin/python
```
For video support, also install:
```bash
uv pip install qwen_vl_utils[video] --python ~/swift-env/bin/python
```

## Setup Issues

### ms-swift Installation Conflicts
**Problem**: Package conflicts during installation.
**Solution**: Use a clean virtual environment. The setup script (`scripts/setup.sh`) creates an isolated env via uv.

### flash-attn Installation Fails
**Problem**: `pip install flash-attn` compilation errors.
**Solution**: Ensure CUDA toolkit matches PyTorch CUDA version. Try:
```bash
pip install flash-attn --no-build-isolation
```
Or use a pre-built wheel for your CUDA version.

### vLLM Installation Issues
**Problem**: vLLM fails to install or import.
**Solution**: Install the ms-swift vLLM extra:
```bash
pip install ms-swift[vllm]
```
Ensure CUDA 12 and compatible PyTorch version.

## Environment Variables Quick Reference

| Variable | Purpose | Example |
|----------|---------|---------|
| `CUDA_VISIBLE_DEVICES` | GPU selection | `0,1,2,3` |
| `NPROC_PER_NODE` | Multi-GPU (auto torchrun) | `4` |
| `NNODES` | Multi-node count | `2` |
| `NODE_RANK` | This node's rank | `0` |
| `MASTER_ADDR` | Master node address | `192.168.1.1` |
| `MASTER_PORT` | Master node port | `29500` |
| `USE_HF` | Use HuggingFace hub | `1` |
| `MODELSCOPE_CACHE` | Model cache directory | `/data/cache` |
| `MAX_PIXELS` | Max image pixels (multimodal) | `1003520` |
| `VIDEO_MAX_PIXELS` | Max video frame pixels | `50176` |
| `FPS_MAX_FRAMES` | Max video frames | `64` |
| `SWIFT_TIMEOUT` | Eval timeout (-1 to disable) | `-1` |
| `SWIFT_UI_LANG` | Web-UI language | `en` |
| `NCCL_DEBUG` | NCCL debugging | `INFO` |
| `OMP_NUM_THREADS` | OpenMP threads (multimodal CPU) | `4` |
