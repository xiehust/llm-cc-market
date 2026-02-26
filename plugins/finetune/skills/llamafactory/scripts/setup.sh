#!/bin/bash
# LLaMA-Factory setup script using uv
set -e

VENV_DIR="${VENV_DIR:-$HOME/lmf-env}"

# Install CUDA toolkit if nvcc is not available (common on EC2 GPU instances)
if ! command -v nvcc &> /dev/null && [ ! -f "/usr/local/cuda/bin/nvcc" ]; then
    echo "CUDA toolkit compiler (nvcc) not found. Installing CUDA toolkit..."
    if command -v apt-get &> /dev/null; then
        sudo apt-get update -qq
        sudo apt-get install -y -qq nvidia-cuda-toolkit 2>/dev/null || {
            echo "nvidia-cuda-toolkit not available via apt, trying NVIDIA repo..."
            DISTRO=$(. /etc/os-release && echo "${ID}${VERSION_ID}" | tr -d '.')
            wget -q "https://developer.download.nvidia.com/compute/cuda/repos/${DISTRO}/x86_64/cuda-keyring_1.1-1_all.deb" -O /tmp/cuda-keyring.deb 2>/dev/null && \
                sudo dpkg -i /tmp/cuda-keyring.deb && \
                sudo apt-get update -qq && \
                sudo apt-get install -y -qq cuda-toolkit 2>/dev/null || \
                echo "WARNING: Could not install CUDA toolkit automatically. Install manually."
        }
    elif command -v yum &> /dev/null; then
        sudo yum install -y -q cuda-toolkit 2>/dev/null || \
            echo "WARNING: Could not install CUDA toolkit via yum. Install manually."
    else
        echo "WARNING: Could not install CUDA toolkit. Package manager not recognized."
        echo "Install CUDA toolkit manually: https://developer.nvidia.com/cuda-downloads"
    fi
fi

# Auto-detect CUDA_HOME if not set (required by DeepSpeed)
if [ -z "$CUDA_HOME" ]; then
    if [ -d "/usr/local/cuda" ]; then
        export CUDA_HOME="/usr/local/cuda"
    elif command -v nvcc &> /dev/null; then
        export CUDA_HOME="$(dirname "$(dirname "$(which nvcc)")")"
    fi
    if [ -n "$CUDA_HOME" ]; then
        echo "Auto-detected CUDA_HOME=$CUDA_HOME"
        echo "export CUDA_HOME=$CUDA_HOME" >> ~/.bashrc
    else
        echo "WARNING: CUDA_HOME not set and could not be auto-detected."
        echo "DeepSpeed will not work without it. Set it manually:"
        echo "  export CUDA_HOME=/usr/local/cuda"
    fi
fi

echo "Installing uv (if not present)..."
if ! command -v uv &> /dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

echo "Creating virtual environment at $VENV_DIR ..."
uv venv "$VENV_DIR" --python 3.12

echo "Installing LLaMA-Factory..."
uv pip install llamafactory --python "$VENV_DIR/bin/python"

# Verify installation
if "$VENV_DIR/bin/llamafactory-cli" version > /dev/null 2>&1; then
    echo ""
    echo "LLaMA-Factory installed successfully!"
    "$VENV_DIR/bin/llamafactory-cli" version 2>/dev/null || true
else
    echo "Installation failed. Check errors above."
    exit 1
fi

echo ""
echo "Usage (always use full path):"
echo "  $VENV_DIR/bin/llamafactory-cli train config.yaml"
echo ""
echo "Or activate the environment:"
echo "  source $VENV_DIR/bin/activate"
echo ""
echo "Optional extras (install as needed):"
echo "  uv pip install llamafactory[torch]      --python $VENV_DIR/bin/python  # PyTorch"
echo "  uv pip install llamafactory[vllm]       --python $VENV_DIR/bin/python  # vLLM inference"
echo "  uv pip install llamafactory[deepspeed]  --python $VENV_DIR/bin/python  # DeepSpeed distributed"
echo "  uv pip install flash-attn               --python $VENV_DIR/bin/python  # Flash Attention 2"
echo "  uv pip install bitsandbytes             --python $VENV_DIR/bin/python  # QLoRA (BnB 4-bit)"
echo "  uv pip install auto_gptq optimum        --python $VENV_DIR/bin/python  # GPTQ quantization"
echo "  uv pip install autoawq                  --python $VENV_DIR/bin/python  # AWQ quantization"
echo "  uv pip install unsloth                  --python $VENV_DIR/bin/python  # Unsloth acceleration"
