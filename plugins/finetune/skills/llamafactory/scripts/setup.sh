#!/bin/bash
# LLaMA-Factory setup script using uv
set -e

VENV_DIR="${VENV_DIR:-$HOME/lmf-env}"

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
