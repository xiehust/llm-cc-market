#!/bin/bash
# ms-swift setup script using uv
set -e

VENV_DIR="${VENV_DIR:-$HOME/swift-env}"

echo "Installing uv (if not present)..."
if ! command -v uv &> /dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

echo "Creating virtual environment at $VENV_DIR ..."
uv venv "$VENV_DIR" --python 3.11

echo "Installing ms-swift..."
uv pip install ms-swift --python "$VENV_DIR/bin/python"

# Verify installation
if "$VENV_DIR/bin/swift" --help > /dev/null 2>&1; then
    echo ""
    echo "ms-swift installed successfully!"
    echo "Version: $("$VENV_DIR/bin/pip" show ms-swift 2>/dev/null | grep Version || echo 'unknown')"
else
    echo "Installation failed. Check errors above."
    exit 1
fi

echo ""
echo "Usage (always use full path):"
echo "  $VENV_DIR/bin/swift sft --model ... --dataset ..."
echo ""
echo "Or activate the environment:"
echo "  source $VENV_DIR/bin/activate"
echo ""
echo "Optional extras (install as needed):"
echo "  uv pip install ms-swift[vllm]      --python $VENV_DIR/bin/python  # vLLM inference/deployment"
echo "  uv pip install ms-swift[lmdeploy]  --python $VENV_DIR/bin/python  # LMDeploy inference"
echo "  uv pip install ms-swift[eval]      --python $VENV_DIR/bin/python  # EvalScope evaluation"
echo "  uv pip install flash-attn          --python $VENV_DIR/bin/python  # Flash Attention 2 (A100+)"
echo "  uv pip install deepspeed           --python $VENV_DIR/bin/python  # DeepSpeed distributed training"
echo "  uv pip install autoawq             --python $VENV_DIR/bin/python  # AWQ quantization"
echo "  uv pip install auto_gptq optimum   --python $VENV_DIR/bin/python  # GPTQ quantization"
echo "  uv pip install bitsandbytes        --python $VENV_DIR/bin/python  # BNB quantization / QLoRA"
