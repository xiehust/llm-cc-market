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

# Verify
"$VENV_DIR/bin/swift" --help > /dev/null 2>&1 && echo "ms-swift installed successfully" || echo "Installation failed"

echo ""
echo "Activate the environment with:"
echo "  source $VENV_DIR/bin/activate"
echo "  export PATH=$VENV_DIR/bin:\$PATH"
echo ""
echo "Optional extras (install as needed):"
echo "  uv pip install ms-swift[vllm]      --python $VENV_DIR/bin/python  # vLLM inference"
echo "  uv pip install ms-swift[lmdeploy]  --python $VENV_DIR/bin/python  # LMDeploy inference"
echo "  uv pip install ms-swift[eval]      --python $VENV_DIR/bin/python  # EvalScope evaluation"
echo "  uv pip install flash-attn          --python $VENV_DIR/bin/python  # Flash Attention 2"
echo "  uv pip install deepspeed           --python $VENV_DIR/bin/python  # DeepSpeed distributed training"
