#!/usr/bin/env python3
"""Verify Megatron-SWIFT environment is correctly installed.

Usage:
    ~/swift-env/bin/python scripts/check_megatron_env.py

Checks all required and optional dependencies for Megatron-SWIFT training,
reports versions, and provides actionable fix instructions for any failures.
"""

import importlib
import os
import shutil
import sys

# ── Formatting helpers ──────────────────────────────────────────────────────

GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
BOLD = "\033[1m"
RESET = "\033[0m"

PASS = f"{GREEN}PASS{RESET}"
WARN = f"{YELLOW}WARN{RESET}"
FAIL = f"{RED}FAIL{RESET}"

results: list[tuple[str, str, str]] = []  # (status, name, detail)


def check(name: str, required: bool = True):
    """Decorator that catches exceptions and records results."""
    def decorator(fn):
        def wrapper():
            try:
                ok, detail = fn()
                if ok:
                    results.append((PASS, name, detail))
                elif required:
                    results.append((FAIL, name, detail))
                else:
                    results.append((WARN, name, detail))
            except Exception as e:
                tag = FAIL if required else WARN
                results.append((tag, name, str(e)))
        return wrapper
    return decorator


# ── Individual checks ───────────────────────────────────────────────────────

@check("Python >= 3.9")
def check_python():
    v = sys.version_info
    ok = v >= (3, 9)
    return ok, f"{v.major}.{v.minor}.{v.micro}"


@check("PyTorch + CUDA")
def check_torch():
    import torch
    cuda_available = torch.cuda.is_available()
    cuda_ver = torch.version.cuda or "N/A"
    gpu_count = torch.cuda.device_count() if cuda_available else 0
    gpu_names = []
    if cuda_available:
        for i in range(min(gpu_count, 4)):
            gpu_names.append(torch.cuda.get_device_name(i))
        if gpu_count > 4:
            gpu_names.append(f"... +{gpu_count - 4} more")
    detail = f"torch {torch.__version__}, CUDA {cuda_ver}, {gpu_count} GPU(s)"
    if gpu_names:
        detail += f" [{', '.join(gpu_names)}]"
    if not cuda_available:
        return False, f"torch {torch.__version__} -- CUDA NOT available"
    return True, detail


@check("ms-swift")
def check_swift():
    import swift
    ver = getattr(swift, "__version__", "unknown")
    return True, f"v{ver}"


@check("transformer_engine")
def check_te():
    import transformer_engine
    ver = getattr(transformer_engine, "__version__", "unknown")
    return True, f"v{ver}"


@check("megatron.core (megatron-core)")
def check_mcore():
    import megatron.core
    ver = getattr(megatron.core, "__version__", None)
    if ver is None:
        # Try package metadata
        from importlib.metadata import version as pkg_version
        try:
            ver = pkg_version("megatron-core")
        except Exception:
            ver = "installed (version unknown)"
    return True, f"v{ver}"


@check("flash_attn", required=False)
def check_flash_attn():
    import flash_attn
    ver = getattr(flash_attn, "__version__", "unknown")
    return True, f"v{ver}"


@check("apex", required=False)
def check_apex():
    import apex
    ver = getattr(apex, "__version__", "installed")
    return True, f"v{ver}"


@check("pybind11", required=False)
def check_pybind11():
    import pybind11
    ver = pybind11.__version__
    return True, f"v{ver}"


@check("CUDA_HOME env var")
def check_cuda_home():
    cuda_home = os.environ.get("CUDA_HOME")
    if not cuda_home:
        return False, (
            "CUDA_HOME not set. Run: export CUDA_HOME=/usr  "
            "(or /usr/local/cuda if that's where nvcc lives)"
        )
    nvcc = os.path.join(cuda_home, "bin", "nvcc")
    if os.path.isfile(nvcc):
        return True, f"{cuda_home} (nvcc found)"
    return False, f"{cuda_home} (nvcc NOT found at {nvcc})"


@check("nvcc compiler")
def check_nvcc():
    nvcc = shutil.which("nvcc")
    if nvcc:
        import subprocess
        try:
            out = subprocess.check_output([nvcc, "--version"], text=True, timeout=5)
            for line in out.strip().splitlines():
                if "release" in line.lower():
                    return True, f"{nvcc} -- {line.strip()}"
            return True, nvcc
        except Exception:
            return True, nvcc
    return False, "nvcc not found in PATH"


@check("MEGATRON_LM_PATH env var", required=False)
def check_megatron_lm_path():
    path = os.environ.get("MEGATRON_LM_PATH")
    if not path:
        return False, (
            "MEGATRON_LM_PATH not set. swift will auto-clone, or set it manually:\n"
            "    git clone --branch core_r0.15.0 https://github.com/NVIDIA/Megatron-LM.git\n"
            "    export MEGATRON_LM_PATH=/path/to/Megatron-LM"
        )
    if os.path.isdir(path):
        return True, path
    return False, f"{path} (directory does not exist)"


@check("NCCL", required=False)
def check_nccl():
    """Check NCCL availability via PyTorch."""
    import torch.distributed as dist
    if dist.is_nccl_available():
        return True, "available via torch.distributed"
    return False, "NCCL not available"


@check("Megatron import smoke test")
def check_megatron_import():
    """Try the imports that megatron sft would do at startup."""
    from megatron.core import parallel_state  # noqa: F401
    from megatron.core.transformer import TransformerConfig  # noqa: F401
    return True, "megatron.core imports OK"


@check("swift megatron CLI entry point")
def check_megatron_cli():
    """Verify the 'megatron' CLI command is importable."""
    try:
        from swift.cli.megatron_sft import main  # noqa: F401
        return True, "swift.cli.megatron_sft importable"
    except ImportError:
        # Older versions may have different path
        try:
            from swift.megatron import MegatronArguments  # noqa: F401
            return True, "swift.megatron importable"
        except ImportError:
            return False, "Cannot import swift megatron modules"


# ── Run all checks ──────────────────────────────────────────────────────────

def main():
    print(f"\n{BOLD}Megatron-SWIFT Environment Check{RESET}")
    print("=" * 60)

    checks = [
        check_python,
        check_torch,
        check_nvcc,
        check_cuda_home,
        check_swift,
        check_te,
        check_mcore,
        check_flash_attn,
        check_apex,
        check_pybind11,
        check_megatron_lm_path,
        check_nccl,
        check_megatron_import,
        check_megatron_cli,
    ]

    for fn in checks:
        fn()

    # Print results
    max_name = max(len(r[1]) for r in results)
    for status, name, detail in results:
        print(f"  [{status}] {name:<{max_name}}  {detail}")

    # Summary
    n_fail = sum(1 for s, _, _ in results if FAIL in s)
    n_warn = sum(1 for s, _, _ in results if WARN in s)
    n_pass = sum(1 for s, _, _ in results if PASS in s)

    print()
    print("-" * 60)
    print(f"  {GREEN}{n_pass} passed{RESET}, {YELLOW}{n_warn} warnings{RESET}, {RED}{n_fail} failures{RESET}")

    if n_fail > 0:
        print(f"\n{RED}{BOLD}Environment NOT ready for Megatron training.{RESET}")
        print("Fix the FAIL items above. See references/megatron-guide.md for install instructions.")
        print()
        # Print condensed install hint
        failed_names = [name for s, name, _ in results if FAIL in s]
        if any("transformer_engine" in n for n in failed_names):
            print("Quick fix for transformer_engine + megatron-core (EC2 / no system cuDNN):")
            print("""
  CUDA_HOME=/usr \\
  CUDNN_HOME=~/swift-env/lib/python3.11/site-packages/nvidia/cudnn \\
  CUDNN_PATH=~/swift-env/lib/python3.11/site-packages/nvidia/cudnn \\
  CPLUS_INCLUDE_PATH="~/swift-env/lib/python3.11/site-packages/nvidia/cudnn/include:~/swift-env/lib/python3.11/site-packages/nvidia/nccl/include" \\
  LIBRARY_PATH="~/swift-env/lib/python3.11/site-packages/nvidia/cudnn/lib:~/swift-env/lib/python3.11/site-packages/nvidia/nccl/lib" \\
  uv pip install "transformer-engine[pytorch]" megatron-core --python ~/swift-env/bin/python
""")
        return 1
    elif n_warn > 0:
        print(f"\n{YELLOW}{BOLD}Environment ready (with warnings).{RESET}")
        print("WARN items are optional but recommended. Megatron training should work.")
        return 0
    else:
        print(f"\n{GREEN}{BOLD}Environment fully ready for Megatron training!{RESET}")
        return 0


if __name__ == "__main__":
    sys.exit(main())
