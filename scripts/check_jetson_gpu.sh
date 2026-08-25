#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIRECTORY="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPOSITORY_ROOT="$(cd -- "${SCRIPT_DIRECTORY}/.." && pwd -P)"
PYTHON="${REPOSITORY_ROOT}/.venv/bin/python"

if [[ ! -x "${PYTHON}" ]]; then
    printf 'Project environment not found. Run: uv sync --locked\n' >&2
    exit 1
fi

printf 'Architecture: %s\n' "$(uname -m)"
if [[ -r /etc/nv_tegra_release ]]; then
    printf 'L4T: '
    head -n 1 /etc/nv_tegra_release
else
    printf 'L4T: not detected\n'
fi

if [[ -x /usr/local/cuda/bin/nvcc ]]; then
    /usr/local/cuda/bin/nvcc --version | tail -n 1
else
    printf 'CUDA toolkit: not found at /usr/local/cuda\n' >&2
fi

"${PYTHON}" -u - <<'PY'
try:
    import torch
except ImportError:
    raise SystemExit("PyTorch is not installed in .venv")

print(f"Python: {__import__('sys').version.split()[0]}")
print(f"PyTorch: {torch.__version__}")
print(f"PyTorch CUDA build: {torch.version.cuda}")
print(f"CUDA available: {torch.cuda.is_available()}")

if not torch.cuda.is_available():
    raise SystemExit("PyTorch cannot access the Jetson GPU")

print(f"GPU: {torch.cuda.get_device_name(0)}")
value = torch.ones(1024, device="cuda").sum().item()
if value != 1024:
    raise SystemExit(f"CUDA tensor test failed: expected 1024, got {value}")
print(f"CUDA tensor test: passed ({value:.0f})")
PY
