"""Runtime measurement and cleanup helpers shared by inference tools."""

import gc
import platform
import re
import time
from pathlib import Path
from typing import Any, Callable, TypeVar

T = TypeVar("T")
TEGRA_RELEASE_PATH = Path("/etc/nv_tegra_release")


def measure_cuda_operation(
    operation: Callable[[], T],
    torch_module: Any,
    *,
    clock: Callable[[], float] = time.perf_counter,
) -> tuple[T, float]:
    """Run an operation and return its result and synchronized elapsed time."""
    torch_module.cuda.synchronize()
    started_at = clock()
    result = operation()
    torch_module.cuda.synchronize()
    elapsed_seconds = clock() - started_at
    return result, elapsed_seconds


def reset_peak_cuda_memory(torch_module: Any, device: str = "cuda:0") -> None:
    """Reset PyTorch's peak-memory counter for a CUDA device."""
    torch_module.cuda.reset_peak_memory_stats(device)


def peak_cuda_memory_mib(torch_module: Any, device: str = "cuda:0") -> float:
    """Return peak allocated CUDA memory in mebibytes."""
    return torch_module.cuda.max_memory_allocated(device) / 1024**2


def cleanup_cuda(
    torch_module: Any,
    *,
    collector: Callable[[], Any] = gc.collect,
) -> None:
    """Collect released model objects and return cached allocations to CUDA."""
    collector()
    if torch_module.cuda.is_available():
        torch_module.cuda.empty_cache()


def is_cuda_out_of_memory(error: Exception, torch_module: Any) -> bool:
    """Recognize PyTorch and lower-level CUDA allocation failures."""
    out_of_memory_type = getattr(torch_module.cuda, "OutOfMemoryError", ())
    if isinstance(error, out_of_memory_type):
        return True
    message = str(error).lower()
    return any(
        marker in message
        for marker in (
            "out of memory",
            "cublas_status_alloc_failed",
            "cudacachingallocator",
        )
    )


def collect_runtime_metadata(
    torch_module: Any,
    *,
    tegra_release_path: Path = TEGRA_RELEASE_PATH,
) -> dict[str, str]:
    """Collect software and device identity included with smoke-test results."""
    l4t_release = "unknown"
    if tegra_release_path.is_file():
        l4t_release = tegra_release_path.read_text(encoding="utf-8").splitlines()[0]

    cuda_available = torch_module.cuda.is_available()
    return {
        "architecture": platform.machine(),
        "python": platform.python_version(),
        "torch": str(torch_module.__version__),
        "cuda": str(torch_module.version.cuda or "unavailable"),
        "cuda_available": str(cuda_available).lower(),
        "device": torch_module.cuda.get_device_name(0) if cuda_available else "unavailable",
        "jetpack": infer_jetpack_family(l4t_release),
        "l4t": l4t_release,
    }


def infer_jetpack_family(l4t_release: str) -> str:
    """Infer the JetPack family without claiming an unavailable patch version."""
    match = re.search(r"R(\d+).*REVISION:\s*(\d+)", l4t_release)
    if not match:
        return "unknown"

    release, revision = (int(value) for value in match.groups())
    known_families = {
        (36, 2): "6.0 DP",
        (36, 3): "6.0",
        (36, 4): "6.2.x",
        (36, 5): "6.2.2",
    }
    return known_families.get((release, revision), "unknown")
