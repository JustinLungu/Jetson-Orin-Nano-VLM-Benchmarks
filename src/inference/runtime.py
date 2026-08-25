"""Runtime measurement and cleanup helpers shared by inference workflows."""

import gc
import platform
import re
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Callable, TypeVar

T = TypeVar("T")
TEGRA_RELEASE_PATH = Path("/etc/nv_tegra_release")
TEGRASTATS_INTERVAL_MS = 100


class TegrastatsMonitor:
    """Collect compact CPU, GPU, power, and temperature summaries on Jetson."""

    def __init__(self, interval_ms: int = TEGRASTATS_INTERVAL_MS) -> None:
        self.interval_ms = interval_ms
        self.process: subprocess.Popen[str] | None = None
        self.reader: threading.Thread | None = None
        self.samples: list[dict[str, float]] = []

    def start(self) -> None:
        """Start sampling when tegrastats is available."""
        executable = shutil.which("tegrastats")
        if executable is None:
            return
        try:
            self.process = subprocess.Popen(
                [executable, "--interval", str(self.interval_ms)],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                bufsize=1,
            )
        except OSError:
            self.process = None
            return
        self.reader = threading.Thread(target=self._read_samples, daemon=True)
        self.reader.start()

    def stop(self) -> dict[str, dict[str, float]] | None:
        """Stop sampling and summarize the collected values."""
        if self.process is not None:
            if self.process.poll() is None:
                self.process.terminate()
                try:
                    self.process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait()
        if self.reader is not None:
            self.reader.join(timeout=2)
        if self.process is not None and self.process.stdout is not None:
            self.process.stdout.close()
        return summarize_tegrastats(self.samples)

    def _read_samples(self) -> None:
        if self.process is None or self.process.stdout is None:
            return
        for line in self.process.stdout:
            sample = parse_tegrastats_line(line)
            if sample is not None:
                self.samples.append(sample)


def parse_tegrastats_line(line: str) -> dict[str, float] | None:
    """Extract requested utilization, memory, power, and temperature measurements."""
    cpu_match = re.search(r"CPU \[([^]]+)]", line)
    gpu_match = re.search(r"GR3D_FREQ\s+(\d+)%", line)
    power_match = re.search(r"VDD_IN\s+(\d+)mW", line)
    temperatures = [
        float(value)
        for value in re.findall(r"\b(?:cpu|gpu|soc\d*|tj)@([\d.]+)C", line)
    ]
    if not (cpu_match and gpu_match and power_match and temperatures):
        return None

    cpu_values = [float(value) for value in re.findall(r"(\d+)%@", cpu_match.group(1))]
    if not cpu_values:
        return None
    sample = {
        "cpu": sum(cpu_values) / len(cpu_values),
        "gpu": float(gpu_match.group(1)),
        "power_watts": float(power_match.group(1)) / 1000,
        "temperature_celsius": max(temperatures),
    }
    ram_match = re.search(r"RAM\s+(\d+)/(\d+)MB", line)
    swap_match = re.search(r"SWAP\s+(\d+)/(\d+)MB", line)
    if ram_match:
        sample["ram_used_mib"] = float(ram_match.group(1))
        sample["ram_total_mib"] = float(ram_match.group(2))
    if swap_match:
        sample["swap_used_mib"] = float(swap_match.group(1))
        sample["swap_total_mib"] = float(swap_match.group(2))
    return sample


def summarize_tegrastats(
    samples: list[dict[str, float]],
) -> dict[str, dict[str, float]] | None:
    """Return average and peak values across tegrastats samples."""
    if not samples:
        return None

    def summary(key: str) -> dict[str, float]:
        values = [sample[key] for sample in samples]
        return {
            "average": round(sum(values) / len(values), 2),
            "peak": round(max(values), 2),
        }

    return {
        "cpu_utilization_percent": summary("cpu"),
        "gpu_utilization_percent": summary("gpu"),
        "power_watts": summary("power_watts"),
        "temperature_celsius": summary("temperature_celsius"),
    }


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
