"""Compact Jetson memory, power, and temperature collection."""

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from src.benchmark.result import JetsonBenchmarkMetrics
from src.inference.runtime import (
    TegrastatsMonitor,
    peak_cuda_memory_mib,
    reset_peak_cuda_memory,
)

MEMINFO_PATH = Path("/proc/meminfo")


@dataclass(frozen=True, slots=True)
class MemorySnapshot:
    """Host-visible unified RAM and swap usage in mebibytes."""

    ram_total_mib: float
    ram_used_mib: float
    swap_used_mib: float


class BenchmarkTelemetry:
    """Collect essential resource summaries around one benchmark run."""

    def __init__(
        self,
        torch_module: Any,
        device: str = "cuda:0",
        *,
        monitor: TegrastatsMonitor | None = None,
        memory_reader: Callable[[], MemorySnapshot] | None = None,
    ) -> None:
        self.torch = torch_module
        self.device = device
        self.monitor = monitor or TegrastatsMonitor()
        self.memory_reader = memory_reader or read_memory_snapshot
        self.before_load: MemorySnapshot | None = None
        self.after_load: MemorySnapshot | None = None
        self.model_loaded = False
        self.metrics: JetsonBenchmarkMetrics | None = None

    def start(self) -> None:
        """Capture the baseline and begin background board sampling."""
        self.before_load = self.memory_reader()
        self.monitor.start()

    def mark_model_loaded(self) -> None:
        """Capture post-load RAM and begin peak CUDA measurement from that baseline."""
        self.after_load = self.memory_reader()
        reset_peak_cuda_memory(self.torch, self.device)
        self.model_loaded = True

    def stop(self) -> JetsonBenchmarkMetrics:
        """Stop sampling and return compact metrics; repeated calls are idempotent."""
        if self.metrics is not None:
            return self.metrics
        if self.before_load is None:
            raise RuntimeError("Benchmark telemetry was not started")

        tegrastats_summary = self.monitor.stop()
        samples = self.monitor.samples
        ram_values = [
            sample["ram_used_mib"] for sample in samples if "ram_used_mib" in sample
        ]
        swap_values = [
            sample["swap_used_mib"] for sample in samples if "swap_used_mib" in sample
        ]
        fallback_ram = [self.before_load.ram_used_mib]
        fallback_swap = [self.before_load.swap_used_mib]
        if self.after_load is not None:
            fallback_ram.append(self.after_load.ram_used_mib)
            fallback_swap.append(self.after_load.swap_used_mib)

        if tegrastats_summary is None:
            warnings.warn(
                "tegrastats telemetry is unavailable; power and temperature are missing",
                RuntimeWarning,
                stacklevel=2,
            )
        power = tegrastats_summary["power_watts"] if tegrastats_summary else None
        temperature = (
            tegrastats_summary["temperature_celsius"] if tegrastats_summary else None
        )
        self.metrics = JetsonBenchmarkMetrics(
            ram_total_mib=self.before_load.ram_total_mib,
            ram_before_load_mib=self.before_load.ram_used_mib,
            ram_after_load_mib=(
                self.after_load.ram_used_mib if self.after_load is not None else None
            ),
            peak_ram_used_mib=max(ram_values + fallback_ram),
            peak_swap_used_mib=max(swap_values + fallback_swap),
            peak_cuda_memory_mib=(
                peak_cuda_memory_mib(self.torch, self.device)
                if self.model_loaded
                else None
            ),
            average_power_watts=power["average"] if power else None,
            peak_power_watts=power["peak"] if power else None,
            peak_temperature_celsius=temperature["peak"] if temperature else None,
            tegrastats_available=tegrastats_summary is not None,
        )
        return self.metrics


def read_memory_snapshot(path: Path = MEMINFO_PATH) -> MemorySnapshot:
    """Read Linux unified-memory and swap usage from `/proc/meminfo`."""
    values: dict[str, int] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        name, raw_value = line.split(":", maxsplit=1)
        if name in {"MemTotal", "MemAvailable", "SwapTotal", "SwapFree"}:
            values[name] = int(raw_value.strip().split()[0])
    required = {"MemTotal", "MemAvailable", "SwapTotal", "SwapFree"}
    if set(values) != required:
        missing = ", ".join(sorted(required - set(values)))
        raise RuntimeError(f"Memory information is missing fields: {missing}")
    return MemorySnapshot(
        ram_total_mib=values["MemTotal"] / 1024,
        ram_used_mib=(values["MemTotal"] - values["MemAvailable"]) / 1024,
        swap_used_mib=(values["SwapTotal"] - values["SwapFree"]) / 1024,
    )
