"""Read-only collection of repository and Jetson benchmark controls."""

import subprocess
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from src.constants import REPOSITORY_ROOT

CPU_FREQUENCY_DIRECTORY = Path("/sys/devices/system/cpu/cpu0/cpufreq")
GPU_FREQUENCY_DIRECTORY = Path(
    "/sys/devices/platform/17000000.gpu/devfreq/17000000.gpu"
)
CommandRunner = Callable[..., subprocess.CompletedProcess[str]]


@dataclass(frozen=True, slots=True)
class BenchmarkProvenance:
    """Source revision and device controls needed to reproduce a run."""

    repository_revision: str
    power_mode: str
    clocks_locked: bool | None


def collect_benchmark_provenance(
    *,
    repository_root: Path = REPOSITORY_ROOT,
    command_runner: CommandRunner = subprocess.run,
    cpu_frequency_directory: Path = CPU_FREQUENCY_DIRECTORY,
    gpu_frequency_directory: Path = GPU_FREQUENCY_DIRECTORY,
) -> BenchmarkProvenance:
    """Collect controls without changing clocks, power mode, or repository state."""
    revision = _command_output(
        ("git", "rev-parse", "HEAD"),
        command_runner,
        cwd=repository_root,
    )
    power_output = _command_output(("nvpmodel", "-q"), command_runner)
    return BenchmarkProvenance(
        repository_revision=revision or "unavailable",
        power_mode=_parse_power_mode(power_output),
        clocks_locked=_read_clocks_locked(
            cpu_frequency_directory,
            gpu_frequency_directory,
        ),
    )


def _command_output(
    command: Sequence[str],
    command_runner: CommandRunner,
    *,
    cwd: Path | None = None,
) -> str | None:
    try:
        result = command_runner(
            list(command),
            cwd=cwd,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return None
    if result.returncode != 0:
        return None
    output = result.stdout.strip()
    return output or None


def _parse_power_mode(output: str | None) -> str:
    if output is None:
        return "unavailable"
    prefix = "NV Power Mode:"
    for line in output.splitlines():
        if line.startswith(prefix):
            mode = line.removeprefix(prefix).strip()
            return mode or "unavailable"
    return "unavailable"


def _read_clocks_locked(
    cpu_directory: Path,
    gpu_directory: Path,
) -> bool | None:
    pairs = (
        (
            cpu_directory / "scaling_min_freq",
            cpu_directory / "scaling_max_freq",
        ),
        (gpu_directory / "min_freq", gpu_directory / "max_freq"),
    )
    frequencies: list[tuple[int, int]] = []
    try:
        for minimum_path, maximum_path in pairs:
            frequencies.append(
                (
                    int(minimum_path.read_text(encoding="utf-8").strip()),
                    int(maximum_path.read_text(encoding="utf-8").strip()),
                )
            )
    except (OSError, ValueError):
        return None
    return all(minimum == maximum for minimum, maximum in frequencies)
