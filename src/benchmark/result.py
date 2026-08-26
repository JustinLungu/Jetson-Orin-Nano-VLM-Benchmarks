"""Validated and atomically persisted benchmark result contracts."""

import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Sequence

BenchmarkSampleStatus = Literal["passed", "failed", "skipped"]
BenchmarkRunScope = Literal["limited", "full"]
BenchmarkRunStatus = Literal["running", "completed", "interrupted", "failed"]
BENCHMARK_SCHEMA_VERSION = 5


@dataclass(frozen=True, slots=True)
class BenchmarkRunMetadata:
    """Configuration and environment required to identify one benchmark run."""

    model: str
    family: str
    runtime_precision: str
    dataset: str
    warmup_iterations: int
    runtime_versions: dict[str, str]
    desktop_active: bool
    dataset_total_images: int
    selected_images: int
    run_scope: BenchmarkRunScope
    input_profile: str
    requested_image_size: int | None


@dataclass(frozen=True, slots=True)
class BenchmarkSampleResult:
    """Inference outcome for one deterministically indexed dataset image."""

    index: int
    sample_id: str
    status: BenchmarkSampleStatus
    inference_time_seconds: float | None = None
    generated_tokens: int | None = None
    error_type: str | None = None
    error_message: str | None = None



@dataclass(frozen=True, slots=True)
class BenchmarkSummary:
    """Compact final latency and throughput aggregates."""

    processed_images: int
    failed_images: int
    skipped_images: int
    model_load_seconds: float
    mean_inference_seconds: float | None
    median_inference_seconds: float | None
    p95_inference_seconds: float | None
    total_run_seconds: float
    images_per_second: float
    generated_tokens_per_second: float | None = None



@dataclass(frozen=True, slots=True)
class JetsonBenchmarkMetrics:
    """Essential memory, power, and temperature measurements for one run."""

    ram_total_mib: float
    ram_before_load_mib: float
    ram_after_load_mib: float | None
    peak_ram_used_mib: float
    peak_swap_used_mib: float
    peak_cuda_memory_mib: float | None
    average_power_watts: float | None
    peak_power_watts: float | None
    peak_temperature_celsius: float | None
    tegrastats_available: bool



@dataclass(frozen=True, slots=True)
class BenchmarkReport:
    """Complete or checkpointed state of one benchmark invocation."""

    created_at: str
    metadata: BenchmarkRunMetadata
    samples: tuple[BenchmarkSampleResult, ...]
    run_status: BenchmarkRunStatus
    error_message: str | None = None
    summary: BenchmarkSummary | None = None
    jetson_metrics: JetsonBenchmarkMetrics | None = None
    schema_version: int = BENCHMARK_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation with a stable schema."""
        return asdict(self)


class BenchmarkReportWriter:
    """Atomically checkpoint one benchmark report at a stable destination."""

    def __init__(
        self,
        destination: Path,
        metadata: BenchmarkRunMetadata,
        *,
        created_at: datetime | None = None,
    ) -> None:
        self.destination = destination
        self.metadata = metadata
        self.created_at = created_at or datetime.now(timezone.utc)

    def write(
        self,
        samples: Sequence[BenchmarkSampleResult],
        *,
        summary: BenchmarkSummary | None = None,
        jetson_metrics: JetsonBenchmarkMetrics | None = None,
        run_status: BenchmarkRunStatus = "running",
        error_message: str | None = None,
    ) -> Path:
        """Replace the report atomically with the supplied progress snapshot."""
        report = BenchmarkReport(
            created_at=self.created_at.isoformat(),
            metadata=self.metadata,
            samples=tuple(samples),
            run_status=run_status,
            error_message=error_message,
            summary=summary,
            jetson_metrics=jetson_metrics,
        )
        self.destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.destination.with_suffix(self.destination.suffix + ".tmp")
        temporary.write_text(
            json.dumps(report.to_dict(), indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, self.destination)
        return self.destination
