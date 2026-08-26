"""Validated and atomically persisted benchmark result contracts."""

import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Sequence

BenchmarkSampleStatus = Literal["passed", "failed", "skipped"]
BenchmarkRunScope = Literal["limited", "full"]
BENCHMARK_SCHEMA_VERSION = 3


@dataclass(frozen=True, slots=True)
class BenchmarkRunMetadata:
    """Configuration and environment required to identify one benchmark run."""

    model: str
    family: str
    runtime_precision: str
    dataset: str
    batch_size: int
    warmup_iterations: int
    checkpoint_revision: str
    runtime_versions: dict[str, str]
    desktop_active: bool
    dataset_total_images: int
    selected_images: int
    requested_limit: int | None
    run_scope: BenchmarkRunScope
    input_profile: str
    requested_image_size: int | None

    def __post_init__(self) -> None:
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.warmup_iterations < 0:
            raise ValueError("warmup_iterations cannot be negative")
        if self.dataset_total_images <= 0:
            raise ValueError("dataset_total_images must be positive")
        if not 0 < self.selected_images <= self.dataset_total_images:
            raise ValueError(
                "selected_images must be positive and no greater than dataset_total_images"
            )
        if self.requested_limit is not None and self.requested_limit <= 0:
            raise ValueError("requested_limit must be positive")
        expected_scope = "limited" if self.requested_limit is not None else "full"
        if self.run_scope != expected_scope:
            raise ValueError("run_scope does not match requested_limit")
        if self.input_profile not in {"fixed-square", "model-native"}:
            raise ValueError(f"Unknown input_profile: {self.input_profile}")
        if self.input_profile == "fixed-square":
            if self.requested_image_size is None or self.requested_image_size <= 0:
                raise ValueError(
                    "fixed-square input requires a positive requested_image_size"
                )
        elif self.requested_image_size is not None:
            raise ValueError("model-native input cannot request a fixed image size")


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
    source_width: int | None = None
    source_height: int | None = None
    processed_width: int | None = None
    processed_height: int | None = None

    def __post_init__(self) -> None:
        if self.status not in {"passed", "failed", "skipped"}:
            raise ValueError(f"Unknown sample status: {self.status}")
        if self.index < 0:
            raise ValueError("index cannot be negative")
        if self.status == "passed":
            if self.inference_time_seconds is None:
                raise ValueError("A passed sample requires inference_time_seconds")
            if self.error_type or self.error_message:
                raise ValueError("A passed sample cannot contain an error")
        elif not self.error_type:
            raise ValueError(f"A {self.status} sample requires an error type")
        if self.inference_time_seconds is not None and self.inference_time_seconds < 0:
            raise ValueError("inference_time_seconds cannot be negative")
        if self.generated_tokens is not None and self.generated_tokens < 0:
            raise ValueError("generated_tokens cannot be negative")
        if (self.source_width is None) != (self.source_height is None):
            raise ValueError("source_width and source_height must be recorded together")
        if (
            self.source_width is not None
            and self.source_height is not None
            and (self.source_width <= 0 or self.source_height <= 0)
        ):
            raise ValueError("source image dimensions must be positive")
        if (self.processed_width is None) != (self.processed_height is None):
            raise ValueError(
                "processed_width and processed_height must be recorded together"
            )
        if (
            self.processed_width is not None
            and self.processed_height is not None
            and (self.processed_width <= 0 or self.processed_height <= 0)
        ):
            raise ValueError("processed image dimensions must be positive")


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

    def __post_init__(self) -> None:
        for field_name in ("processed_images", "failed_images", "skipped_images"):
            if getattr(self, field_name) < 0:
                raise ValueError(f"{field_name} cannot be negative")
        for field_name in (
            "mean_inference_seconds",
            "median_inference_seconds",
            "p95_inference_seconds",
            "total_run_seconds",
            "images_per_second",
            "generated_tokens_per_second",
            "model_load_seconds",
        ):
            value = getattr(self, field_name)
            if value is not None and value < 0:
                raise ValueError(f"{field_name} cannot be negative")


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

    def __post_init__(self) -> None:
        for field_name in (
            "ram_total_mib",
            "ram_before_load_mib",
            "ram_after_load_mib",
            "peak_ram_used_mib",
            "peak_swap_used_mib",
            "peak_cuda_memory_mib",
            "average_power_watts",
            "peak_power_watts",
            "peak_temperature_celsius",
        ):
            value = getattr(self, field_name)
            if value is not None and value < 0:
                raise ValueError(f"{field_name} cannot be negative")


@dataclass(frozen=True, slots=True)
class BenchmarkReport:
    """Complete or checkpointed state of one benchmark invocation."""

    created_at: str
    metadata: BenchmarkRunMetadata
    samples: tuple[BenchmarkSampleResult, ...]
    run_completed: bool
    summary: BenchmarkSummary | None = None
    jetson_metrics: JetsonBenchmarkMetrics | None = None
    schema_version: int = BENCHMARK_SCHEMA_VERSION

    def __post_init__(self) -> None:
        indices = [sample.index for sample in self.samples]
        if indices != list(range(len(self.samples))):
            raise ValueError("Sample indices must be contiguous and start at zero")
        if self.run_completed and self.summary is None:
            raise ValueError("A completed report requires a summary")
        if not self.run_completed and self.summary is not None:
            raise ValueError("An incomplete report cannot contain a final summary")
        if self.summary is not None:
            reported_samples = (
                self.summary.processed_images
                + self.summary.failed_images
                + self.summary.skipped_images
            )
            if reported_samples != len(self.samples):
                raise ValueError("Summary counts must equal the number of samples")

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
        run_completed: bool = False,
        summary: BenchmarkSummary | None = None,
        jetson_metrics: JetsonBenchmarkMetrics | None = None,
    ) -> Path:
        """Replace the report atomically with the supplied progress snapshot."""
        report = BenchmarkReport(
            created_at=self.created_at.isoformat(),
            metadata=self.metadata,
            samples=tuple(samples),
            run_completed=run_completed,
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
