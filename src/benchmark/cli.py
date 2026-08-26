"""Safe command-line orchestration for full-dataset Jetson benchmarks."""

import argparse
import hashlib
import json
import subprocess
from collections.abc import Callable, Sequence
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from src.benchmark.datasets import (
    BenchmarkDataset,
    load_benchmark_dataset,
    validate_dataset_compatibility,
)
from src.benchmark.result import (
    BenchmarkReportWriter,
    BenchmarkRunMetadata,
    BenchmarkSampleResult,
    BenchmarkSummary,
)
from src.benchmark.runner import BenchmarkExecutionError, run_benchmark
from src.constants import (
    MODEL_SELECTORS,
    REPOSITORY_ROOT,
    SMALL_VLM_MODEL_DIRECTORY,
    VLM_RUNTIME_PRECISIONS,
    YOLO_MODEL_DIRECTORY,
    YOLO_MODELS,
)
from src.inference.base import InferenceSession
from src.inference.runtime import collect_runtime_metadata
from src.inference.vlm import VlmInferenceSession
from src.inference.yolo import YoloInferenceSession

BENCHMARK_RESULTS_DIRECTORY = REPOSITORY_ROOT / "results" / "benchmarks"
DEFAULT_BENCHMARK_YOLO_IMAGE_SIZE = 640
UNSAFE_BENCHMARK_MODELS = {"qwen2.5-vl-3b", "phi-3.5-vision"}
DatasetLoader = Callable[..., BenchmarkDataset]
SessionFactory = Callable[[str, str, int | None], InferenceSession]
BenchmarkRunner = Callable[..., BenchmarkSummary]


def validate_benchmark_configuration(
    model: str,
    dataset: str,
    precision: str | None,
) -> tuple[str, str]:
    """Validate safety and return the model family and effective precision."""
    if model not in MODEL_SELECTORS:
        raise ValueError(f"Unknown model selector: {model}")
    if model in UNSAFE_BENCHMARK_MODELS:
        raise ValueError(
            f"{model} is excluded: Qwen restarted the 8 GB Jetson during FP16 loading, "
            "and Phi is beyond the established safe capacity boundary"
        )

    family = "yolo" if model in YOLO_MODELS else "small-vlm"
    validate_dataset_compatibility(family, dataset)
    if family == "yolo":
        if precision is not None:
            raise ValueError("Do not pass --precision for YOLO; its benchmark uses FP16")
        return family, "fp16"

    if precision is None:
        raise ValueError("VLM benchmarks require --precision fp16 or --precision fp32")
    supported = VLM_RUNTIME_PRECISIONS[model]
    if precision not in supported:
        raise ValueError(
            f"{model} does not support {precision}; supported: {', '.join(supported)}"
        )
    return family, precision


def create_inference_session(
    model: str,
    precision: str,
    yolo_image_size: int | None = None,
) -> InferenceSession:
    """Create the validated family-specific loaded-model session."""
    if model in YOLO_MODELS:
        return YoloInferenceSession(
            model,
            image_size=(
                DEFAULT_BENCHMARK_YOLO_IMAGE_SIZE
                if yolo_image_size is None
                else yolo_image_size
            ),
        )
    return VlmInferenceSession(model, precision=precision)


def resolve_checkpoint_revision(
    model: str,
    *,
    yolo_directory: Path = YOLO_MODEL_DIRECTORY,
    vlm_directory: Path = SMALL_VLM_MODEL_DIRECTORY,
) -> str:
    """Return the pinned Hugging Face revision or YOLO checkpoint SHA-256."""
    if model in YOLO_MODELS:
        checkpoint = yolo_directory / YOLO_MODELS[model]
        if not checkpoint.is_file():
            raise FileNotFoundError(f"YOLO checkpoint not found: {checkpoint}")
        digest = hashlib.sha256()
        with checkpoint.open("rb") as checkpoint_file:
            for chunk in iter(lambda: checkpoint_file.read(1024 * 1024), b""):
                digest.update(chunk)
        return f"sha256:{digest.hexdigest()}"

    metadata_path = vlm_directory / model / "download_metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"VLM download metadata not found: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    revision = metadata.get("revision")
    if not isinstance(revision, str) or not revision:
        raise RuntimeError(f"VLM download metadata has no revision: {metadata_path}")
    return revision


def collect_benchmark_runtime_versions(
    session: InferenceSession,
) -> dict[str, str]:
    """Extend shared Jetson metadata with the relevant inference package version."""
    versions = collect_runtime_metadata(session.torch)
    package = "ultralytics" if session.family == "yolo" else "transformers"
    try:
        versions[package] = version(package)
    except PackageNotFoundError:
        versions[package] = "unavailable"
    return versions


def desktop_is_active() -> bool:
    """Return whether the graphical login target is currently active."""
    try:
        result = subprocess.run(
            ["systemctl", "is-active", "--quiet", "graphical.target"],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except OSError:
        return False
    return result.returncode == 0


def benchmark_report_path(
    model: str,
    dataset: str,
    precision: str,
    created_at: datetime,
    output_directory: Path = BENCHMARK_RESULTS_DIRECTORY,
) -> Path:
    """Build the default timestamped benchmark report path."""
    timestamp = created_at.strftime("%Y%m%dT%H%M%SZ")
    return output_directory / f"{model}-{dataset}-{precision}-{timestamp}.json"


def format_progress(result: BenchmarkSampleResult, completed: int, total: int) -> str:
    """Format one compact progress update."""
    return f"[{completed}/{total}] {result.sample_id}: {result.status}"


def main(
    argv: Sequence[str] | None = None,
    *,
    dataset_loader: DatasetLoader = load_benchmark_dataset,
    session_factory: SessionFactory = create_inference_session,
    revision_resolver: Callable[[str], str] = resolve_checkpoint_revision,
    runtime_collector: Callable[[InferenceSession], dict[str, str]] = (
        collect_benchmark_runtime_versions
    ),
    desktop_detector: Callable[[], bool] = desktop_is_active,
    runner: BenchmarkRunner = run_benchmark,
    created_at: datetime | None = None,
) -> int:
    parser = argparse.ArgumentParser(
        prog="./scripts/run_benchmark.sh",
        description="Benchmark one validated model over a complete local dataset.",
    )
    parser.add_argument("model", help="one configured model selector")
    parser.add_argument("dataset", choices=("coco", "imagenette"))
    parser.add_argument(
        "--precision",
        choices=("fp16", "fp32"),
        help="required for VLMs; omit for YOLO",
    )
    parser.add_argument("--warmup", type=int, default=3, help="excluded warm-up runs")
    parser.add_argument("--limit", type=int, help="development-only image limit")
    parser.add_argument(
        "--image-size",
        type=int,
        help="YOLO square input size (default: 640); invalid for VLMs",
    )
    parser.add_argument("--output", type=Path, help="exact destination JSON path")
    arguments = parser.parse_args(argv)

    try:
        family, precision = validate_benchmark_configuration(
            arguments.model,
            arguments.dataset,
            arguments.precision,
        )
        if arguments.warmup < 0:
            raise ValueError("--warmup cannot be negative")
        if arguments.limit is not None and arguments.limit <= 0:
            raise ValueError("--limit must be a positive integer")
        if family == "yolo":
            yolo_image_size = (
                DEFAULT_BENCHMARK_YOLO_IMAGE_SIZE
                if arguments.image_size is None
                else arguments.image_size
            )
            if yolo_image_size <= 0 or yolo_image_size % 32 != 0:
                raise ValueError("--image-size must be a positive multiple of 32")
        else:
            if arguments.image_size is not None:
                raise ValueError("--image-size is only valid for YOLO benchmarks")
            yolo_image_size = None

        dataset = dataset_loader(arguments.dataset, limit=arguments.limit)
        session = session_factory(arguments.model, precision, yolo_image_size)
        desktop_active = desktop_detector()
        metadata = BenchmarkRunMetadata(
            model=arguments.model,
            family=family,
            runtime_precision=precision,
            dataset=arguments.dataset,
            batch_size=1,
            warmup_iterations=arguments.warmup,
            checkpoint_revision=revision_resolver(arguments.model),
            runtime_versions=runtime_collector(session),
            desktop_active=desktop_active,
            dataset_total_images=dataset.total_image_count,
            selected_images=len(dataset),
            requested_limit=dataset.requested_limit,
            run_scope=dataset.run_scope,
            input_profile="fixed-square" if family == "yolo" else "model-native",
            requested_image_size=yolo_image_size,
        )
        report_created_at = created_at or datetime.now(timezone.utc)
        report_path = arguments.output or benchmark_report_path(
            arguments.model,
            arguments.dataset,
            precision,
            report_created_at,
        )
        writer = BenchmarkReportWriter(
            report_path,
            metadata,
            created_at=report_created_at,
        )
        if arguments.model == "smolvlm2-2.2b" and desktop_active:
            print("WARNING: SmolVLM2-2.2B should be benchmarked without the desktop.")
        print(
            f"Benchmarking {arguments.model} ({precision}) on "
            f"{len(dataset)} {arguments.dataset} images"
        )
        print(f"Report: {report_path}")
        progress_interval = max(1, len(dataset) // 10)

        def show_progress(
            result: BenchmarkSampleResult,
            completed: int,
            total: int,
        ) -> None:
            if completed == 1 or completed == total or completed % progress_interval == 0:
                print(format_progress(result, completed, total), flush=True)

        summary = runner(
            session,
            dataset,
            writer,
            warmup_iterations=arguments.warmup,
            progress_callback=show_progress,
        )
    except KeyboardInterrupt:
        print("\nBenchmark interrupted; partial report retained.")
        return 130
    except (BenchmarkExecutionError, OSError, RuntimeError, ValueError) as error:
        print(f"Benchmark failed: {error}")
        return 1

    if summary.median_inference_seconds is None:
        print(f"Completed with no successful images: 0/{len(dataset)}")
    else:
        print(
            f"Completed: {summary.processed_images}/{len(dataset)} images, "
            f"median={summary.median_inference_seconds:.3f}s, "
            f"p95={summary.p95_inference_seconds:.3f}s"
        )
    return 0
