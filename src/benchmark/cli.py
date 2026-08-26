"""Run one fixed benchmark configuration for the grouped workflow."""

import subprocess
from datetime import datetime, timezone
from pathlib import Path

from src.benchmark.datasets import load_benchmark_dataset
from src.benchmark.result import BenchmarkReportWriter, BenchmarkRunMetadata
from src.benchmark.runner import run_benchmark
from src.constants import REPOSITORY_ROOT, VLM_RUNTIME_PRECISIONS, YOLO_MODELS
from src.inference.runtime import collect_runtime_metadata
from src.inference.vlm import VlmInferenceSession
from src.inference.yolo import YoloInferenceSession

RESULTS_DIRECTORY = REPOSITORY_ROOT / "results" / "benchmarks"
WARMUP_ITERATIONS = 3
YOLO_IMAGE_SIZE = 640


def run_model_benchmark(
    model: str,
    dataset_name: str,
    precision: str | None,
    limit: int | None,
) -> int:
    """Run one configuration selected by the four benchmark groups."""
    try:
        family, effective_precision = _configuration(model, precision)
        dataset = load_benchmark_dataset(dataset_name, limit=limit)
        session = (
            YoloInferenceSession(model, image_size=YOLO_IMAGE_SIZE)
            if family == "yolo"
            else VlmInferenceSession(model, precision=effective_precision)
        )
        created_at = datetime.now(timezone.utc)
        metadata = BenchmarkRunMetadata(
            model=model,
            family=family,
            runtime_precision=effective_precision,
            dataset=dataset_name,
            warmup_iterations=WARMUP_ITERATIONS,
            runtime_versions=collect_runtime_metadata(session.torch),
            desktop_active=_desktop_is_active(),
            dataset_total_images=dataset.total_image_count,
            selected_images=len(dataset),
            run_scope=dataset.run_scope,
            input_profile="fixed-square" if family == "yolo" else "model-native",
            requested_image_size=YOLO_IMAGE_SIZE if family == "yolo" else None,
        )
        report_path = _report_path(
            model,
            dataset_name,
            effective_precision,
            dataset.run_scope,
            created_at,
        )
        writer = BenchmarkReportWriter(report_path, metadata, created_at=created_at)
        print(
            f"Benchmarking {model} ({effective_precision}) on "
            f"{len(dataset)} {dataset_name} images"
        )
        print(f"Report: {report_path}")
        interval = max(1, len(dataset) // 10)

        def progress(result, completed: int, total: int) -> None:
            if completed == 1 or completed == total or completed % interval == 0:
                print(f"[{completed}/{total}] {result.sample_id}: {result.status}")

        summary = run_benchmark(
            session,
            dataset,
            writer,
            warmup_iterations=WARMUP_ITERATIONS,
            progress_callback=progress,
        )
        print(
            f"Completed: {summary.processed_images}/{len(dataset)} images, "
            f"median={summary.median_inference_seconds:.3f}s, "
            f"p95={summary.p95_inference_seconds:.3f}s"
        )
        return 0
    except KeyboardInterrupt:
        print("\nBenchmark interrupted; partial report retained.")
        return 130
    except (OSError, RuntimeError, ValueError) as error:
        print(f"Benchmark failed: {error}")
        return 1


def _configuration(model: str, precision: str | None) -> tuple[str, str]:
    if model in YOLO_MODELS:
        return "yolo", "fp16"
    supported = VLM_RUNTIME_PRECISIONS.get(model)
    if supported is None or precision not in supported:
        raise ValueError(f"Unsupported benchmark configuration: {model} {precision}")
    return "small-vlm", precision


def _report_path(
    model: str,
    dataset: str,
    precision: str,
    run_scope: str,
    created_at: datetime,
) -> Path:
    directory = RESULTS_DIRECTORY / "full_run" if run_scope == "full" else RESULTS_DIRECTORY
    timestamp = created_at.strftime("%Y%m%dT%H%M%SZ")
    return directory / f"{model}-{dataset}-{precision}-{timestamp}.json"


def _desktop_is_active() -> bool:
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
