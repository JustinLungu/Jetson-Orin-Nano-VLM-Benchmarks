"""Load-once benchmark execution and compact latency aggregation."""

import math
import statistics
import time
from collections.abc import Callable, Sequence
from typing import Any

from src.benchmark.datasets import BenchmarkImage, DatasetImageError
from src.benchmark.result import (
    BenchmarkReportWriter,
    BenchmarkSampleResult,
    BenchmarkSummary,
)
from src.benchmark.telemetry import BenchmarkTelemetry
from src.inference.base import InferenceSession
from src.inference.runtime import (
    cleanup_cuda,
    is_cuda_out_of_memory,
    measure_cuda_operation,
)

ProgressCallback = Callable[[BenchmarkSampleResult, int, int], None]


class BenchmarkExecutionError(RuntimeError):
    """A fatal benchmark condition stopped the run after checkpointing."""


def run_benchmark(
    session: InferenceSession,
    dataset: Sequence[BenchmarkImage],
    writer: BenchmarkReportWriter,
    *,
    warmup_iterations: int,
    clock: Callable[[], float] = time.perf_counter,
    telemetry: BenchmarkTelemetry | None = None,
    progress_callback: ProgressCallback | None = None,
) -> BenchmarkSummary:
    """Load once, benchmark every image, and atomically checkpoint each result."""
    _validate_run_configuration(session, writer, warmup_iterations)
    results: list[BenchmarkSampleResult] = []
    writer.write(results)
    telemetry = telemetry or BenchmarkTelemetry(session.torch, session.device)
    run_started = clock()
    telemetry_started = False
    try:
        telemetry.start()
        telemetry_started = True
        load_started = clock()
        session.load()
        session.torch.cuda.synchronize()
        model_load_seconds = clock() - load_started
        telemetry.mark_model_loaded()
        _run_warmups(session, dataset, warmup_iterations)

        for sample in dataset:
            result = _run_sample(session, sample, clock)
            results.append(result)
            writer.write(results)
            if progress_callback is not None:
                progress_callback(result, len(results), len(dataset))
            if result.error_type == "cuda_out_of_memory":
                raise BenchmarkExecutionError(
                    f"CUDA out of memory while processing {sample.sample_id}"
                )

        total_run_seconds = clock() - run_started
        summary = aggregate_benchmark_results(
            results,
            model_load_seconds=model_load_seconds,
            total_run_seconds=total_run_seconds,
        )
        telemetry_started = False
        metrics = telemetry.stop()
        writer.write(
            results,
            summary=summary,
            jetson_metrics=metrics,
            run_status="completed",
        )
        return summary
    except BaseException as error:
        metrics = None
        if telemetry_started:
            telemetry_started = False
            try:
                metrics = telemetry.stop()
            except Exception:
                metrics = None
        run_status = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
        error_message = str(error) or "Interrupted by user"
        writer.write(
            results,
            jetson_metrics=metrics,
            run_status=run_status,
            error_message=error_message,
        )
        raise
    finally:
        if telemetry_started:
            telemetry.stop()
        session.close()
        cleanup_cuda(session.torch)
def aggregate_benchmark_results(
    results: Sequence[BenchmarkSampleResult],
    *,
    model_load_seconds: float,
    total_run_seconds: float,
) -> BenchmarkSummary:
    """Aggregate passed inference samples without mixing failures or skips."""
    passed = [result for result in results if result.status == "passed"]
    latencies = [
        result.inference_time_seconds
        for result in passed
        if result.inference_time_seconds is not None
    ]
    total_inference_seconds = sum(latencies)
    generated_tokens = sum(result.generated_tokens or 0 for result in passed)
    return BenchmarkSummary(
        processed_images=len(passed),
        failed_images=sum(result.status == "failed" for result in results),
        skipped_images=sum(result.status == "skipped" for result in results),
        model_load_seconds=model_load_seconds,
        mean_inference_seconds=statistics.fmean(latencies) if latencies else None,
        median_inference_seconds=statistics.median(latencies) if latencies else None,
        p95_inference_seconds=_nearest_rank_percentile(latencies, 0.95),
        total_run_seconds=total_run_seconds,
        images_per_second=(
            len(passed) / total_inference_seconds if total_inference_seconds else 0.0
        ),
        generated_tokens_per_second=(
            generated_tokens / total_inference_seconds
            if generated_tokens and total_inference_seconds
            else None
        ),
    )


def _run_warmups(
    session: InferenceSession,
    dataset: Sequence[BenchmarkImage],
    iterations: int,
) -> None:
    if iterations == 0:
        return
    warmup_sample = _first_readable_sample(dataset)
    if warmup_sample is None:
        raise RuntimeError("Cannot warm up because the dataset has no readable images")

    for _ in range(iterations):
        with warmup_sample.open_rgb() as image:
            prepared = session.prepare(image)
            output = session.infer(prepared)
            session.torch.cuda.synchronize()
            del output, prepared


def _first_readable_sample(
    dataset: Sequence[BenchmarkImage],
) -> BenchmarkImage | None:
    for sample in dataset:
        try:
            with sample.open_rgb():
                return sample
        except DatasetImageError:
            continue
    return None


def _run_sample(
    session: InferenceSession,
    sample: BenchmarkImage,
    clock: Callable[[], float],
) -> BenchmarkSampleResult:
    try:
        with sample.open_rgb() as image:
            prepared = session.prepare(image)
            output, inference_time = measure_cuda_operation(
                lambda: session.infer(prepared),
                session.torch,
                clock=clock,
            )
            _, generated_tokens = session.summarize(output, prepared)
        return BenchmarkSampleResult(
            index=sample.index,
            sample_id=sample.sample_id,
            status="passed",
            inference_time_seconds=inference_time,
            generated_tokens=generated_tokens,
        )
    except DatasetImageError as error:
        return BenchmarkSampleResult(
            index=sample.index,
            sample_id=sample.sample_id,
            status="skipped",
            error_type="unreadable_image",
            error_message=str(error),
        )
    except Exception as error:
        error_type = (
            "cuda_out_of_memory"
            if is_cuda_out_of_memory(error, session.torch)
            else "inference_error"
        )
        return BenchmarkSampleResult(
            index=sample.index,
            sample_id=sample.sample_id,
            status="failed",
            error_type=error_type,
            error_message=str(error),
        )


def _nearest_rank_percentile(values: Sequence[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, math.ceil(percentile * len(ordered)) - 1)
    return ordered[index]


def _validate_run_configuration(
    session: InferenceSession,
    writer: BenchmarkReportWriter,
    warmup_iterations: int,
) -> None:
    metadata = writer.metadata
    if warmup_iterations < 0:
        raise ValueError("warmup_iterations cannot be negative")
    if metadata.warmup_iterations != warmup_iterations:
        raise ValueError("Writer metadata warm-up count does not match the runner")
    if metadata.batch_size != 1:
        raise ValueError("The initial benchmark runner supports only batch size 1")
    expected = (session.selector, session.family, session.precision)
    actual = (metadata.model, metadata.family, metadata.runtime_precision)
    if actual != expected:
        raise ValueError("Writer metadata does not match the inference session")
