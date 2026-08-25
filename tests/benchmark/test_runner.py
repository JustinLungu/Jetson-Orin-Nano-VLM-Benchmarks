"""Offline tests for benchmark execution and latency aggregation."""

import json
import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import Mock

from PIL import Image

from src.benchmark.datasets import BenchmarkImage
from src.benchmark.result import (
    BenchmarkReportWriter,
    BenchmarkRunMetadata,
    BenchmarkSampleResult,
)
from src.benchmark.runner import (
    BenchmarkExecutionError,
    aggregate_benchmark_results,
    run_benchmark,
)
from src.inference.base import InferenceSession


class FakeOutOfMemoryError(RuntimeError):
    pass


class FakeCuda:
    OutOfMemoryError = FakeOutOfMemoryError

    def __init__(self) -> None:
        self.synchronizations = 0
        self.empty_cache_calls = 0

    def synchronize(self) -> None:
        self.synchronizations += 1

    @staticmethod
    def is_available() -> bool:
        return True

    def empty_cache(self) -> None:
        self.empty_cache_calls += 1


class FakeTorch:
    def __init__(self) -> None:
        self.cuda = FakeCuda()
        self.inference_mode = nullcontext


class FakeSession(InferenceSession):
    family = "small-vlm"
    precision = "fp16"

    def __init__(self) -> None:
        super().__init__("smolvlm2-256m", torch_module=FakeTorch())
        self.load_calls = 0
        self.inference_calls = 0
        self.close_calls = 0

    def load(self) -> None:
        self.load_calls += 1
        self.model = object()

    def prepare(self, image: Image.Image) -> object:
        return object()

    def infer(self, prepared: object) -> object:
        self.inference_calls += 1
        return object()

    def summarize(self, output: object, prepared: object) -> tuple[str, int]:
        return "complete", 2

    def close(self) -> None:
        self.close_calls += 1
        super().close()


def metadata(warmups: int = 1) -> BenchmarkRunMetadata:
    return BenchmarkRunMetadata(
        model="smolvlm2-256m",
        family="small-vlm",
        runtime_precision="fp16",
        dataset="imagenette",
        batch_size=1,
        warmup_iterations=warmups,
        checkpoint_revision="revision",
        runtime_versions={},
        desktop_active=False,
    )


def write_images(directory: Path, count: int) -> tuple[BenchmarkImage, ...]:
    samples = []
    for index in range(count):
        path = directory / f"image-{index}.png"
        with Image.new("RGB", (1, 1), "red") as image:
            image.save(path)
        samples.append(BenchmarkImage(index, path.name, path))
    return tuple(samples)


class BenchmarkRunnerTests(unittest.TestCase):
    def test_loads_once_excludes_warmup_and_aggregates_latency(self) -> None:
        session = FakeSession()
        clock = Mock(
            side_effect=(
                0.0,
                0.1,
                0.6,
                1.0,
                1.1,
                1.2,
                1.4,
                1.5,
                1.8,
                2.0,
            )
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            destination = root / "benchmark.json"
            writer = BenchmarkReportWriter(destination, metadata())
            summary = run_benchmark(
                session,
                write_images(root, 3),
                writer,
                warmup_iterations=1,
                clock=clock,
            )
            report = json.loads(destination.read_text(encoding="utf-8"))

        self.assertEqual(1, session.load_calls)
        self.assertEqual(4, session.inference_calls)
        self.assertEqual(1, session.close_calls)
        self.assertEqual(8, session.torch.cuda.synchronizations)
        self.assertEqual(1, session.torch.cuda.empty_cache_calls)
        self.assertAlmostEqual(0.5, summary.model_load_seconds)
        self.assertAlmostEqual(0.2, summary.mean_inference_seconds)
        self.assertAlmostEqual(0.2, summary.median_inference_seconds)
        self.assertAlmostEqual(0.3, summary.p95_inference_seconds)
        self.assertAlmostEqual(5.0, summary.images_per_second)
        self.assertAlmostEqual(10.0, summary.generated_tokens_per_second)
        self.assertEqual(3, len(report["samples"]))
        self.assertTrue(report["run_completed"])

    def test_load_failure_preserves_empty_incomplete_report(self) -> None:
        session = FakeSession()
        session.load = Mock(side_effect=RuntimeError("load failed"))
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "benchmark.json"
            writer = BenchmarkReportWriter(destination, metadata(warmups=0))
            with self.assertRaisesRegex(RuntimeError, "load failed"):
                run_benchmark(
                    session,
                    (),
                    writer,
                    warmup_iterations=0,
                    clock=Mock(side_effect=(0.0, 0.1)),
                )
            report = json.loads(destination.read_text(encoding="utf-8"))

        self.assertFalse(report["run_completed"])
        self.assertEqual([], report["samples"])
        self.assertEqual(1, session.close_calls)

    def test_aggregation_separates_failures_and_skips(self) -> None:
        results = [
            BenchmarkSampleResult(0, "ok.jpg", "passed", 0.25),
            BenchmarkSampleResult(
                1,
                "failed.jpg",
                "failed",
                error_type="inference_error",
            ),
            BenchmarkSampleResult(
                2,
                "bad.jpg",
                "skipped",
                error_type="unreadable_image",
            ),
        ]

        summary = aggregate_benchmark_results(
            results,
            model_load_seconds=1.0,
            total_run_seconds=2.0,
        )

        self.assertEqual(1, summary.processed_images)
        self.assertEqual(1, summary.failed_images)
        self.assertEqual(1, summary.skipped_images)
        self.assertEqual(4.0, summary.images_per_second)
        self.assertIsNone(summary.generated_tokens_per_second)

    def test_cuda_oom_is_checkpointed_and_stops_the_run(self) -> None:
        session = FakeSession()
        session.infer = Mock(side_effect=FakeOutOfMemoryError("CUDA out of memory"))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            destination = root / "benchmark.json"
            writer = BenchmarkReportWriter(destination, metadata(warmups=0))
            with self.assertRaisesRegex(BenchmarkExecutionError, "CUDA out of memory"):
                run_benchmark(
                    session,
                    write_images(root, 2),
                    writer,
                    warmup_iterations=0,
                    clock=Mock(side_effect=(0.0, 0.1, 0.2, 0.3)),
                )
            report = json.loads(destination.read_text(encoding="utf-8"))

        self.assertFalse(report["run_completed"])
        self.assertEqual(1, len(report["samples"]))
        self.assertEqual("cuda_out_of_memory", report["samples"][0]["error_type"])
        self.assertEqual(1, session.infer.call_count)

    def test_runner_rejects_metadata_mismatch_before_writing(self) -> None:
        session = FakeSession()
        wrong_metadata = metadata(warmups=0)
        with tempfile.TemporaryDirectory() as directory:
            writer = BenchmarkReportWriter(Path(directory) / "report.json", wrong_metadata)
            with self.assertRaisesRegex(ValueError, "warm-up count"):
                run_benchmark(session, (), writer, warmup_iterations=1)


if __name__ == "__main__":
    unittest.main()
