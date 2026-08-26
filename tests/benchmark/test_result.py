"""Focused tests for atomic benchmark reports."""

import json
import tempfile
import unittest
from pathlib import Path

from src.benchmark.result import (
    BenchmarkReportWriter,
    BenchmarkRunMetadata,
    BenchmarkSampleResult,
    BenchmarkSummary,
)


def metadata() -> BenchmarkRunMetadata:
    return BenchmarkRunMetadata(
        model="smolvlm2-256m",
        family="small-vlm",
        runtime_precision="fp16",
        dataset="imagenette",
        warmup_iterations=3,
        runtime_versions={"torch": "2.8.0"},
        desktop_active=True,
        dataset_total_images=3925,
        selected_images=1,
        run_scope="limited",
        input_profile="model-native",
        requested_image_size=None,
    )


class BenchmarkResultTests(unittest.TestCase):
    def test_writer_replaces_a_running_checkpoint_with_the_completed_report(self) -> None:
        sample = BenchmarkSampleResult(0, "image.jpg", "passed", 0.5, 16)
        summary = BenchmarkSummary(1, 0, 0, 1.0, 0.5, 0.5, 0.5, 1.5, 2.0, 32.0)
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "benchmark.json"
            writer = BenchmarkReportWriter(destination, metadata())
            writer.write([sample])
            self.assertEqual(
                "running",
                json.loads(destination.read_text(encoding="utf-8"))["run_status"],
            )

            writer.write([sample], summary=summary, run_status="completed")
            report = json.loads(destination.read_text(encoding="utf-8"))
            self.assertFalse(destination.with_suffix(".json.tmp").exists())

        self.assertEqual("completed", report["run_status"])
        self.assertEqual(1, report["summary"]["processed_images"])

    def test_failed_report_keeps_completed_samples_and_error(self) -> None:
        sample = BenchmarkSampleResult(0, "image.jpg", "passed", 0.5, 16)
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "benchmark.json"
            BenchmarkReportWriter(destination, metadata()).write(
                [sample],
                run_status="failed",
                error_message="CUDA out of memory",
            )
            report = json.loads(destination.read_text(encoding="utf-8"))

        self.assertEqual("failed", report["run_status"])
        self.assertEqual(1, len(report["samples"]))
        self.assertEqual("CUDA out of memory", report["error_message"])


if __name__ == "__main__":
    unittest.main()
