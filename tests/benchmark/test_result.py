"""Tests for checkpointed benchmark result contracts."""

import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from src.benchmark.result import (
    BenchmarkReportWriter,
    BenchmarkRunMetadata,
    BenchmarkSampleResult,
    BenchmarkSummary,
)
from src.benchmark.provenance import BenchmarkProvenance


def metadata() -> BenchmarkRunMetadata:
    return BenchmarkRunMetadata(
        model="smolvlm2-256m",
        family="small-vlm",
        runtime_precision="fp16",
        dataset="imagenette",
        batch_size=1,
        warmup_iterations=2,
        checkpoint_revision="test-revision",
        runtime_versions={"torch": "2.8.0", "cuda": "12.6"},
        desktop_active=False,
        dataset_total_images=100,
        selected_images=1,
        requested_limit=1,
        run_scope="limited",
        input_profile="model-native",
        requested_image_size=None,
        provenance=BenchmarkProvenance("abc123", "25W", True),
    )


def passed_sample(index: int = 0) -> BenchmarkSampleResult:
    return BenchmarkSampleResult(
        index=index,
        sample_id=f"image-{index}.JPEG",
        status="passed",
        inference_time_seconds=0.5,
        generated_tokens=16,
    )


def summary(sample_count: int = 1) -> BenchmarkSummary:
    return BenchmarkSummary(
        processed_images=sample_count,
        failed_images=0,
        skipped_images=0,
        model_load_seconds=1.25,
        mean_inference_seconds=0.5,
        median_inference_seconds=0.5,
        p95_inference_seconds=0.5,
        total_run_seconds=0.75,
        images_per_second=2.0,
        generated_tokens_per_second=32.0,
    )


class BenchmarkResultTests(unittest.TestCase):
    def test_passed_sample_requires_nonnegative_latency(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires inference_time_seconds"):
            BenchmarkSampleResult(index=0, sample_id="image.jpg", status="passed")
        with self.assertRaisesRegex(ValueError, "cannot be negative"):
            BenchmarkSampleResult(
                index=0,
                sample_id="image.jpg",
                status="passed",
                inference_time_seconds=-0.1,
            )

    def test_skipped_sample_requires_a_reason(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires an error type"):
            BenchmarkSampleResult(index=0, sample_id="bad.jpg", status="skipped")

    def test_writer_checkpoints_and_completes_same_report(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "nested" / "benchmark.json"
            writer = BenchmarkReportWriter(
                destination,
                metadata(),
                created_at=datetime(2026, 8, 26, 12, 0, tzinfo=timezone.utc),
            )
            writer.write([passed_sample()])

            checkpoint = json.loads(destination.read_text(encoding="utf-8"))
            self.assertEqual("running", checkpoint["run_status"])
            self.assertIsNone(checkpoint["summary"])
            self.assertEqual("fp16", checkpoint["metadata"]["runtime_precision"])
            self.assertEqual(0.5, checkpoint["samples"][0]["inference_time_seconds"])
            self.assertFalse(destination.with_suffix(".json.tmp").exists())

            writer.write(
                [passed_sample()],
                summary=summary(),
                run_status="completed",
            )
            completed = json.loads(destination.read_text(encoding="utf-8"))

        self.assertEqual(4, completed["schema_version"])
        self.assertEqual(1, completed["summary"]["processed_images"])
        self.assertEqual("limited", completed["metadata"]["run_scope"])
        self.assertEqual("completed", completed["run_status"])
        self.assertEqual("abc123", completed["metadata"]["provenance"]["repository_revision"])

    def test_metadata_scope_must_match_requested_limit(self) -> None:
        values = metadata()
        with self.assertRaisesRegex(ValueError, "does not match"):
            BenchmarkRunMetadata(
                model=values.model,
                family=values.family,
                runtime_precision=values.runtime_precision,
                dataset=values.dataset,
                batch_size=values.batch_size,
                warmup_iterations=values.warmup_iterations,
                checkpoint_revision=values.checkpoint_revision,
                runtime_versions=values.runtime_versions,
                desktop_active=values.desktop_active,
                dataset_total_images=values.dataset_total_images,
                selected_images=values.selected_images,
                requested_limit=values.requested_limit,
                run_scope="full",
                input_profile=values.input_profile,
                requested_image_size=values.requested_image_size,
                provenance=values.provenance,
            )

    def test_completed_report_requires_matching_summary(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            writer = BenchmarkReportWriter(Path(directory) / "report.json", metadata())
            with self.assertRaisesRegex(ValueError, "counts must equal"):
                writer.write(
                    [passed_sample()],
                    summary=summary(sample_count=2),
                    run_status="completed",
                )

    def test_failed_report_requires_and_persists_a_termination_reason(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "report.json"
            writer = BenchmarkReportWriter(destination, metadata())
            writer.write(
                [passed_sample()],
                run_status="failed",
                error_message="load failed",
            )
            report = json.loads(destination.read_text(encoding="utf-8"))

        self.assertEqual("failed", report["run_status"])
        self.assertEqual("load failed", report["error_message"])

    def test_sample_indices_must_be_contiguous(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            writer = BenchmarkReportWriter(Path(directory) / "report.json", metadata())
            with self.assertRaisesRegex(ValueError, "contiguous"):
                writer.write([passed_sample(index=1)])


if __name__ == "__main__":
    unittest.main()
