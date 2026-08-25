"""Offline tests for the safe performance benchmark CLI."""

import hashlib
import json
import subprocess
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from src.benchmark.cli import (
    benchmark_report_path,
    main,
    resolve_checkpoint_revision,
    validate_benchmark_configuration,
)
from src.benchmark.result import BenchmarkSummary

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_SCRIPT = REPOSITORY_ROOT / "scripts/run_benchmark.sh"


def completed_summary() -> BenchmarkSummary:
    return BenchmarkSummary(
        processed_images=2,
        failed_images=0,
        skipped_images=0,
        model_load_seconds=1.0,
        mean_inference_seconds=0.5,
        median_inference_seconds=0.5,
        p95_inference_seconds=0.6,
        total_run_seconds=2.0,
        images_per_second=2.0,
        generated_tokens_per_second=32.0,
    )


class BenchmarkConfigurationTests(unittest.TestCase):
    def test_validates_supported_precision_and_dataset(self) -> None:
        self.assertEqual(
            ("small-vlm", "fp32"),
            validate_benchmark_configuration(
                "smolvlm2-256m",
                "imagenette",
                "fp32",
            ),
        )
        self.assertEqual(
            ("yolo", "fp16"),
            validate_benchmark_configuration("yolo11n", "coco", None),
        )

    def test_rejects_unsafe_models_and_invalid_combinations(self) -> None:
        with self.assertRaisesRegex(ValueError, "excluded"):
            validate_benchmark_configuration("qwen2.5-vl-3b", "imagenette", "fp16")
        with self.assertRaisesRegex(ValueError, "require --precision"):
            validate_benchmark_configuration("smolvlm2-500m", "imagenette", None)
        with self.assertRaisesRegex(ValueError, "does not support fp32"):
            validate_benchmark_configuration("smolvlm2-2.2b", "imagenette", "fp32")
        with self.assertRaisesRegex(ValueError, "does not support dataset"):
            validate_benchmark_configuration("yolo11n", "imagenette", None)
        with self.assertRaisesRegex(ValueError, "Do not pass --precision"):
            validate_benchmark_configuration("yolo11n", "coco", "fp16")


class CheckpointRevisionTests(unittest.TestCase):
    def test_reads_vlm_revision_and_hashes_yolo_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            vlm = root / "vlm/smolvlm2-256m"
            vlm.mkdir(parents=True)
            (vlm / "download_metadata.json").write_text(
                json.dumps({"revision": "pinned-revision"}),
                encoding="utf-8",
            )
            yolo = root / "yolo"
            yolo.mkdir()
            checkpoint = yolo / "yolo11n.pt"
            checkpoint.write_bytes(b"checkpoint")

            vlm_revision = resolve_checkpoint_revision(
                "smolvlm2-256m",
                vlm_directory=root / "vlm",
            )
            yolo_revision = resolve_checkpoint_revision(
                "yolo11n",
                yolo_directory=yolo,
            )

        self.assertEqual("pinned-revision", vlm_revision)
        self.assertEqual(
            f"sha256:{hashlib.sha256(b'checkpoint').hexdigest()}",
            yolo_revision,
        )


class BenchmarkCliTests(unittest.TestCase):
    def test_runs_validated_configuration_with_expected_metadata(self) -> None:
        dataset = (SimpleNamespace(sample_id="one"), SimpleNamespace(sample_id="two"))
        session = SimpleNamespace()
        runner = Mock(return_value=completed_summary())
        created_at = datetime(2026, 8, 26, 12, 30, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "report.json"
            exit_code = main(
                [
                    "smolvlm2-256m",
                    "imagenette",
                    "--precision",
                    "fp16",
                    "--warmup",
                    "2",
                    "--limit",
                    "2",
                    "--output",
                    str(output),
                ],
                dataset_loader=Mock(return_value=dataset),
                session_factory=Mock(return_value=session),
                revision_resolver=Mock(return_value="revision"),
                runtime_collector=Mock(return_value={"torch": "2.8.0"}),
                desktop_detector=Mock(return_value=False),
                runner=runner,
                created_at=created_at,
            )

        self.assertEqual(0, exit_code)
        call = runner.call_args
        self.assertIs(session, call.args[0])
        self.assertEqual(dataset, call.args[1])
        self.assertEqual(2, call.kwargs["warmup_iterations"])
        writer = call.args[2]
        self.assertEqual(output, writer.destination)
        self.assertEqual("fp16", writer.metadata.runtime_precision)
        self.assertEqual("revision", writer.metadata.checkpoint_revision)

    def test_unsafe_model_is_rejected_before_dependencies_are_called(self) -> None:
        dataset_loader = Mock()
        session_factory = Mock()

        exit_code = main(
            ["qwen2.5-vl-3b", "imagenette", "--precision", "fp16"],
            dataset_loader=dataset_loader,
            session_factory=session_factory,
        )

        self.assertEqual(1, exit_code)
        dataset_loader.assert_not_called()
        session_factory.assert_not_called()

    def test_default_report_path_contains_configuration(self) -> None:
        path = benchmark_report_path(
            "yolo11n",
            "coco",
            "fp16",
            datetime(2026, 8, 26, 12, 30, tzinfo=timezone.utc),
            Path("results"),
        )

        self.assertEqual(
            Path("results/yolo11n-coco-fp16-20260826T123000Z.json"),
            path,
        )

    def test_shell_entry_point_has_valid_syntax(self) -> None:
        subprocess.run(["bash", "-n", str(BENCHMARK_SCRIPT)], check=True)


if __name__ == "__main__":
    unittest.main()
