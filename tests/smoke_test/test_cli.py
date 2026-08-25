"""Tests for sequential smoke-test orchestration and reporting."""

import json
import subprocess
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import Mock

from src.constants import MODEL_SELECTORS
from src.smoke_test.cli import main, run_selected_models, select_smoke_models
from src.smoke_test.result import SmokeTestResult

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SMOKE_SCRIPT = REPOSITORY_ROOT / "scripts/smoke_test_models.sh"
JETSON_METRICS = {
    "cpu_utilization_percent": {"average": 75.0, "peak": 85.0},
    "gpu_utilization_percent": {"average": 30.0, "peak": 47.0},
    "power_watts": {"average": 8.3, "peak": 8.5},
    "temperature_celsius": {"average": 48.5, "peak": 49.3},
}


class FakeMonitor:
    def start(self) -> None:
        pass

    def stop(self):
        return JETSON_METRICS


def make_result(model: str, family: str, status: str = "passed") -> SmokeTestResult:
    arguments = {
        "model": model,
        "family": family,
        "status": status,
        "device": "cuda:0",
        "runtime_versions": {"torch": "2.8.0", "cuda": "12.6"},
    }
    if status == "passed":
        arguments.update(
            inference_time_seconds=0.1,
            peak_cuda_memory_mib=64.0,
            prediction_summary="ok",
        )
    else:
        arguments.update(error_type="cuda_out_of_memory", error_message="OOM")
    return SmokeTestResult(**arguments)


class SmokeSelectionTests(unittest.TestCase):
    def test_all_uses_configured_registry_order(self) -> None:
        self.assertEqual(list(MODEL_SELECTORS), select_smoke_models(["all"]))

    def test_family_and_individual_selector_are_deduplicated(self) -> None:
        self.assertEqual(
            ["yolov8n", "yolo11n", "yolo26n"],
            select_smoke_models(["yolo", "yolo11n"]),
        )


class SequentialRunnerTests(unittest.TestCase):
    def test_models_run_sequentially_and_continue_after_failure(self) -> None:
        calls = []

        def yolo_runner(selector: str, image: Path) -> SmokeTestResult:
            calls.append(selector)
            status = "failed" if selector == "yolov8n" else "passed"
            return make_result(selector, "yolo", status)

        results = run_selected_models(
            ["yolov8n", "yolo11n"],
            Path("fixture.ppm"),
            runners={"yolo": yolo_runner, "small-vlm": Mock()},
            monitor_factory=FakeMonitor,
        )

        self.assertEqual(["yolov8n", "yolo11n"], calls)
        self.assertEqual(["failed", "passed"], [result.status for result in results])

    def test_unexpected_runner_error_is_recorded_and_does_not_stop_run(self) -> None:
        runner = Mock(side_effect=(RuntimeError("broken"), make_result("yolo11n", "yolo")))

        results = run_selected_models(
            ["yolov8n", "yolo11n"],
            Path("fixture.ppm"),
            runners={"yolo": runner, "small-vlm": Mock()},
            monitor_factory=FakeMonitor,
        )

        self.assertEqual("runner_error", results[0].error_type)
        self.assertEqual("passed", results[1].status)


class SmokeCliTests(unittest.TestCase):
    def test_main_writes_timestamped_json_and_returns_failure(self) -> None:
        created_at = datetime(2026, 8, 25, 12, 30, tzinfo=timezone.utc)
        runner = Mock(return_value=make_result("yolov8n", "yolo", "failed"))

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            exit_code = main(
                ["yolov8n"],
                runners={"yolo": runner, "small-vlm": Mock()},
                output_directory=output,
                created_at=created_at,
                monitor_factory=FakeMonitor,
            )
            report_path = output / "smoke-20260825T123000Z.json"
            report = json.loads(report_path.read_text(encoding="utf-8"))

        self.assertEqual(1, exit_code)
        self.assertEqual(1, report["schema_version"])
        self.assertTrue(report["run_completed"])
        self.assertEqual("failed", report["results"][0]["status"])
        self.assertEqual(JETSON_METRICS, report["results"][0]["jetson_metrics"])

    def test_report_is_checkpointed_after_each_model(self) -> None:
        created_at = datetime(2026, 8, 25, 12, 30, tzinfo=timezone.utc)

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            report_path = output / "smoke-20260825T123000Z.json"

            def runner(selector: str, image: Path) -> SmokeTestResult:
                if selector == "yolo11n":
                    checkpoint = json.loads(report_path.read_text(encoding="utf-8"))
                    self.assertEqual(["yolov8n"], [item["model"] for item in checkpoint["results"]])
                    self.assertEqual(["yolov8n", "yolo11n"], checkpoint["selected_models"])
                    self.assertFalse(checkpoint["run_completed"])
                return make_result(selector, "yolo")

            exit_code = main(
                ["yolov8n", "yolo11n"],
                runners={"yolo": runner, "small-vlm": Mock()},
                output_directory=output,
                created_at=created_at,
                monitor_factory=FakeMonitor,
            )

            report = json.loads(report_path.read_text(encoding="utf-8"))

        self.assertEqual(0, exit_code)
        self.assertTrue(report["run_completed"])
        self.assertEqual(["yolov8n", "yolo11n"], [item["model"] for item in report["results"]])

    def test_shell_entry_point_has_valid_syntax_and_preserves_environment(self) -> None:
        subprocess.run(["bash", "-n", str(SMOKE_SCRIPT)], check=True)
        script = SMOKE_SCRIPT.read_text(encoding="utf-8")
        self.assertIn("--no-sync", script)


if __name__ == "__main__":
    unittest.main()
