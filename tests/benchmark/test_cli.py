"""Tests for the fixed benchmark configuration."""

import unittest
from datetime import datetime, timezone
from unittest.mock import Mock

from src.benchmark.cli import _configuration, _report_path, main


class BenchmarkConfigurationTests(unittest.TestCase):
    def test_group_configurations_resolve_expected_precision(self) -> None:
        self.assertEqual(("yolo", "fp16"), _configuration("yolo11n", None))
        self.assertEqual(
            ("small-vlm", "fp32"),
            _configuration("smolvlm2-500m", "fp32"),
        )
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            _configuration("smolvlm2-2.2b", "fp32")

    def test_full_reports_use_the_full_run_directory(self) -> None:
        path = _report_path(
            "yolo11n",
            "coco",
            "fp16",
            "full",
            datetime(2026, 8, 26, 12, 30, tzinfo=timezone.utc),
        )
        self.assertEqual("full_run", path.parent.name)
        self.assertEqual(
            "yolo11n-coco-fp16-20260826T123000Z.json",
            path.name,
        )

    def test_single_model_cli_runs_only_the_requested_configuration(self) -> None:
        command = Mock(return_value=0)

        exit_code = main(
            [
                "smolvlm2-2.2b",
                "imagenette",
                "--precision",
                "fp16",
                "--limit",
                "1",
            ],
            benchmark_command=command,
        )

        self.assertEqual(0, exit_code)
        command.assert_called_once_with("smolvlm2-2.2b", "imagenette", "fp16", 1)


if __name__ == "__main__":
    unittest.main()
