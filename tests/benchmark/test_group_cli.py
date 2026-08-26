"""Offline tests for grouped benchmark orchestration."""

import subprocess
import unittest
from pathlib import Path
from unittest.mock import Mock, call

from src.benchmark.group_cli import main

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
GROUP_SCRIPT = REPOSITORY_ROOT / "scripts/run_benchmark_group.sh"


class BenchmarkGroupCliTests(unittest.TestCase):
    def test_yolo_group_expands_in_stable_order(self) -> None:
        command = Mock(return_value=0)

        exit_code = main(
            ["yolo", "imagenette", "--limit", "20"],
            benchmark_command=command,
        )

        self.assertEqual(0, exit_code)
        self.assertEqual(
            [
                call(model, "imagenette", None, 20)
                for model in ("yolov8n", "yolo11n", "yolo26n")
            ],
            command.call_args_list,
        )

    def test_smolvlm_group_includes_every_safe_precision(self) -> None:
        command = Mock(return_value=0)

        exit_code = main(
            ["smolvlm", "coco", "--limit", "10"],
            benchmark_command=command,
        )

        self.assertEqual(0, exit_code)
        self.assertEqual(
            [
                call("smolvlm2-2.2b", "coco", "fp16", 10),
                call("smolvlm2-256m", "coco", "fp16", 10),
                call("smolvlm2-256m", "coco", "fp32", 10),
                call("smolvlm2-500m", "coco", "fp16", 10),
                call("smolvlm2-500m", "coco", "fp32", 10),
            ],
            command.call_args_list,
        )

    def test_group_stops_after_the_first_failed_configuration(self) -> None:
        command = Mock(side_effect=(0, 1))

        exit_code = main(
            ["yolo", "coco"],
            benchmark_command=command,
        )

        self.assertEqual(1, exit_code)
        self.assertEqual(2, command.call_count)

    def test_shell_entry_point_has_valid_syntax(self) -> None:
        subprocess.run(["bash", "-n", str(GROUP_SCRIPT)], check=True)


if __name__ == "__main__":
    unittest.main()
