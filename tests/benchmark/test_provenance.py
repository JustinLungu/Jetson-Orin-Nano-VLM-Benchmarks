"""Offline tests for repository and Jetson benchmark provenance."""

import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from src.benchmark.provenance import collect_benchmark_provenance


class BenchmarkProvenanceTests(unittest.TestCase):
    def test_collects_revision_power_mode_and_locked_clocks(self) -> None:
        runner = Mock(
            side_effect=(
                subprocess.CompletedProcess([], 0, "abc123\n", ""),
                subprocess.CompletedProcess([], 0, "NV Power Mode: 25W\n1\n", ""),
            )
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cpu = root / "cpu"
            gpu = root / "gpu"
            cpu.mkdir()
            gpu.mkdir()
            (cpu / "scaling_min_freq").write_text("1344000\n", encoding="utf-8")
            (cpu / "scaling_max_freq").write_text("1344000\n", encoding="utf-8")
            (gpu / "min_freq").write_text("918000000\n", encoding="utf-8")
            (gpu / "max_freq").write_text("918000000\n", encoding="utf-8")

            result = collect_benchmark_provenance(
                repository_root=root,
                command_runner=runner,
                cpu_frequency_directory=cpu,
                gpu_frequency_directory=gpu,
            )

        self.assertEqual("abc123", result.repository_revision)
        self.assertEqual("25W", result.power_mode)
        self.assertTrue(result.clocks_locked)

    def test_unavailable_commands_and_unlocked_clocks_are_explicit(self) -> None:
        runner = Mock(side_effect=OSError("missing"))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cpu = root / "cpu"
            gpu = root / "gpu"
            cpu.mkdir()
            gpu.mkdir()
            (cpu / "scaling_min_freq").write_text("729600\n", encoding="utf-8")
            (cpu / "scaling_max_freq").write_text("1344000\n", encoding="utf-8")
            (gpu / "min_freq").write_text("306000000\n", encoding="utf-8")
            (gpu / "max_freq").write_text("918000000\n", encoding="utf-8")

            result = collect_benchmark_provenance(
                repository_root=root,
                command_runner=runner,
                cpu_frequency_directory=cpu,
                gpu_frequency_directory=gpu,
            )

        self.assertEqual("unavailable", result.repository_revision)
        self.assertEqual("unavailable", result.power_mode)
        self.assertFalse(result.clocks_locked)


if __name__ == "__main__":
    unittest.main()
