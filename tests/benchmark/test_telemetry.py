"""Offline tests for compact Jetson benchmark telemetry."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from src.benchmark.telemetry import (
    BenchmarkTelemetry,
    MemorySnapshot,
    read_memory_snapshot,
)
from src.inference.runtime import parse_tegrastats_line, summarize_tegrastats


class FakeMonitor:
    def __init__(self, samples) -> None:
        self.samples = samples
        self.start_calls = 0
        self.stop_calls = 0

    def start(self) -> None:
        self.start_calls += 1

    def stop(self):
        self.stop_calls += 1
        return summarize_tegrastats(self.samples)


class BenchmarkTelemetryTests(unittest.TestCase):
    SAMPLE_ONE = (
        "RAM 2100/7620MB SWAP 100/3810MB CPU [20%@729,30%@729] "
        "GR3D_FREQ 40%@[712] cpu@47.0C gpu@48.0C VDD_IN 6000mW/5000mW"
    )
    SAMPLE_TWO = (
        "RAM 3500/7620MB SWAP 250/3810MB CPU [40%@729,50%@729] "
        "GR3D_FREQ 80%@[918] cpu@49.0C gpu@50.0C VDD_IN 9000mW/6000mW"
    )

    def test_collects_compact_memory_power_and_temperature_metrics(self) -> None:
        samples = [
            parse_tegrastats_line(self.SAMPLE_ONE),
            parse_tegrastats_line(self.SAMPLE_TWO),
        ]
        monitor = FakeMonitor(samples)
        memory_reader = Mock(
            side_effect=(
                MemorySnapshot(7620, 1700, 50),
                MemorySnapshot(7620, 2300, 120),
            )
        )
        cuda = Mock()
        cuda.max_memory_allocated.return_value = 512 * 1024**2
        telemetry = BenchmarkTelemetry(
            SimpleNamespace(cuda=cuda),
            monitor=monitor,
            memory_reader=memory_reader,
        )

        telemetry.start()
        telemetry.mark_model_loaded()
        metrics = telemetry.stop()

        self.assertEqual(1700, metrics.ram_before_load_mib)
        self.assertEqual(2300, metrics.ram_after_load_mib)
        self.assertEqual(3500, metrics.peak_ram_used_mib)
        self.assertEqual(250, metrics.peak_swap_used_mib)
        self.assertEqual(512, metrics.peak_cuda_memory_mib)
        self.assertEqual(7.5, metrics.average_power_watts)
        self.assertEqual(9.0, metrics.peak_power_watts)
        self.assertEqual(50.0, metrics.peak_temperature_celsius)
        self.assertTrue(metrics.tegrastats_available)
        cuda.reset_peak_memory_stats.assert_called_once_with("cuda:0")
        self.assertEqual(1, monitor.start_calls)
        self.assertEqual(1, monitor.stop_calls)

    def test_missing_tegrastats_uses_memory_fallback_and_warns(self) -> None:
        monitor = FakeMonitor([])
        cuda = Mock()
        cuda.max_memory_allocated.return_value = 64 * 1024**2
        telemetry = BenchmarkTelemetry(
            SimpleNamespace(cuda=cuda),
            monitor=monitor,
            memory_reader=Mock(
                side_effect=(
                    MemorySnapshot(7620, 1700, 10),
                    MemorySnapshot(7620, 2200, 20),
                )
            ),
        )

        telemetry.start()
        telemetry.mark_model_loaded()
        with self.assertWarnsRegex(RuntimeWarning, "tegrastats telemetry is unavailable"):
            metrics = telemetry.stop()

        self.assertFalse(metrics.tegrastats_available)
        self.assertEqual(2200, metrics.peak_ram_used_mib)
        self.assertEqual(20, metrics.peak_swap_used_mib)
        self.assertIsNone(metrics.average_power_watts)
        self.assertIsNone(metrics.peak_temperature_celsius)

    def test_reads_linux_memory_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            meminfo = Path(directory) / "meminfo"
            meminfo.write_text(
                "MemTotal:       7802880 kB\n"
                "MemAvailable:   6062080 kB\n"
                "SwapTotal:      3901440 kB\n"
                "SwapFree:       3799040 kB\n",
                encoding="utf-8",
            )
            snapshot = read_memory_snapshot(meminfo)

        self.assertEqual(7620, snapshot.ram_total_mib)
        self.assertEqual(1700, snapshot.ram_used_mib)
        self.assertEqual(100, snapshot.swap_used_mib)


if __name__ == "__main__":
    unittest.main()
