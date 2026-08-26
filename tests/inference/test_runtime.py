"""Offline tests for shared inference runtime utilities."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from src.inference.runtime import (
    cleanup_cuda,
    collect_runtime_metadata,
    measure_cuda_operation,
    parse_tegrastats_line,
    summarize_tegrastats,
)


class CudaTimingTests(unittest.TestCase):
    def test_measurement_synchronizes_around_operation(self) -> None:
        events = []
        cuda = SimpleNamespace(synchronize=lambda: events.append("synchronize"))
        clock = Mock(side_effect=(10.0, 10.25))

        result, elapsed = measure_cuda_operation(
            lambda: events.append("operation") or "prediction",
            SimpleNamespace(cuda=cuda),
            clock=clock,
        )

        self.assertEqual("prediction", result)
        self.assertEqual(0.25, elapsed)
        self.assertEqual(["synchronize", "operation", "synchronize"], events)

class CudaCleanupTests(unittest.TestCase):
    def test_cleanup_collects_objects_before_emptying_cuda_cache(self) -> None:
        events = []
        cuda = SimpleNamespace(
            is_available=lambda: True,
            empty_cache=lambda: events.append("empty_cache"),
        )

        cleanup_cuda(
            SimpleNamespace(cuda=cuda),
            collector=lambda: events.append("collect"),
        )

        self.assertEqual(["collect", "empty_cache"], events)

class RuntimeMetadataTests(unittest.TestCase):
    def test_collects_jetson_and_torch_metadata(self) -> None:
        cuda = SimpleNamespace(
            is_available=lambda: True,
            get_device_name=lambda index: "Orin",
        )
        torch = SimpleNamespace(
            __version__="2.8.0",
            version=SimpleNamespace(cuda="12.6"),
            cuda=cuda,
        )

        with tempfile.TemporaryDirectory() as directory:
            release_path = Path(directory) / "nv_tegra_release"
            release_path.write_text(
                "# R36 (release), REVISION: 4.7, BOARD: generic\n",
                encoding="utf-8",
            )
            with patch("src.inference.runtime.platform.machine", return_value="aarch64"), patch(
                "src.inference.runtime.platform.python_version",
                return_value="3.10.12",
            ):
                metadata = collect_runtime_metadata(
                    torch,
                    tegra_release_path=release_path,
                )

        self.assertEqual("aarch64", metadata["architecture"])
        self.assertEqual("3.10.12", metadata["python"])
        self.assertEqual("2.8.0", metadata["torch"])
        self.assertEqual("12.6", metadata["cuda"])
        self.assertEqual("true", metadata["cuda_available"])
        self.assertEqual("Orin", metadata["device"])
        self.assertEqual("6.2.x", metadata["jetpack"])

class TegrastatsParsingTests(unittest.TestCase):
    SAMPLE_ONE = (
        "RAM 3415/7620MB SWAP 654/3810MB CPU [75%@1344,85%@1344,off,79%@1344] "
        "GR3D_FREQ 13%@[917] cpu@49.031C gpu@48.406C tj@49.031C "
        "VDD_IN 8527mW/4783mW"
    )
    SAMPLE_TWO = (
        "RAM 3488/7620MB SWAP 700/3810MB CPU [81%@1344,68%@1344,66%@1344] "
        "GR3D_FREQ 47%@[712] cpu@49.343C gpu@48.468C tj@49.156C "
        "VDD_IN 8147mW/4814mW"
    )

    def test_parses_requested_metrics(self) -> None:
        sample = parse_tegrastats_line(self.SAMPLE_ONE)

        self.assertAlmostEqual(79.67, sample["cpu"], places=2)
        self.assertEqual(13.0, sample["gpu"])
        self.assertEqual(8.527, sample["power_watts"])
        self.assertEqual(49.031, sample["temperature_celsius"])
        self.assertEqual(3415, sample["ram_used_mib"])
        self.assertEqual(654, sample["swap_used_mib"])

    def test_summarizes_average_and_peak(self) -> None:
        samples = [
            parse_tegrastats_line(self.SAMPLE_ONE),
            parse_tegrastats_line(self.SAMPLE_TWO),
        ]

        summary = summarize_tegrastats(samples)

        self.assertEqual({"average": 30.0, "peak": 47.0}, summary["gpu_utilization_percent"])
        self.assertEqual({"average": 8.34, "peak": 8.53}, summary["power_watts"])
        self.assertEqual(79.67, summary["cpu_utilization_percent"]["peak"])
        self.assertEqual(49.34, summary["temperature_celsius"]["peak"])


if __name__ == "__main__":
    unittest.main()
