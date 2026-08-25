"""Offline tests for shared inference lifecycle utilities."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, call, patch

from src.smoke_test.runtime import (
    cleanup_cuda,
    collect_runtime_metadata,
    infer_jetpack_family,
    measure_cuda_operation,
    peak_cuda_memory_mib,
    reset_peak_cuda_memory,
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

    def test_peak_memory_is_reported_in_mib(self) -> None:
        cuda = Mock()
        cuda.max_memory_allocated.return_value = 512 * 1024**2
        torch = SimpleNamespace(cuda=cuda)

        reset_peak_cuda_memory(torch)
        peak = peak_cuda_memory_mib(torch)

        self.assertEqual(call("cuda:0"), cuda.reset_peak_memory_stats.call_args)
        self.assertEqual(call("cuda:0"), cuda.max_memory_allocated.call_args)
        self.assertEqual(512.0, peak)


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

    def test_cleanup_skips_cuda_cache_when_cuda_is_unavailable(self) -> None:
        cuda = Mock()
        cuda.is_available.return_value = False

        cleanup_cuda(SimpleNamespace(cuda=cuda), collector=Mock())

        cuda.empty_cache.assert_not_called()


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
            with patch("src.smoke_test.runtime.platform.machine", return_value="aarch64"), patch(
                "src.smoke_test.runtime.platform.python_version",
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

    def test_unknown_l4t_release_does_not_guess_jetpack(self) -> None:
        self.assertEqual("unknown", infer_jetpack_family("not a Jetson release"))


if __name__ == "__main__":
    unittest.main()
