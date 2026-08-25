"""Offline tests for the YOLO smoke-test adapter."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from src.smoke_yolo import run_yolo_smoke_test, summarize_yolo_predictions


class FakeOutOfMemoryError(RuntimeError):
    pass


class FakeCuda:
    OutOfMemoryError = FakeOutOfMemoryError

    def __init__(self) -> None:
        self.synchronizations = 0
        self.reset_devices = []
        self.peak_devices = []
        self.empty_cache_calls = 0

    @staticmethod
    def is_available() -> bool:
        return True

    @staticmethod
    def get_device_name(index: int) -> str:
        return "Fake Orin"

    def synchronize(self) -> None:
        self.synchronizations += 1

    def reset_peak_memory_stats(self, device: str) -> None:
        self.reset_devices.append(device)

    def max_memory_allocated(self, device: str) -> int:
        self.peak_devices.append(device)
        return 256 * 1024**2

    def empty_cache(self) -> None:
        self.empty_cache_calls += 1


def make_fake_torch() -> SimpleNamespace:
    return SimpleNamespace(
        __version__="2.8.0",
        version=SimpleNamespace(cuda="12.6"),
        cuda=FakeCuda(),
    )


def write_test_image(path: Path) -> None:
    path.write_text("P3\n1 1\n255\n255 0 0\n", encoding="ascii")


class YoloSmokeTestAdapterTests(unittest.TestCase):
    def test_runs_warmup_and_measured_inference_from_local_checkpoint(self) -> None:
        torch = make_fake_torch()
        model_instance = Mock()
        model_instance.predict.return_value = [SimpleNamespace(boxes=[object(), object()])]
        yolo_class = Mock(return_value=model_instance)
        clock = Mock(side_effect=(1.0, 1.4, 2.0, 2.25))

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "yolo11n.pt"
            checkpoint.touch()
            image = root / "image.ppm"
            write_test_image(image)

            result = run_yolo_smoke_test(
                "yolo11n",
                image,
                model_directory=root,
                torch_module=torch,
                yolo_class=yolo_class,
                clock=clock,
            )

        yolo_class.assert_called_once_with(str(checkpoint), task="detect")
        self.assertEqual(2, model_instance.predict.call_count)
        prediction_arguments = model_instance.predict.call_args.kwargs
        self.assertEqual("RGB", prediction_arguments["source"].mode)
        self.assertEqual("cuda:0", prediction_arguments["device"])
        self.assertEqual(320, prediction_arguments["imgsz"])
        self.assertTrue(prediction_arguments["half"])
        self.assertFalse(prediction_arguments["verbose"])
        self.assertEqual("passed", result.status)
        self.assertAlmostEqual(0.4, result.load_time_seconds)
        self.assertEqual(0.25, result.inference_time_seconds)
        self.assertEqual(256.0, result.peak_cuda_memory_mib)
        self.assertEqual("detections=2", result.prediction_summary)
        self.assertEqual(2, torch.cuda.synchronizations)
        self.assertEqual(["cuda:0"], torch.cuda.reset_devices)
        self.assertEqual(1, torch.cuda.empty_cache_calls)

    def test_missing_checkpoint_fails_without_calling_yolo(self) -> None:
        torch = make_fake_torch()
        yolo_class = Mock()

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "image.ppm"
            write_test_image(image)
            result = run_yolo_smoke_test(
                "yolov8n",
                image,
                model_directory=root,
                torch_module=torch,
                yolo_class=yolo_class,
            )

        self.assertEqual("failed", result.status)
        self.assertEqual("model_loading_error", result.error_type)
        self.assertIn("checkpoint not found", result.error_message)
        yolo_class.assert_not_called()
        self.assertEqual(1, torch.cuda.empty_cache_calls)

    def test_cuda_out_of_memory_is_classified_and_cleaned_up(self) -> None:
        torch = make_fake_torch()
        model_instance = Mock()
        model_instance.predict.side_effect = FakeOutOfMemoryError("CUDA out of memory")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "yolo26n.pt").touch()
            image = root / "image.ppm"
            write_test_image(image)
            result = run_yolo_smoke_test(
                "yolo26n",
                image,
                model_directory=root,
                torch_module=torch,
                yolo_class=Mock(return_value=model_instance),
                clock=Mock(side_effect=(1.0, 1.1)),
            )

        self.assertEqual("failed", result.status)
        self.assertEqual("cuda_out_of_memory", result.error_type)
        self.assertEqual("CUDA out of memory", result.error_message)
        self.assertEqual(1, torch.cuda.empty_cache_calls)

    def test_cublas_allocation_failure_is_classified_as_out_of_memory(self) -> None:
        torch = make_fake_torch()
        model_instance = Mock()
        model_instance.predict.side_effect = RuntimeError("CUBLAS_STATUS_ALLOC_FAILED")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "yolov8n.pt").touch()
            image = root / "image.ppm"
            write_test_image(image)
            result = run_yolo_smoke_test(
                "yolov8n",
                image,
                model_directory=root,
                torch_module=torch,
                yolo_class=Mock(return_value=model_instance),
                clock=Mock(side_effect=(1.0, 1.1)),
            )

        self.assertEqual("cuda_out_of_memory", result.error_type)

    def test_prediction_summary_handles_empty_results(self) -> None:
        self.assertEqual("detections=0", summarize_yolo_predictions([]))
        self.assertEqual(
            "detections=0",
            summarize_yolo_predictions([SimpleNamespace(boxes=None)]),
        )


if __name__ == "__main__":
    unittest.main()
