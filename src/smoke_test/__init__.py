"""Single-image inference smoke tests for configured benchmark models."""

from src.smoke_test.result import SmokeTestResult
from src.smoke_test.vlm import VlmSmokeTestAdapter, run_vlm_smoke_test
from src.smoke_test.yolo import YoloSmokeTestAdapter, run_yolo_smoke_test

__all__ = [
    "SmokeTestResult",
    "VlmSmokeTestAdapter",
    "YoloSmokeTestAdapter",
    "run_vlm_smoke_test",
    "run_yolo_smoke_test",
]
