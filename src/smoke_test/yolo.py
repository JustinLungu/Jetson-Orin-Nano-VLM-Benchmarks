"""YOLO adapter for single-image CUDA smoke tests."""

from pathlib import Path
from typing import Any, Callable

from src.constants import YOLO_MODEL_DIRECTORY
from src.inference.yolo import (
    DEFAULT_YOLO_IMAGE_SIZE,
    YoloInferenceSession,
    summarize_yolo_predictions,
)
from src.smoke_test.base import SmokeTestAdapter
from src.smoke_test.result import SmokeTestResult

SMOKE_IMAGE_SIZE = DEFAULT_YOLO_IMAGE_SIZE


class YoloSmokeTestAdapter(SmokeTestAdapter):
    """Run local Ultralytics checkpoints through the shared smoke-test lifecycle."""

    family = "yolo"

    def __init__(
        self,
        selector: str,
        image_path: Path,
        *,
        model_directory: Path = YOLO_MODEL_DIRECTORY,
        yolo_class: Any = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(selector, image_path, **kwargs)
        self.precision = "fp16"
        if yolo_class is None:
            from ultralytics import YOLO as yolo_class
        self.model_directory = model_directory
        self.yolo_class = yolo_class

    def create_session(self) -> YoloInferenceSession:
        return YoloInferenceSession(
            self.selector,
            device=self.device,
            torch_module=self.torch,
            model_directory=self.model_directory,
            yolo_class=self.yolo_class,
            image_size=SMOKE_IMAGE_SIZE,
        )


def run_yolo_smoke_test(
    selector: str,
    image_path: Path,
    *,
    device: str = "cuda:0",
    model_directory: Path = YOLO_MODEL_DIRECTORY,
    torch_module: Any = None,
    yolo_class: Any = None,
    clock: Callable[[], float] | None = None,
) -> SmokeTestResult:
    """Compatibility wrapper around :class:`YoloSmokeTestAdapter`."""
    arguments: dict[str, Any] = {
        "device": device,
        "model_directory": model_directory,
        "torch_module": torch_module,
        "yolo_class": yolo_class,
    }
    if clock is not None:
        arguments["clock"] = clock
    return YoloSmokeTestAdapter(selector, image_path, **arguments).run()
