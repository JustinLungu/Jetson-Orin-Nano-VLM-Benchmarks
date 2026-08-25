"""YOLO adapter for single-image CUDA smoke tests."""

from pathlib import Path
from typing import Any, Callable

from PIL import Image

from src.constants import YOLO_MODEL_DIRECTORY, YOLO_MODELS
from src.smoke_test.base import SmokeTestAdapter
from src.smoke_test.result import SmokeTestResult

SMOKE_IMAGE_SIZE = 320


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
        self.prediction_arguments: dict[str, Any] = {}

    def validate_selector(self) -> None:
        if self.selector not in YOLO_MODELS:
            raise ValueError(f"Unknown YOLO selector: {self.selector}")

    def load_model(self) -> tuple[Any, None]:
        checkpoint = self.model_directory / YOLO_MODELS[self.selector]
        if not checkpoint.is_file():
            raise FileNotFoundError(f"YOLO checkpoint not found: {checkpoint}")
        return self.yolo_class(str(checkpoint), task="detect"), None

    def prepare_inputs(self, image: Image.Image) -> dict[str, Any]:
        self.prediction_arguments = {
            "source": image,
            "device": self.device,
            "imgsz": SMOKE_IMAGE_SIZE,
            "quantize": 16,
            "verbose": False,
        }
        return self.prediction_arguments

    def infer(self) -> Any:
        return self.model.predict(**self.prediction_arguments)

    def summarize(self, output: Any) -> tuple[str, None]:
        return summarize_yolo_predictions(output), None


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


def summarize_yolo_predictions(predictions: Any) -> str:
    """Return a compact detection summary."""
    if not predictions:
        return "detections=0"
    boxes = getattr(predictions[0], "boxes", None)
    return f"detections={len(boxes) if boxes is not None else 0}"
