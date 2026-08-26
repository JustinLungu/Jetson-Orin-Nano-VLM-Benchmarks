"""Reusable YOLO inference session."""

from pathlib import Path
from typing import Any

from PIL import Image

from src.constants import YOLO_MODEL_DIRECTORY, YOLO_MODELS
from src.inference.base import InferenceSession

DEFAULT_YOLO_IMAGE_SIZE = 320


class YoloInferenceSession(InferenceSession):
    """Load one YOLO checkpoint once and predict multiple images."""

    family = "yolo"
    precision = "fp16"

    def __init__(
        self,
        selector: str,
        *,
        model_directory: Path = YOLO_MODEL_DIRECTORY,
        yolo_class: Any = None,
        image_size: int = DEFAULT_YOLO_IMAGE_SIZE,
        **kwargs: Any,
    ) -> None:
        super().__init__(selector, **kwargs)
        if yolo_class is None:
            from ultralytics import YOLO as yolo_class
        self.model_directory = model_directory
        self.yolo_class = yolo_class
        if image_size <= 0 or image_size % 32 != 0:
            raise ValueError("YOLO image_size must be a positive multiple of 32")
        self.image_size = image_size

    def load(self) -> None:
        if self.selector not in YOLO_MODELS:
            raise ValueError(f"Unknown YOLO selector: {self.selector}")
        checkpoint = self.model_directory / YOLO_MODELS[self.selector]
        if not checkpoint.is_file():
            raise FileNotFoundError(f"YOLO checkpoint not found: {checkpoint}")
        self.model = self.yolo_class(str(checkpoint), task="detect")

    def prepare(self, image: Image.Image) -> dict[str, Any]:
        return {
            "source": image,
            "device": self.device,
            "imgsz": self.image_size,
            "rect": False,
            "quantize": 16,
            "verbose": False,
        }

    def infer(self, prepared: dict[str, Any]) -> Any:
        if self.model is None:
            raise RuntimeError("YOLO session must be loaded before inference")
        return self.model.predict(**prepared)

    def processed_image_size(self, prepared: dict[str, Any]) -> tuple[int, int]:
        """Return the fixed square tensor shape requested from Ultralytics."""
        image_size = int(prepared["imgsz"])
        return image_size, image_size

    def summarize(
        self,
        output: Any,
        prepared: dict[str, Any],
    ) -> tuple[str, None]:
        return summarize_yolo_predictions(output), None


def summarize_yolo_predictions(predictions: Any) -> str:
    """Return a compact detection summary."""
    if not predictions:
        return "detections=0"
    boxes = getattr(predictions[0], "boxes", None)
    return f"detections={len(boxes) if boxes is not None else 0}"
