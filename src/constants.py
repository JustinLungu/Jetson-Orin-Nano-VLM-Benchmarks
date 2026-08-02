"""Shared paths and model groups used by the benchmark tooling."""

from pathlib import Path

from src.model_registry import MODEL_REPOSITORIES

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
YOLO_MODEL_DIRECTORY = REPOSITORY_ROOT / "models" / "yolo"
SMALL_VLM_MODEL_DIRECTORY = REPOSITORY_ROOT / "models" / "small_vlm"

YOLO_MODELS = {
    "yolov8n": "yolov8n.pt",
    "yolo11n": "yolo11n.pt",
    "yolo26n": "yolo26n.pt",
}
MODEL_SELECTORS = (*YOLO_MODELS, *MODEL_REPOSITORIES)
MODEL_GROUPS = {
    "yolo": tuple(YOLO_MODELS),
    "small-vlm": tuple(MODEL_REPOSITORIES),
    "all": MODEL_SELECTORS,
}
