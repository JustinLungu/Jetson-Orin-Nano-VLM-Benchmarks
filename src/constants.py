"""Shared paths and model groups used by the benchmark tooling."""

from pathlib import Path

MODEL_REPOSITORIES = {
    "smolvlm2-256m": "HuggingFaceTB/SmolVLM2-256M-Video-Instruct",
    "smolvlm2-500m": "HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
    "smolvlm2-2.2b": "HuggingFaceTB/SmolVLM2-2.2B-Instruct",
    "qwen2.5-vl-3b": "Qwen/Qwen2.5-VL-3B-Instruct",
    "phi-3.5-vision": "microsoft/Phi-3.5-vision-instruct",
}

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
