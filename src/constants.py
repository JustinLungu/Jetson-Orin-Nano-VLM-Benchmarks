"""Shared paths and model groups used by the benchmark tooling."""

from pathlib import Path

MODEL_REPOSITORIES = {
    "smolvlm2-256m": "HuggingFaceTB/SmolVLM2-256M-Video-Instruct",
    "smolvlm2-500m": "HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
    "smolvlm2-2.2b": "HuggingFaceTB/SmolVLM2-2.2B-Instruct",
    "qwen2.5-vl-3b": "Qwen/Qwen2.5-VL-3B-Instruct",
    "phi-3.5-vision": "microsoft/Phi-3.5-vision-instruct",
}

# Keep only files needed by Transformers inference. In particular, SmolVLM
# repositories also contain several large ONNX variants that are not used by
# the PyTorch/Transformers benchmark path.
VLM_DOWNLOAD_ALLOW_PATTERNS = (
    "*.json",
    "*.safetensors",
    "*.py",
    "*.txt",
    "*.model",
    "*.tiktoken",
    "*.jinja",
)
VLM_DOWNLOAD_IGNORE_PATTERNS = (
    "onnx/*",
    "*.onnx",
    "*.bin",
    "*.gguf",
    "*.h5",
    "*.msgpack",
)

VLM_LOADER_CLASSES = {
    "smolvlm2-256m": ("AutoModelForImageTextToText", False),
    "smolvlm2-500m": ("AutoModelForImageTextToText", False),
    "smolvlm2-2.2b": ("AutoModelForImageTextToText", False),
    "qwen2.5-vl-3b": ("AutoModelForImageTextToText", False),
    "phi-3.5-vision": ("AutoModelForCausalLM", True),
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
