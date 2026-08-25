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

VLM_RUNTIME_PRECISIONS = {
    "smolvlm2-256m": ("fp32", "fp16"),
    "smolvlm2-500m": ("fp32", "fp16"),
    "smolvlm2-2.2b": ("fp16",),
    "qwen2.5-vl-3b": ("fp16",),
    "phi-3.5-vision": ("fp16",),
}

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIRECTORY = REPOSITORY_ROOT / "datasets"
COCO_DIRECTORY = DATASET_DIRECTORY / "coco"
IMAGENETTE_DIRECTORY = DATASET_DIRECTORY / "imagenette"
YOLO_MODEL_DIRECTORY = REPOSITORY_ROOT / "models" / "yolo"
SMALL_VLM_MODEL_DIRECTORY = REPOSITORY_ROOT / "models" / "small_vlm"

COCO_IMAGES_URL = "http://images.cocodataset.org/zips/val2017.zip"
COCO_ANNOTATIONS_URL = (
    "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
)
IMAGENETTE_URL = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz"
IMAGENETTE_CLASS_IDS = {
    "n01440764": 0,
    "n02102040": 217,
    "n02979186": 482,
    "n03000684": 491,
    "n03028079": 497,
    "n03394916": 566,
    "n03417042": 569,
    "n03425413": 571,
    "n03445777": 574,
    "n03888257": 701,
}

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
