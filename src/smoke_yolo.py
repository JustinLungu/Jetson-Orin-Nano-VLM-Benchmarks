"""YOLO model adapter for single-image CUDA smoke tests."""

import time
from pathlib import Path
from typing import Any, Callable

from PIL import Image

from src.constants import YOLO_MODEL_DIRECTORY, YOLO_MODELS
from src.inference_utils import (
    cleanup_cuda,
    collect_runtime_metadata,
    is_cuda_out_of_memory,
    measure_cuda_operation,
    peak_cuda_memory_mib,
    reset_peak_cuda_memory,
)
from src.smoke_results import SmokeTestResult

SMOKE_IMAGE_SIZE = 320


def run_yolo_smoke_test(
    selector: str,
    image_path: Path,
    *,
    device: str = "cuda:0",
    model_directory: Path = YOLO_MODEL_DIRECTORY,
    torch_module: Any = None,
    yolo_class: Any = None,
    clock: Callable[[], float] = time.perf_counter,
) -> SmokeTestResult:
    """Load a local YOLO checkpoint and run warm-up and measured inference."""
    if torch_module is None:
        import torch as torch_module
    if yolo_class is None:
        from ultralytics import YOLO as yolo_class

    runtime_versions = collect_runtime_metadata(torch_module)
    load_time_seconds = None
    inference_time_seconds = None
    peak_memory_mib = None
    phase = "model_loading"
    model = None
    image = None

    try:
        if selector not in YOLO_MODELS:
            raise ValueError(f"Unknown YOLO selector: {selector}")
        if not image_path.is_file():
            raise FileNotFoundError(f"Smoke-test image not found: {image_path}")

        checkpoint = model_directory / YOLO_MODELS[selector]
        if not checkpoint.is_file():
            raise FileNotFoundError(f"YOLO checkpoint not found: {checkpoint}")

        started_at = clock()
        model = yolo_class(str(checkpoint), task="detect")
        load_time_seconds = clock() - started_at
        with Image.open(image_path) as source_image:
            image = source_image.convert("RGB")

        reset_peak_cuda_memory(torch_module, device)
        phase = "warmup"
        prediction_arguments = {
            "source": image,
            "device": device,
            "imgsz": SMOKE_IMAGE_SIZE,
            "half": True,
            "verbose": False,
        }
        model.predict(**prediction_arguments)

        phase = "inference"
        predictions, inference_time_seconds = measure_cuda_operation(
            lambda: model.predict(**prediction_arguments),
            torch_module,
            clock=clock,
        )
        peak_memory_mib = peak_cuda_memory_mib(torch_module, device)

        return SmokeTestResult(
            model=selector,
            family="yolo",
            status="passed",
            device=device,
            runtime_versions=runtime_versions,
            load_time_seconds=load_time_seconds,
            inference_time_seconds=inference_time_seconds,
            peak_cuda_memory_mib=peak_memory_mib,
            prediction_summary=summarize_yolo_predictions(predictions),
        )
    except Exception as error:
        error_type = (
            "cuda_out_of_memory"
            if is_cuda_out_of_memory(error, torch_module)
            else f"{phase}_error"
        )
        return SmokeTestResult(
            model=selector,
            family="yolo",
            status="failed",
            device=device,
            runtime_versions=runtime_versions,
            load_time_seconds=load_time_seconds,
            inference_time_seconds=inference_time_seconds,
            peak_cuda_memory_mib=peak_memory_mib,
            error_type=error_type,
            error_message=str(error),
        )
    finally:
        model = None
        if image is not None:
            image.close()
        cleanup_cuda(torch_module)


def summarize_yolo_predictions(predictions: Any) -> str:
    """Return a compact summary without evaluating detection accuracy."""
    if not predictions:
        return "detections=0"
    boxes = getattr(predictions[0], "boxes", None)
    return f"detections={len(boxes) if boxes is not None else 0}"
