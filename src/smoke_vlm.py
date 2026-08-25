"""Vision-language model adapter for single-image CUDA smoke tests."""

import time
from pathlib import Path
from typing import Any, Callable

from PIL import Image

from src.constants import MODEL_REPOSITORIES
from src.inference_utils import (
    cleanup_cuda,
    collect_runtime_metadata,
    is_cuda_out_of_memory,
    measure_cuda_operation,
    peak_cuda_memory_mib,
    reset_peak_cuda_memory,
)
from src.load_models import load_vlm_fp16
from src.smoke_results import SmokeTestResult

SMOKE_PROMPT = "Describe this image briefly."
SMOKE_MAX_NEW_TOKENS = 16


def run_vlm_smoke_test(
    selector: str,
    image_path: Path,
    *,
    device: str = "cuda:0",
    torch_module: Any = None,
    model_loader: Callable[..., tuple[Any, Any]] = load_vlm_fp16,
    clock: Callable[[], float] = time.perf_counter,
) -> SmokeTestResult:
    """Load a local VLM and run warm-up and measured deterministic generation."""
    if torch_module is None:
        import torch as torch_module

    runtime_versions = collect_runtime_metadata(torch_module)
    load_time_seconds = None
    inference_time_seconds = None
    peak_memory_mib = None
    generated_tokens = None
    phase = "model_loading"
    model = processor = inputs = image = None

    try:
        if selector not in MODEL_REPOSITORIES:
            raise ValueError(f"Unknown VLM selector: {selector}")
        if not image_path.is_file():
            raise FileNotFoundError(f"Smoke-test image not found: {image_path}")

        started_at = clock()
        model, processor = model_loader(
            selector,
            device=device,
            torch_module=torch_module,
        )
        load_time_seconds = clock() - started_at

        with Image.open(image_path) as source_image:
            image = source_image.convert("RGB")
        inputs = prepare_vlm_inputs(selector, processor, image, device)
        generation_arguments = {
            **inputs,
            "max_new_tokens": SMOKE_MAX_NEW_TOKENS,
            "do_sample": False,
        }

        reset_peak_cuda_memory(torch_module, device)
        phase = "warmup"
        with torch_module.inference_mode():
            warmup_output = model.generate(**generation_arguments)
        del warmup_output

        phase = "inference"
        with torch_module.inference_mode():
            output, inference_time_seconds = measure_cuda_operation(
                lambda: model.generate(**generation_arguments),
                torch_module,
                clock=clock,
            )
        peak_memory_mib = peak_cuda_memory_mib(torch_module, device)
        generated = generated_output_tokens(output, inputs)
        generated_tokens = generated.shape[-1]
        summary = processor.batch_decode(generated, skip_special_tokens=True)[0].strip()

        return SmokeTestResult(
            model=selector,
            family="small-vlm",
            status="passed",
            device=device,
            runtime_versions=runtime_versions,
            load_time_seconds=load_time_seconds,
            inference_time_seconds=inference_time_seconds,
            peak_cuda_memory_mib=peak_memory_mib,
            generated_tokens=generated_tokens,
            prediction_summary=summary,
        )
    except Exception as error:
        error_type = (
            "cuda_out_of_memory"
            if is_cuda_out_of_memory(error, torch_module)
            else f"{phase}_error"
        )
        return SmokeTestResult(
            model=selector,
            family="small-vlm",
            status="failed",
            device=device,
            runtime_versions=runtime_versions,
            load_time_seconds=load_time_seconds,
            inference_time_seconds=inference_time_seconds,
            peak_cuda_memory_mib=peak_memory_mib,
            generated_tokens=generated_tokens,
            error_type=error_type,
            error_message=str(error),
        )
    finally:
        model = processor = inputs = None
        if image is not None:
            image.close()
        cleanup_cuda(torch_module)


def prepare_vlm_inputs(
    selector: str,
    processor: Any,
    image: Image.Image,
    device: str,
) -> Any:
    """Prepare model-specific image and prompt inputs behind a common interface."""
    if selector == "phi-3.5-vision":
        prompt = f"<|user|>\n<|image_1|>\n{SMOKE_PROMPT}<|end|>\n<|assistant|>\n"
        inputs = processor(prompt, [image], return_tensors="pt")
    else:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": SMOKE_PROMPT},
                ],
            }
        ]
        prompt = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
            processor_kwargs={"num_frames": 1},
        )
        processor_arguments = {
            "text": prompt,
            "images": [image],
            "return_tensors": "pt",
        }
        if selector.startswith("smolvlm2-"):
            processor_arguments["images_kwargs"] = {"do_image_splitting": False}
        inputs = processor(**processor_arguments)
    return inputs.to(device)


def generated_output_tokens(output: Any, inputs: Any) -> Any:
    """Remove decoder-only prompt tokens from generated sequences."""
    input_length = inputs["input_ids"].shape[-1]
    return output[:, input_length:]
