"""Vision-language model adapter for single-image CUDA smoke tests."""

from pathlib import Path
from typing import Any, Callable

from PIL import Image

from src.constants import MODEL_REPOSITORIES
from src.load_models import load_vlm
from src.smoke_test.base import SmokeTestAdapter
from src.smoke_test.result import SmokeTestResult

SMOKE_PROMPT = "Describe this image briefly."
SMOKE_MAX_NEW_TOKENS = 16


class VlmSmokeTestAdapter(SmokeTestAdapter):
    """Run local VLM checkpoints through the shared smoke-test lifecycle."""

    family = "small-vlm"

    def __init__(
        self,
        selector: str,
        image_path: Path,
        *,
        precision: str = "fp16",
        model_loader: Callable[..., tuple[Any, Any]] = load_vlm,
        **kwargs: Any,
    ) -> None:
        super().__init__(selector, image_path, **kwargs)
        self.model_loader = model_loader
        self.precision = precision
        self.generation_arguments: dict[str, Any] = {}

    def validate_selector(self) -> None:
        if self.selector not in MODEL_REPOSITORIES:
            raise ValueError(f"Unknown VLM selector: {self.selector}")

    def load_model(self) -> tuple[Any, Any]:
        return self.model_loader(
            self.selector,
            device=self.device,
            precision=self.precision,
            torch_module=self.torch,
        )

    def prepare_inputs(self, image: Image.Image) -> Any:
        inputs = prepare_vlm_inputs(
            self.selector,
            self.processor,
            image,
            self.device,
        )
        self.generation_arguments = {
            **inputs,
            "max_new_tokens": SMOKE_MAX_NEW_TOKENS,
            "do_sample": False,
        }
        return inputs

    def infer(self) -> Any:
        with self.torch.inference_mode():
            return self.model.generate(**self.generation_arguments)

    def summarize(self, output: Any) -> tuple[str, int]:
        generated = generated_output_tokens(output, self.inputs)
        generated_tokens = generated.shape[-1]
        summary = self.processor.batch_decode(
            generated,
            skip_special_tokens=True,
        )[0].strip()
        return summary, generated_tokens


def run_vlm_smoke_test(
    selector: str,
    image_path: Path,
    *,
    device: str = "cuda:0",
    precision: str = "fp16",
    torch_module: Any = None,
    model_loader: Callable[..., tuple[Any, Any]] = load_vlm,
    clock: Callable[[], float] | None = None,
) -> SmokeTestResult:
    """Compatibility wrapper around :class:`VlmSmokeTestAdapter`."""
    arguments: dict[str, Any] = {
        "device": device,
        "precision": precision,
        "torch_module": torch_module,
        "model_loader": model_loader,
    }
    if clock is not None:
        arguments["clock"] = clock
    return VlmSmokeTestAdapter(selector, image_path, **arguments).run()


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
