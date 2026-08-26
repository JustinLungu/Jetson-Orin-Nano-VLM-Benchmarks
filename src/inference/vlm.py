"""Reusable vision-language model inference session."""

from dataclasses import dataclass
from typing import Any, Callable

from PIL import Image

from src.constants import MODEL_REPOSITORIES
from src.inference.base import InferenceSession
from src.load_models import load_vlm

DEFAULT_VLM_PROMPT = "Describe this image briefly."
DEFAULT_VLM_MAX_NEW_TOKENS = 16


@dataclass(frozen=True, slots=True)
class PreparedVlmInput:
    """Processor output and deterministic generation arguments for one image."""

    inputs: Any
    generation_arguments: dict[str, Any]


class VlmInferenceSession(InferenceSession):
    """Load one VLM once and generate outputs for multiple images."""

    family = "small-vlm"

    def __init__(
        self,
        selector: str,
        *,
        precision: str = "fp16",
        model_loader: Callable[..., tuple[Any, Any]] = load_vlm,
        prompt: str = DEFAULT_VLM_PROMPT,
        max_new_tokens: int = DEFAULT_VLM_MAX_NEW_TOKENS,
        **kwargs: Any,
    ) -> None:
        super().__init__(selector, **kwargs)
        self.precision = precision
        self.model_loader = model_loader
        self.prompt = prompt
        self.max_new_tokens = max_new_tokens

    def load(self) -> None:
        if self.selector not in MODEL_REPOSITORIES:
            raise ValueError(f"Unknown VLM selector: {self.selector}")
        self.model, self.processor = self.model_loader(
            self.selector,
            device=self.device,
            precision=self.precision,
            torch_module=self.torch,
        )

    def prepare(self, image: Image.Image) -> PreparedVlmInput:
        if self.processor is None:
            raise RuntimeError("VLM session must be loaded before preparing inputs")
        inputs = prepare_vlm_inputs(
            self.selector,
            self.processor,
            image,
            self.device,
            prompt=self.prompt,
        )
        return PreparedVlmInput(
            inputs=inputs,
            generation_arguments={
                **inputs,
                "max_new_tokens": self.max_new_tokens,
                "do_sample": False,
            },
        )

    def infer(self, prepared: PreparedVlmInput) -> Any:
        if self.model is None:
            raise RuntimeError("VLM session must be loaded before inference")
        with self.torch.inference_mode():
            return self.model.generate(**prepared.generation_arguments)

    def processed_image_size(
        self,
        prepared: PreparedVlmInput,
    ) -> tuple[int, int] | None:
        """Read the model-native processed image shape from processor tensors."""
        inputs = prepared.inputs
        pixel_values = inputs.get("pixel_values") if hasattr(inputs, "get") else None
        shape = getattr(pixel_values, "shape", None)
        if shape is None or len(shape) < 2:
            return None
        height, width = int(shape[-2]), int(shape[-1])
        return width, height

    def summarize(
        self,
        output: Any,
        prepared: PreparedVlmInput,
    ) -> tuple[str, int]:
        generated = generated_output_tokens(output, prepared.inputs)
        summary = self.processor.batch_decode(
            generated,
            skip_special_tokens=True,
        )[0].strip()
        return summary, generated.shape[-1]


def prepare_vlm_inputs(
    selector: str,
    processor: Any,
    image: Image.Image,
    device: str,
    *,
    prompt: str = DEFAULT_VLM_PROMPT,
) -> Any:
    """Prepare model-specific image and prompt inputs behind a common interface."""
    if selector == "phi-3.5-vision":
        formatted_prompt = (
            f"<|user|>\n<|image_1|>\n{prompt}<|end|>\n<|assistant|>\n"
        )
        inputs = processor(formatted_prompt, [image], return_tensors="pt")
    else:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        formatted_prompt = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
            processor_kwargs={"num_frames": 1},
        )
        processor_arguments = {
            "text": formatted_prompt,
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
