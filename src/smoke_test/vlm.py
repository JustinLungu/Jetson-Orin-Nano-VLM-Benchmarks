"""Vision-language model adapter for single-image CUDA smoke tests."""

from pathlib import Path
from typing import Any, Callable

from src.inference.vlm import (
    DEFAULT_VLM_MAX_NEW_TOKENS,
    DEFAULT_VLM_PROMPT,
    VlmInferenceSession,
    generated_output_tokens,
    prepare_vlm_inputs,
)
from src.load_models import load_vlm
from src.smoke_test.base import SmokeTestAdapter
from src.smoke_test.result import SmokeTestResult

SMOKE_PROMPT = DEFAULT_VLM_PROMPT
SMOKE_MAX_NEW_TOKENS = DEFAULT_VLM_MAX_NEW_TOKENS


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

    def create_session(self) -> VlmInferenceSession:
        return VlmInferenceSession(
            self.selector,
            device=self.device,
            precision=self.precision,
            torch_module=self.torch,
            model_loader=self.model_loader,
            prompt=SMOKE_PROMPT,
            max_new_tokens=SMOKE_MAX_NEW_TOKENS,
        )


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
