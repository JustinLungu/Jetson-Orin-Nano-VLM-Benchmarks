"""Base lifecycle for model smoke-test adapters."""

import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable

from PIL import Image

from src.smoke_test.result import SmokeTestResult
from src.smoke_test.runtime import (
    cleanup_cuda,
    collect_runtime_metadata,
    is_cuda_out_of_memory,
    measure_cuda_operation,
    peak_cuda_memory_mib,
    reset_peak_cuda_memory,
)


class SmokeTestAdapter(ABC):
    """Run the common load, warm-up, inference, reporting, and cleanup lifecycle."""

    family: str

    def __init__(
        self,
        selector: str,
        image_path: Path,
        *,
        device: str = "cuda:0",
        torch_module: Any = None,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        if torch_module is None:
            import torch as torch_module
        self.selector = selector
        self.image_path = image_path
        self.device = device
        self.torch = torch_module
        self.clock = clock
        self.model: Any = None
        self.processor: Any = None
        self.inputs: Any = None
        self.image: Image.Image | None = None

    def run(self) -> SmokeTestResult:
        """Execute one smoke test and convert all expected failures to a result."""
        runtime_versions = collect_runtime_metadata(self.torch)
        load_time = inference_time = peak_memory = None
        phase = "model_loading"

        try:
            self.validate_selector()
            if not self.image_path.is_file():
                raise FileNotFoundError(f"Smoke-test image not found: {self.image_path}")

            started_at = self.clock()
            self.model, self.processor = self.load_model()
            load_time = self.clock() - started_at

            with Image.open(self.image_path) as source_image:
                self.image = source_image.convert("RGB")
            self.inputs = self.prepare_inputs(self.image)

            reset_peak_cuda_memory(self.torch, self.device)
            phase = "warmup"
            warmup_output = self.infer()
            del warmup_output

            phase = "inference"
            output, inference_time = measure_cuda_operation(
                self.infer,
                self.torch,
                clock=self.clock,
            )
            peak_memory = peak_cuda_memory_mib(self.torch, self.device)
            summary, generated_tokens = self.summarize(output)
            return SmokeTestResult(
                model=self.selector,
                family=self.family,
                status="passed",
                device=self.device,
                runtime_versions=runtime_versions,
                load_time_seconds=load_time,
                inference_time_seconds=inference_time,
                peak_cuda_memory_mib=peak_memory,
                generated_tokens=generated_tokens,
                prediction_summary=summary,
            )
        except Exception as error:
            error_type = (
                "cuda_out_of_memory"
                if is_cuda_out_of_memory(error, self.torch)
                else f"{phase}_error"
            )
            return SmokeTestResult(
                model=self.selector,
                family=self.family,
                status="failed",
                device=self.device,
                runtime_versions=runtime_versions,
                load_time_seconds=load_time,
                inference_time_seconds=inference_time,
                peak_cuda_memory_mib=peak_memory,
                error_type=error_type,
                error_message=str(error),
            )
        finally:
            self.release()

    def release(self) -> None:
        """Release adapter-owned objects before returning cached CUDA memory."""
        self.model = self.processor = self.inputs = None
        if self.image is not None:
            self.image.close()
            self.image = None
        cleanup_cuda(self.torch)

    @abstractmethod
    def validate_selector(self) -> None:
        """Reject selectors not supported by this adapter."""

    @abstractmethod
    def load_model(self) -> tuple[Any, Any]:
        """Load local model resources and return model and optional processor."""

    @abstractmethod
    def prepare_inputs(self, image: Image.Image) -> Any:
        """Prepare model-specific inputs from a decoded RGB image."""

    @abstractmethod
    def infer(self) -> Any:
        """Run one inference using prepared adapter state."""

    @abstractmethod
    def summarize(self, output: Any) -> tuple[str, int | None]:
        """Return a compact prediction summary and optional generated-token count."""
