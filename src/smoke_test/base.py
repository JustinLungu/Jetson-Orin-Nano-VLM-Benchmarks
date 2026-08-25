"""Base lifecycle for model smoke-test adapters."""

import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable

from PIL import Image

from src.inference.base import InferenceSession
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
        self.session: InferenceSession | None = None
        self.prepared: Any = None
        self.image: Image.Image | None = None

    def run(self) -> SmokeTestResult:
        """Execute one smoke test and convert all expected failures to a result."""
        runtime_versions = collect_runtime_metadata(self.torch)
        load_time = inference_time = peak_memory = None
        phase = "model_loading"

        try:
            if not self.image_path.is_file():
                raise FileNotFoundError(f"Smoke-test image not found: {self.image_path}")

            self.session = self.create_session()
            started_at = self.clock()
            self.session.load()
            load_time = self.clock() - started_at

            with Image.open(self.image_path) as source_image:
                self.image = source_image.convert("RGB")
            self.prepared = self.session.prepare(self.image)

            reset_peak_cuda_memory(self.torch, self.device)
            phase = "warmup"
            warmup_output = self.session.infer(self.prepared)
            del warmup_output

            phase = "inference"
            output, inference_time = measure_cuda_operation(
                lambda: self.session.infer(self.prepared),
                self.torch,
                clock=self.clock,
            )
            peak_memory = peak_cuda_memory_mib(self.torch, self.device)
            summary, generated_tokens = self.session.summarize(output, self.prepared)
            return SmokeTestResult(
                model=self.selector,
                family=self.family,
                status="passed",
                device=self.device,
                runtime_versions=runtime_versions,
                runtime_precision=(
                    self.session.precision
                    if self.session is not None
                    else getattr(self, "precision", None)
                ),
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
                runtime_precision=(
                    self.session.precision
                    if self.session is not None
                    else getattr(self, "precision", None)
                ),
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
        self.prepared = None
        if self.session is not None:
            self.session.close()
            self.session = None
        if self.image is not None:
            self.image.close()
            self.image = None
        cleanup_cuda(self.torch)

    @abstractmethod
    def create_session(self) -> InferenceSession:
        """Create the family-specific loaded-model session."""
