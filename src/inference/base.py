"""Common contract for reusable loaded-model inference sessions."""

from abc import ABC, abstractmethod
from typing import Any

from PIL import Image


class InferenceSession(ABC):
    """Own one loaded model and process multiple images with it."""

    family: str
    precision: str

    def __init__(
        self,
        selector: str,
        *,
        device: str = "cuda:0",
        torch_module: Any = None,
    ) -> None:
        if torch_module is None:
            import torch as torch_module
        self.selector = selector
        self.device = device
        self.torch = torch_module
        self.model: Any = None
        self.processor: Any = None

    @abstractmethod
    def load(self) -> None:
        """Load local model resources once for this session."""

    @abstractmethod
    def prepare(self, image: Image.Image) -> Any:
        """Prepare one image without retaining it on the session."""

    @abstractmethod
    def infer(self, prepared: Any) -> Any:
        """Run inference for one prepared input."""

    @abstractmethod
    def summarize(self, output: Any, prepared: Any) -> tuple[str, int | None]:
        """Return a compact output summary and optional generated-token count."""

    def processed_image_size(self, prepared: Any) -> tuple[int, int] | None:
        """Return the prepared input width and height when they are discoverable."""
        return None

    def close(self) -> None:
        """Release references owned by the loaded session."""
        self.model = None
        self.processor = None
