"""Serializable result contract for model smoke tests."""

from dataclasses import asdict, dataclass
from typing import Any, Literal

SmokeStatus = Literal["passed", "failed"]


@dataclass(frozen=True, slots=True)
class SmokeTestResult:
    """Result of loading a model and attempting one measured inference."""

    model: str
    family: str
    status: SmokeStatus
    device: str
    runtime_versions: dict[str, str]
    load_time_seconds: float | None = None
    inference_time_seconds: float | None = None
    peak_cuda_memory_mib: float | None = None
    jetson_metrics: dict[str, dict[str, float]] | None = None
    generated_tokens: int | None = None
    prediction_summary: str | None = None
    error_type: str | None = None
    error_message: str | None = None

    def __post_init__(self) -> None:
        if self.status == "passed" and (self.error_type or self.error_message):
            raise ValueError("A passed smoke test cannot contain an error")
        if self.status == "failed" and not self.error_type:
            raise ValueError("A failed smoke test must include an error type")
        for field_name in (
            "load_time_seconds",
            "inference_time_seconds",
            "peak_cuda_memory_mib",
        ):
            value = getattr(self, field_name)
            if value is not None and value < 0:
                raise ValueError(f"{field_name} cannot be negative")
        if self.generated_tokens is not None and self.generated_tokens < 0:
            raise ValueError("generated_tokens cannot be negative")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dictionary with a stable schema."""
        return asdict(self)
