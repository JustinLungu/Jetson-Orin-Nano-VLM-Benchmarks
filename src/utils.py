"""Reusable utilities shared by model-management and benchmark code."""

import json
import struct
from collections import Counter
from pathlib import Path


def select_model_names(
    arguments: list[str],
    groups: dict[str, tuple[str, ...]],
    selectors: tuple[str, ...],
) -> list[str]:
    """Resolve individual model and family selectors in registry order."""
    if not arguments:
        raise ValueError("Select at least one model, 'yolo', 'small-vlm', or 'all'")
    if "all" in arguments and len(arguments) != 1:
        raise ValueError("Use 'all' alone, or select individual models or families")

    selected = []
    for argument in arguments:
        if argument in groups:
            selected.extend(groups[argument])
        elif argument in selectors:
            selected.append(argument)
        else:
            raise ValueError(f"Unknown model selector: {argument}")
    return list(dict.fromkeys(selected))


def inspect_safetensors(directory: Path) -> dict[str, int]:
    """Validate local safetensors files and count tensors by stored dtype."""
    dtype_counts: Counter[str] = Counter()
    weight_files = sorted(directory.glob("*.safetensors"))
    if not weight_files:
        raise RuntimeError(f"Downloaded snapshot has no safetensors weights: {directory}")

    for weight_file in weight_files:
        with weight_file.open("rb") as stream:
            header_size_bytes = stream.read(8)
            if len(header_size_bytes) != 8:
                raise RuntimeError(f"Invalid safetensors header: {weight_file}")
            header_size = struct.unpack("<Q", header_size_bytes)[0]
            try:
                header = json.loads(stream.read(header_size))
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise RuntimeError(f"Invalid safetensors metadata: {weight_file}") from error

        tensors = [value for key, value in header.items() if key != "__metadata__"]
        if not tensors:
            raise RuntimeError(f"Safetensors file contains no tensors: {weight_file}")
        expected_size = 8 + header_size + max(value["data_offsets"][1] for value in tensors)
        if weight_file.stat().st_size != expected_size:
            raise RuntimeError(f"Truncated safetensors file: {weight_file}")
        dtype_counts.update(value["dtype"] for value in tensors)

    return dict(sorted(dtype_counts.items()))
