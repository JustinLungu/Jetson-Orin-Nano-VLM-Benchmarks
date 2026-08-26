"""Prepare the persistent SmolVLM2-2.2B FP16 checkpoint."""

import argparse
import json
import os
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from src.constants import SMALL_VLM_MODEL_DIRECTORY
from src.utils import inspect_safetensors

FP16_DIRECTORY_NAME = "fp16"
CONVERSION_METADATA_NAME = "conversion_metadata.json"
MAX_OUTPUT_SHARD_BYTES = 256 * 1024**2


def prepared_fp16_path(model_directory: Path) -> Path | None:
    """Return a validated prepared checkpoint path when one exists."""
    candidate = model_directory / FP16_DIRECTORY_NAME
    metadata_path = candidate / CONVERSION_METADATA_NAME
    if not metadata_path.is_file():
        return None
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("target_dtype") != "float16":
        return None
    if set(inspect_safetensors(candidate)) != {"F16"}:
        raise RuntimeError(f"Prepared checkpoint is not entirely FP16: {candidate}")
    return candidate


def convert_checkpoint_to_fp16(source: Path, destination: Path) -> Path:
    """Convert safetensors shards to FP16 without modifying the source checkpoint."""
    if destination.exists():
        raise FileExistsError(f"Prepared checkpoint already exists: {destination}")
    index_path = source / "model.safetensors.index.json"
    if not (source / "config.json").is_file() or not index_path.is_file():
        raise RuntimeError(f"Expected a sharded Transformers checkpoint under {source}")

    index = json.loads(index_path.read_text(encoding="utf-8"))
    expected_names = set(index["weight_map"])
    source_shard_names = sorted(set(index["weight_map"].values()))
    temporary = destination.with_name(destination.name + ".tmp")
    if temporary.exists():
        raise FileExistsError(f"Incomplete conversion already exists: {temporary}")
    temporary.mkdir(parents=True)

    converted_names: set[str] = set()
    converted_weight_map: dict[str, str] = {}
    total_size = 0
    output_shard_index = 0
    try:
        for source_index, source_shard_name in enumerate(source_shard_names, start=1):
            print(
                f"[{source_index}/{len(source_shard_names)}] converting {source_shard_name}",
                flush=True,
            )
            tensors: dict[str, torch.Tensor] = {}
            buffered_size = 0
            with safe_open(
                source / source_shard_name,
                framework="pt",
                device="cpu",
            ) as checkpoint:
                metadata = checkpoint.metadata()
                for name in checkpoint.keys():
                    tensor = checkpoint.get_tensor(name)
                    converted = tensor.to(dtype=torch.float16)
                    tensors[name] = converted
                    converted_size = converted.numel() * converted.element_size()
                    converted_names.add(name)
                    total_size += converted_size
                    buffered_size += converted_size
                    if buffered_size >= MAX_OUTPUT_SHARD_BYTES:
                        output_shard_index += 1
                        output_name = f"model-fp16-{output_shard_index:05d}.safetensors"
                        save_file(tensors, temporary / output_name, metadata=metadata)
                        converted_weight_map.update(
                            {tensor_name: output_name for tensor_name in tensors}
                        )
                        tensors = {}
                        buffered_size = 0
                if tensors:
                    output_shard_index += 1
                    output_name = f"model-fp16-{output_shard_index:05d}.safetensors"
                    save_file(tensors, temporary / output_name, metadata=metadata)
                    converted_weight_map.update(
                        {tensor_name: output_name for tensor_name in tensors}
                    )

        if converted_names != expected_names:
            missing = sorted(expected_names - converted_names)
            unexpected = sorted(converted_names - expected_names)
            raise RuntimeError(
                f"Converted tensor names differ from the index; missing={missing}, "
                f"unexpected={unexpected}"
            )

        for path in source.iterdir():
            if path.is_file() and path.suffix != ".safetensors" and path.name != index_path.name:
                shutil.copy2(path, temporary / path.name)

        index["metadata"]["total_size"] = total_size
        index["weight_map"] = converted_weight_map
        (temporary / index_path.name).write_text(
            json.dumps(index, indent=2) + "\n",
            encoding="utf-8",
        )
        conversion_metadata = {
            "source": str(source),
            "source_dtypes": inspect_safetensors(source),
            "target_dtype": "float16",
            "tensor_count": len(converted_names),
            "total_tensor_bytes": total_size,
        }
        (temporary / CONVERSION_METADATA_NAME).write_text(
            json.dumps(conversion_metadata, indent=2) + "\n",
            encoding="utf-8",
        )
        if set(inspect_safetensors(temporary)) != {"F16"}:
            raise RuntimeError("Converted checkpoint contains non-FP16 tensors")
        os.replace(temporary, destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise

    print(f"FP16 checkpoint ready: {destination} ({total_size / 1024**3:.2f} GiB tensors)")
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a persistent FP16 VLM checkpoint.")
    parser.add_argument("model", choices=("smolvlm2-2.2b",))
    arguments = parser.parse_args()
    source = SMALL_VLM_MODEL_DIRECTORY / arguments.model
    convert_checkpoint_to_fp16(source, source / FP16_DIRECTORY_NAME)


if __name__ == "__main__":
    main()
