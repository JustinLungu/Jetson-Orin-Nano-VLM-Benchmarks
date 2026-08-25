"""Download selected YOLO and small-VLM checkpoints."""

import argparse
import json
import os
from pathlib import Path
from typing import Any

from src.constants import (
    MODEL_GROUPS,
    MODEL_REPOSITORIES,
    MODEL_SELECTORS,
    REPOSITORY_ROOT,
    SMALL_VLM_MODEL_DIRECTORY,
    VLM_DOWNLOAD_ALLOW_PATTERNS,
    VLM_DOWNLOAD_IGNORE_PATTERNS,
    VLM_LOADER_CLASSES,
    YOLO_MODEL_DIRECTORY,
    YOLO_MODELS,
)
from src.utils import inspect_safetensors, select_model_names


def select_models(arguments: list[str]) -> list[str]:
    """Resolve selectors using the configured model registry."""
    return select_model_names(arguments, MODEL_GROUPS, MODEL_SELECTORS)


def download_yolo_model(selector: str, loader=None) -> Path:
    """Download and load one YOLO checkpoint."""
    if loader is None:
        from ultralytics import YOLO

        loader = YOLO

    YOLO_MODEL_DIRECTORY.mkdir(parents=True, exist_ok=True)
    model_name = YOLO_MODELS[selector]
    model_path = YOLO_MODEL_DIRECTORY / model_name
    previous_directory = Path.cwd()
    try:
        os.chdir(YOLO_MODEL_DIRECTORY)
        print(f"Downloading {selector}...")
        loader(model_name, task="detect")
    finally:
        os.chdir(previous_directory)

    if not model_path.is_file():
        raise RuntimeError(f"YOLO checkpoint was not downloaded to {model_path}")
    print(f"  ready: {model_path} ({model_path.stat().st_size / 1_000_000:.1f} MB)")
    return model_path


def download_small_vlm_model(selector: str, api, downloader) -> Path:
    """Download one immutable Hugging Face snapshot and record its identity."""
    repository = MODEL_REPOSITORIES[selector]
    revision = api.model_info(repository).sha
    destination = SMALL_VLM_MODEL_DIRECTORY / selector
    destination.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {selector}...")
    print(f"  repository: {repository}")
    print(f"  revision:   {revision}")
    downloader(
        repo_id=repository,
        revision=revision,
        local_dir=destination,
        allow_patterns=VLM_DOWNLOAD_ALLOW_PATTERNS,
        ignore_patterns=VLM_DOWNLOAD_IGNORE_PATTERNS,
    )

    if not (destination / "config.json").is_file():
        raise RuntimeError(f"Downloaded snapshot has no config.json: {destination}")
    checkpoint_dtypes = inspect_safetensors(destination)

    metadata = {
        "selector": selector,
        "repository": repository,
        "revision": revision,
        "checkpoint_dtypes": checkpoint_dtypes,
        "jetson_runtime_dtype": "float16",
    }
    (destination / "download_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    size_gib = sum(path.stat().st_size for path in destination.rglob("*") if path.is_file())
    print(f"  ready: {destination} ({size_gib / 1024**3:.2f} GiB)")
    print(f"  checkpoint dtypes: {checkpoint_dtypes}")
    print("  Jetson runtime dtype: float16")
    return destination


def download_models(selectors: list[str]) -> None:
    """Download the selected model families sequentially."""
    vlm_api = None
    vlm_downloader = None
    for selector in selectors:
        if selector in YOLO_MODELS:
            download_yolo_model(selector)
            continue

        if vlm_api is None:
            from huggingface_hub import HfApi, snapshot_download

            vlm_api = HfApi()
            vlm_downloader = snapshot_download
        download_small_vlm_model(selector, vlm_api, vlm_downloader)


def load_vlm_fp16(
    selector: str,
    device: str = "cuda",
    *,
    torch_module: Any = None,
    transformers_module: Any = None,
) -> tuple[Any, Any]:
    """Load a configured VLM in FP16 and fail if parameters use another dtype."""
    if selector not in VLM_LOADER_CLASSES:
        raise ValueError(f"Unknown VLM selector: {selector}")

    if torch_module is None:
        import torch as torch_module
    if transformers_module is None:
        import transformers as transformers_module

    model_path = SMALL_VLM_MODEL_DIRECTORY / selector
    if not (model_path / "config.json").is_file():
        raise FileNotFoundError(f"Model is not downloaded: {model_path}")

    from src.model_preparation.fp16 import prepared_fp16_path

    model_path = prepared_fp16_path(model_path) or model_path

    loader_name, trust_remote_code = VLM_LOADER_CLASSES[selector]
    loader = getattr(transformers_module, loader_name)
    common_arguments = {
        "local_files_only": True,
        "trust_remote_code": trust_remote_code,
    }
    processor = transformers_module.AutoProcessor.from_pretrained(
        model_path,
        **common_arguments,
    )
    model = loader.from_pretrained(
        model_path,
        dtype=torch_module.float16,
        low_cpu_mem_usage=True,
        **common_arguments,
    )
    model = model.to(device).eval()

    parameter_dtypes = {parameter.dtype for parameter in model.parameters()}
    if not parameter_dtypes or not parameter_dtypes.issubset({torch_module.float16}):
        raise RuntimeError(
            f"Expected every floating-point model parameter to be FP16, got {parameter_dtypes}"
        )
    return model, processor


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download one, several, or all configured YOLO and small-VLM models.",
    )
    parser.add_argument(
        "models",
        nargs="+",
        metavar="MODEL",
        help="model selector(s), 'yolo', 'small-vlm', or 'all'",
    )
    arguments = parser.parse_args()
    try:
        selected = select_models(arguments.models)
    except ValueError as error:
        parser.error(str(error))

    download_models(selected)
    print(f"\nDownloaded {len(selected)} model(s) under {REPOSITORY_ROOT / 'models'}")


if __name__ == "__main__":
    main()
