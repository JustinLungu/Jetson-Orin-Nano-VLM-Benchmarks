"""Download selected YOLO and small-VLM checkpoints."""

import argparse
import json
import os
from pathlib import Path

from src.constants import (
    MODEL_GROUPS,
    MODEL_REPOSITORIES,
    MODEL_SELECTORS,
    REPOSITORY_ROOT,
    SMALL_VLM_MODEL_DIRECTORY,
    YOLO_MODEL_DIRECTORY,
    YOLO_MODELS,
)


def select_models(arguments: list[str]) -> list[str]:
    """Resolve individual model and family selectors in registry order."""
    if not arguments:
        raise ValueError("Select at least one model, 'yolo', 'small-vlm', or 'all'")
    if "all" in arguments and len(arguments) != 1:
        raise ValueError("Use 'all' alone, or select individual models or families")

    selected = []
    for argument in arguments:
        if argument in MODEL_GROUPS:
            selected.extend(MODEL_GROUPS[argument])
        elif argument in MODEL_SELECTORS:
            selected.append(argument)
        else:
            raise ValueError(f"Unknown model selector: {argument}")
    return list(dict.fromkeys(selected))


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
    downloader(repo_id=repository, revision=revision, local_dir=destination)

    if not (destination / "config.json").is_file():
        raise RuntimeError(f"Downloaded snapshot has no config.json: {destination}")

    metadata = {
        "selector": selector,
        "repository": repository,
        "revision": revision,
    }
    (destination / "download_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    size_gib = sum(path.stat().st_size for path in destination.rglob("*") if path.is_file())
    print(f"  ready: {destination} ({size_gib / 1024**3:.2f} GiB)")
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
