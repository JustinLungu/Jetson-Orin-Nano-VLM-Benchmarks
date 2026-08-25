"""Deterministic, label-free benchmark dataset iterators."""

import csv
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from PIL import Image, UnidentifiedImageError

from src.constants import COCO_DIRECTORY, IMAGENETTE_DIRECTORY

DATASET_SELECTORS = ("coco", "imagenette")
SUPPORTED_DATASETS_BY_FAMILY = {
    "yolo": ("coco",),
    "small-vlm": ("imagenette",),
}
IMAGE_SUFFIXES = {".jpeg", ".jpg", ".png"}


class DatasetImageError(RuntimeError):
    """An individual dataset image could not be decoded."""


@dataclass(frozen=True, slots=True)
class BenchmarkImage:
    """One stable, indexed input from a benchmark dataset."""

    index: int
    sample_id: str
    path: Path

    @contextmanager
    def open_rgb(self) -> Iterator[Image.Image]:
        """Decode this sample as RGB and close its image resources afterward."""
        try:
            with Image.open(self.path) as source:
                image = source.convert("RGB")
        except (OSError, UnidentifiedImageError, ValueError) as error:
            raise DatasetImageError(f"Unreadable image {self.sample_id}: {error}") from error
        try:
            yield image
        finally:
            image.close()


def load_benchmark_dataset(
    selector: str,
    *,
    limit: int | None = None,
    coco_directory: Path = COCO_DIRECTORY,
    imagenette_directory: Path = IMAGENETTE_DIRECTORY,
) -> tuple[BenchmarkImage, ...]:
    """Validate and return one dataset in deterministic order."""
    if selector not in DATASET_SELECTORS:
        raise ValueError(f"Unknown dataset selector: {selector}")
    if limit is not None and limit <= 0:
        raise ValueError("Dataset limit must be a positive integer")

    paths = (
        _coco_image_paths(coco_directory)
        if selector == "coco"
        else _imagenette_image_paths(imagenette_directory)
    )
    if limit is not None:
        paths = paths[:limit]
    imagenette_root = imagenette_directory.resolve()
    return tuple(
        BenchmarkImage(
            index=index,
            sample_id=(
                path.name
                if selector == "coco"
                else str(path.relative_to(imagenette_root))
            ),
            path=path,
        )
        for index, path in enumerate(paths)
    )


def validate_dataset_compatibility(family: str, dataset: str) -> None:
    """Reject unsupported model-family and dataset combinations."""
    if family not in SUPPORTED_DATASETS_BY_FAMILY:
        raise ValueError(f"Unknown model family: {family}")
    supported = SUPPORTED_DATASETS_BY_FAMILY[family]
    if dataset not in supported:
        raise ValueError(
            f"{family} does not support dataset {dataset}; supported: {', '.join(supported)}"
        )


def _coco_image_paths(directory: Path) -> list[Path]:
    images = directory / "images"
    annotations = directory / "annotations" / "instances_val2017.json"
    if not images.is_dir() or not annotations.is_file():
        raise FileNotFoundError(
            f"Invalid COCO layout under {directory}; run the dataset downloader first"
        )
    paths = sorted(
        path for path in images.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not paths:
        raise RuntimeError(f"COCO dataset contains no images under {images}")
    return paths


def _imagenette_image_paths(directory: Path) -> list[Path]:
    images = directory / "validation" / "images"
    manifest = directory / "validation_labels.csv"
    if not images.is_dir() or not manifest.is_file():
        raise FileNotFoundError(
            f"Invalid Imagenette layout under {directory}; run the dataset downloader first"
        )

    with manifest.open(newline="", encoding="utf-8") as manifest_file:
        reader = csv.DictReader(manifest_file)
        if reader.fieldnames is None or "image_path" not in reader.fieldnames:
            raise RuntimeError(f"Imagenette manifest has no image_path column: {manifest}")
        relative_paths = [Path(row["image_path"]) for row in reader]

    paths = sorted(_validated_manifest_path(directory, path) for path in relative_paths)
    if not paths:
        raise RuntimeError(f"Imagenette manifest contains no images: {manifest}")
    if len(paths) != len(set(paths)):
        raise RuntimeError(f"Imagenette manifest contains duplicate image paths: {manifest}")
    missing = next((path for path in paths if not path.is_file()), None)
    if missing is not None:
        raise FileNotFoundError(f"Imagenette manifest image is missing: {missing}")
    return paths


def _validated_manifest_path(directory: Path, relative_path: Path) -> Path:
    if relative_path.is_absolute():
        raise RuntimeError(f"Imagenette manifest path must be relative: {relative_path}")
    resolved_directory = directory.resolve()
    resolved_path = (directory / relative_path).resolve()
    if resolved_path != resolved_directory and resolved_directory not in resolved_path.parents:
        raise RuntimeError(f"Imagenette manifest path escapes dataset: {relative_path}")
    return resolved_path
