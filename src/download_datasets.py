"""Download and prepare benchmark validation datasets."""

import argparse
import csv
import shutil
import tempfile
from pathlib import Path
from typing import Callable

from src.constants import (
    COCO_ANNOTATIONS_URL,
    COCO_DIRECTORY,
    COCO_IMAGES_URL,
    IMAGENETTE_CLASS_IDS,
    IMAGENETTE_DIRECTORY,
    IMAGENETTE_URL,
)
from src.utils import download_file, extract_tar_prefix_safely, extract_zip_safely

DATASET_SELECTORS = ("coco", "imagenette")


def select_datasets(arguments: list[str]) -> list[str]:
    """Resolve dataset selectors while preserving registry order."""
    if not arguments:
        raise ValueError("Select at least one dataset or 'all'")
    if "all" in arguments:
        if len(arguments) != 1:
            raise ValueError("Use 'all' alone, or select individual datasets")
        return list(DATASET_SELECTORS)

    unknown = [argument for argument in arguments if argument not in DATASET_SELECTORS]
    if unknown:
        raise ValueError(f"Unknown dataset selector: {unknown[0]}")
    return list(dict.fromkeys(arguments))


def download_coco(
    destination: Path = COCO_DIRECTORY,
    downloader: Callable[[str, Path], None] = download_file,
) -> Path:
    """Download and prepare COCO 2017 validation images and annotations."""
    images_destination = destination / "images"
    annotations_destination = destination / "annotations"
    if images_destination.exists() or annotations_destination.exists():
        raise FileExistsError(f"COCO validation data already exists under {destination}")

    with tempfile.TemporaryDirectory(prefix="coco-val.") as temporary_directory:
        temporary = Path(temporary_directory)
        images_archive = temporary / "val2017.zip"
        annotations_archive = temporary / "annotations_trainval2017.zip"
        extracted = temporary / "extracted"
        extracted.mkdir()

        print("Downloading COCO 2017 validation images...")
        downloader(COCO_IMAGES_URL, images_archive)
        print("Downloading COCO 2017 annotations...")
        downloader(COCO_ANNOTATIONS_URL, annotations_archive)
        print("Extracting COCO validation data...")
        extract_zip_safely(images_archive, extracted)
        extract_zip_safely(annotations_archive, extracted)

        extracted_images = extracted / "val2017"
        extracted_annotations = extracted / "annotations"
        image_count = len(list(extracted_images.glob("*.jpg")))
        if image_count != 5000:
            raise RuntimeError(f"Expected 5000 COCO validation images, found {image_count}")
        if not (extracted_annotations / "instances_val2017.json").is_file():
            raise RuntimeError("COCO archive has no instances_val2017.json")

        destination.mkdir(parents=True, exist_ok=True)
        shutil.move(extracted_images, images_destination)
        shutil.move(extracted_annotations, annotations_destination)

    print(f"COCO 2017 validation ready: {image_count} images under {destination}")
    return destination


def download_imagenette(
    destination: Path = IMAGENETTE_DIRECTORY,
    downloader: Callable[[str, Path], None] = download_file,
) -> Path:
    """Download Imagenette-160 and build its ImageNet-compatible manifest."""
    validation_destination = destination / "validation"
    manifest_destination = destination / "validation_labels.csv"
    if validation_destination.exists() or manifest_destination.exists():
        raise FileExistsError(f"Imagenette validation data already exists under {destination}")

    with tempfile.TemporaryDirectory(prefix="imagenette.") as temporary_directory:
        temporary = Path(temporary_directory)
        archive = temporary / "imagenette2-160.tgz"
        extracted = temporary / "extracted"
        prepared = temporary / "prepared"
        images = prepared / "validation/images"
        images.mkdir(parents=True)

        print("Downloading Imagenette-160...")
        downloader(IMAGENETTE_URL, archive)
        print("Extracting Imagenette validation data...")
        extract_tar_prefix_safely(archive, extracted, "imagenette2-160/val")
        source_validation = extracted / "imagenette2-160/val"

        rows = []
        for synset, class_id in IMAGENETTE_CLASS_IDS.items():
            source_class = source_validation / synset
            if not source_class.is_dir():
                raise RuntimeError(f"Missing Imagenette class directory: {synset}")
            destination_class = images / synset
            shutil.move(source_class, destination_class)
            for image in sorted(destination_class.iterdir()):
                if image.suffix.lower() in {".jpeg", ".jpg", ".png"}:
                    rows.append((image.relative_to(prepared), class_id))

        if not rows:
            raise RuntimeError("Imagenette archive contained no validation images")
        prepared_manifest = prepared / "validation_labels.csv"
        with prepared_manifest.open("w", newline="", encoding="utf-8") as manifest:
            writer = csv.writer(manifest)
            writer.writerow(("image_path", "class_id"))
            writer.writerows(rows)

        destination.mkdir(parents=True, exist_ok=True)
        shutil.move(prepared / "validation", validation_destination)
        shutil.move(prepared_manifest, manifest_destination)

    print(f"Imagenette validation ready: {len(rows)} images under {destination}")
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description="Download benchmark validation datasets")
    parser.add_argument(
        "datasets",
        nargs="+",
        metavar="DATASET",
        help="dataset selector(s): coco, imagenette, or all",
    )
    arguments = parser.parse_args()
    try:
        selected = select_datasets(arguments.datasets)
    except ValueError as error:
        parser.error(str(error))

    for selector in selected:
        if selector == "coco":
            download_coco()
        else:
            download_imagenette()


if __name__ == "__main__":
    main()
