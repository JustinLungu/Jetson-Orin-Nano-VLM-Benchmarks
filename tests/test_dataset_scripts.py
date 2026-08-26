"""Offline tests for dataset downloads and their shell entry point."""

import io
import shutil
import subprocess
import tarfile
import tempfile
import unittest
import zipfile
from pathlib import Path

from src import download_datasets
from src.constants import IMAGENETTE_CLASS_IDS

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DATASET_SCRIPT = REPOSITORY_ROOT / "scripts/download_datasets.sh"


class DatasetScriptTests(unittest.TestCase):
    def test_script_has_valid_shell_syntax(self) -> None:
        subprocess.run(["bash", "-n", str(DATASET_SCRIPT)], check=True)


class DatasetSelectionTests(unittest.TestCase):
    def test_all_selects_both_datasets(self) -> None:
        self.assertEqual(["coco", "imagenette"], download_datasets.select_datasets(["all"]))


class DatasetDownloadTests(unittest.TestCase):
    def test_coco_download_prepares_expected_layout(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            images_archive = temporary / "images.zip"
            annotations_archive = temporary / "annotations.zip"
            with zipfile.ZipFile(images_archive, "w") as archive:
                for index in range(5000):
                    archive.writestr(f"val2017/{index:012d}.jpg", b"")
            with zipfile.ZipFile(annotations_archive, "w") as archive:
                archive.writestr("annotations/instances_val2017.json", "{}")

            def downloader(url: str, destination: Path) -> None:
                source = annotations_archive if "annotations" in url else images_archive
                shutil.copyfile(source, destination)

            destination = temporary / "coco"
            result = download_datasets.download_coco(destination, downloader)

            self.assertEqual(destination, result)
            self.assertEqual(5000, len(list((destination / "images").glob("*.jpg"))))
            self.assertTrue((destination / "annotations/instances_val2017.json").is_file())

    def test_imagenette_download_builds_imagenet_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            archive_path = temporary / "imagenette.tgz"
            with tarfile.open(archive_path, "w:gz") as archive:
                for synset in IMAGENETTE_CLASS_IDS:
                    data = b"image"
                    member = tarfile.TarInfo(
                        f"imagenette2-160/val/{synset}/{synset}.JPEG"
                    )
                    member.size = len(data)
                    archive.addfile(member, io.BytesIO(data))

            def downloader(url: str, destination: Path) -> None:
                shutil.copyfile(archive_path, destination)

            destination = temporary / "imagenette"
            result = download_datasets.download_imagenette(destination, downloader)

            self.assertEqual(destination, result)
            manifest_lines = (destination / "validation_labels.csv").read_text().splitlines()
            self.assertEqual(11, len(manifest_lines))
            self.assertEqual("image_path,class_id", manifest_lines[0])
            self.assertTrue((destination / "validation/images/n01440764/n01440764.JPEG").is_file())
