"""Offline tests for deterministic benchmark datasets."""

import csv
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from src.benchmark.datasets import (
    DatasetImageError,
    load_benchmark_dataset,
)


def write_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with Image.new("RGB", (1, 1), "red") as image:
        image.save(path)


class BenchmarkDatasetTests(unittest.TestCase):
    def test_coco_is_sorted_and_limited(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "annotations").mkdir()
            (root / "annotations/instances_val2017.json").write_text("{}")
            write_image(root / "images/0002.jpg")
            write_image(root / "images/0001.jpg")

            samples = load_benchmark_dataset(
                "coco",
                limit=1,
                coco_directory=root,
            )

        self.assertEqual(1, len(samples))
        self.assertEqual(0, samples[0].index)
        self.assertEqual("0001.jpg", samples[0].sample_id)
        self.assertEqual(2, samples.total_image_count)
        self.assertEqual(1, samples.requested_limit)
        self.assertEqual("limited", samples.run_scope)

    def test_coco_rejects_invalid_layout(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(FileNotFoundError, "Invalid COCO layout"):
                load_benchmark_dataset("coco", coco_directory=Path(directory))

    def test_imagenette_uses_stable_manifest_order_without_labels(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = Path("validation/images/class/first.JPEG")
            second = Path("validation/images/class/second.JPEG")
            write_image(root / second)
            write_image(root / first)
            with (root / "validation_labels.csv").open(
                "w",
                newline="",
                encoding="utf-8",
            ) as manifest:
                writer = csv.writer(manifest)
                writer.writerow(("image_path", "class_id"))
                writer.writerow((second, 1))
                writer.writerow((first, 0))

            samples = load_benchmark_dataset(
                "imagenette",
                imagenette_directory=root,
            )

        self.assertEqual([str(first), str(second)], [sample.sample_id for sample in samples])
        self.assertEqual(2, samples.total_image_count)
        self.assertIsNone(samples.requested_limit)
        self.assertEqual("full", samples.run_scope)

    def test_corrupt_image_has_specific_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "annotations").mkdir()
            (root / "annotations/instances_val2017.json").write_text("{}")
            corrupt = root / "images/corrupt.jpg"
            corrupt.parent.mkdir()
            corrupt.write_text("not an image")
            sample = load_benchmark_dataset("coco", coco_directory=root)[0]

            with self.assertRaisesRegex(DatasetImageError, "Unreadable image corrupt.jpg"):
                with sample.open_rgb():
                    pass

if __name__ == "__main__":
    unittest.main()
