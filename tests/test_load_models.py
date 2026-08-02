"""Offline tests for model selection and checkpoint bookkeeping."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src import load_models


class ModelSelectionTests(unittest.TestCase):
    def test_all_selects_every_model_in_registry_order(self) -> None:
        self.assertEqual(list(load_models.MODEL_SELECTORS), load_models.select_models(["all"]))

    def test_family_and_individual_selections_are_deduplicated(self) -> None:
        self.assertEqual(
            list(load_models.MODEL_GROUPS["yolo"]),
            load_models.select_models(["yolo", "yolov8n"]),
        )

    def test_unknown_selector_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unknown model selector"):
            load_models.select_models(["not-a-model"])


class ModelDownloadTests(unittest.TestCase):
    def test_yolo_loader_must_create_expected_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory)

            def fake_loader(model_name: str, task: str) -> None:
                self.assertEqual("detect", task)
                Path(model_name).touch()

            with patch.object(load_models, "YOLO_MODEL_DIRECTORY", target):
                result = load_models.download_yolo_model("yolov8n", loader=fake_loader)

            self.assertEqual(target / "yolov8n.pt", result)


if __name__ == "__main__":
    unittest.main()
