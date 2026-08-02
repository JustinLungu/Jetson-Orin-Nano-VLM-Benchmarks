"""Offline tests for model selection and checkpoint bookkeeping."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from src import load_models
from src.constants import MODEL_REPOSITORIES


class ModelSelectionTests(unittest.TestCase):
    def test_all_selects_every_model_in_registry_order(self) -> None:
        self.assertEqual(list(load_models.MODEL_SELECTORS), load_models.select_models(["all"]))

    def test_small_vlm_group_uses_constants_registry(self) -> None:
        self.assertEqual(
            list(MODEL_REPOSITORIES),
            load_models.select_models(["small-vlm"]),
        )

    def test_family_and_individual_selections_are_deduplicated(self) -> None:
        self.assertEqual(
            list(load_models.MODEL_GROUPS["yolo"]),
            load_models.select_models(["yolo", "yolov8n"]),
        )

    def test_unknown_selector_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unknown model selector"):
            load_models.select_models(["not-a-model"])

    def test_all_cannot_be_combined_with_other_selectors(self) -> None:
        with self.assertRaisesRegex(ValueError, "Use 'all' alone"):
            load_models.select_models(["all", "yolov8n"])


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

    def test_vlm_download_pins_revision_and_writes_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory)
            calls = []

            class FakeApi:
                @staticmethod
                def model_info(repository: str) -> SimpleNamespace:
                    self.assertEqual(MODEL_REPOSITORIES["smolvlm2-256m"], repository)
                    return SimpleNamespace(sha="test-revision")

            def fake_downloader(**kwargs: str) -> None:
                calls.append(kwargs)
                (Path(kwargs["local_dir"]) / "config.json").write_text("{}\n")

            with patch.object(load_models, "SMALL_VLM_MODEL_DIRECTORY", target):
                result = load_models.download_small_vlm_model(
                    "smolvlm2-256m",
                    FakeApi(),
                    fake_downloader,
                )

            self.assertEqual(target / "smolvlm2-256m", result)
            self.assertEqual("test-revision", calls[0]["revision"])
            metadata = json.loads((result / "download_metadata.json").read_text())
            self.assertEqual("test-revision", metadata["revision"])
            self.assertEqual(MODEL_REPOSITORIES["smolvlm2-256m"], metadata["repository"])

    def test_vlm_download_rejects_snapshot_without_config(self) -> None:
        api = SimpleNamespace(model_info=lambda repository: SimpleNamespace(sha="revision"))

        with tempfile.TemporaryDirectory() as directory:
            with patch.object(load_models, "SMALL_VLM_MODEL_DIRECTORY", Path(directory)):
                with self.assertRaisesRegex(RuntimeError, "has no config.json"):
                    load_models.download_small_vlm_model(
                        "smolvlm2-256m",
                        api,
                        lambda **kwargs: None,
                    )


if __name__ == "__main__":
    unittest.main()
