"""Offline tests for model selection, downloads, validation, and FP16 loading."""

import json
import struct
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from src import load_models
from src.constants import MODEL_REPOSITORIES


def write_safetensors(path: Path, *, dtype: str = "F16", truncate: bool = False) -> None:
    """Write the smallest structurally valid safetensors fixture."""
    header = json.dumps(
        {"weight": {"dtype": dtype, "shape": [1], "data_offsets": [0, 2]}},
        separators=(",", ":"),
    ).encode()
    contents = struct.pack("<Q", len(header)) + header + b"\0\0"
    path.write_bytes(contents[:-1] if truncate else contents)


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
                destination = Path(kwargs["local_dir"])
                (destination / "config.json").write_text("{}\n")
                write_safetensors(destination / "model.safetensors")

            with patch.object(load_models, "SMALL_VLM_MODEL_DIRECTORY", target):
                result = load_models.download_small_vlm_model(
                    "smolvlm2-256m",
                    FakeApi(),
                    fake_downloader,
                )

            self.assertEqual(target / "smolvlm2-256m", result)
            self.assertEqual("test-revision", calls[0]["revision"])
            self.assertIn("*.safetensors", calls[0]["allow_patterns"])
            self.assertIn("onnx/*", calls[0]["ignore_patterns"])
            metadata = json.loads((result / "download_metadata.json").read_text())
            self.assertEqual("test-revision", metadata["revision"])
            self.assertEqual(MODEL_REPOSITORIES["smolvlm2-256m"], metadata["repository"])
            self.assertEqual({"F16": 1}, metadata["checkpoint_dtypes"])
            self.assertEqual(["fp32", "fp16"], metadata["supported_runtime_precisions"])

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

    def test_safetensors_validation_detects_truncated_weights(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory)
            write_safetensors(target / "model.safetensors", truncate=True)

            with self.assertRaisesRegex(RuntimeError, "Truncated safetensors"):
                load_models.inspect_safetensors(target)


class FakeModel:
    def __init__(self, dtype: str) -> None:
        self.parameter = SimpleNamespace(dtype=dtype)
        self.device = None

    def to(self, device: str):
        self.device = device
        return self

    def eval(self):
        return self

    def parameters(self):
        return iter((self.parameter,))


class FakeLoader:
    arguments = None
    model = FakeModel("float16")

    @classmethod
    def from_pretrained(cls, path: Path, **kwargs):
        cls.arguments = (path, kwargs)
        return cls.model


class FakeProcessor:
    arguments = None

    @classmethod
    def from_pretrained(cls, path: Path, **kwargs):
        cls.arguments = (path, kwargs)
        return "processor"


class ModelLoadingTests(unittest.TestCase):
    def test_loader_enforces_fp16_and_local_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model_directory = root / "smolvlm2-500m"
            model_directory.mkdir()
            (model_directory / "config.json").write_text("{}\n")
            fake_transformers = SimpleNamespace(
                AutoModelForImageTextToText=FakeLoader,
                AutoProcessor=FakeProcessor,
            )
            fake_torch = SimpleNamespace(float16="float16", float32="float32")

            with patch.object(load_models, "SMALL_VLM_MODEL_DIRECTORY", root):
                model, processor = load_models.load_vlm_fp16(
                    "smolvlm2-500m",
                    torch_module=fake_torch,
                    transformers_module=fake_transformers,
                )

            self.assertEqual("processor", processor)
            self.assertEqual("cuda", model.device)
            self.assertEqual("float16", FakeLoader.arguments[1]["dtype"])
            self.assertTrue(FakeLoader.arguments[1]["local_files_only"])

    def test_loader_rejects_non_fp16_parameters(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model_directory = root / "smolvlm2-500m"
            model_directory.mkdir()
            (model_directory / "config.json").touch()

            class Float32Loader:
                @classmethod
                def from_pretrained(cls, path: Path, **kwargs):
                    return FakeModel("float32")

            transformers = SimpleNamespace(
                AutoModelForImageTextToText=Float32Loader,
                AutoProcessor=FakeProcessor,
            )

            with patch.object(load_models, "SMALL_VLM_MODEL_DIRECTORY", root):
                with self.assertRaisesRegex(RuntimeError, "Expected every.*fp16"):
                    load_models.load_vlm_fp16(
                        "smolvlm2-500m",
                        torch_module=SimpleNamespace(float16="float16", float32="float32"),
                        transformers_module=transformers,
                    )

    def test_loader_supports_native_fp32_for_small_models(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model_directory = root / "smolvlm2-256m"
            model_directory.mkdir()
            (model_directory / "config.json").touch()
            fp32_model = FakeModel("float32")

            class Float32Loader:
                arguments = None

                @classmethod
                def from_pretrained(cls, path: Path, **kwargs):
                    cls.arguments = (path, kwargs)
                    return fp32_model

            transformers = SimpleNamespace(
                AutoModelForImageTextToText=Float32Loader,
                AutoProcessor=FakeProcessor,
            )
            torch = SimpleNamespace(float16="float16", float32="float32")

            with patch.object(load_models, "SMALL_VLM_MODEL_DIRECTORY", root):
                model, _ = load_models.load_vlm(
                    "smolvlm2-256m",
                    precision="fp32",
                    torch_module=torch,
                    transformers_module=transformers,
                )

            self.assertEqual("float32", Float32Loader.arguments[1]["dtype"])
            self.assertEqual("cuda", model.device)

    def test_loader_rejects_fp32_for_2_2b(self) -> None:
        with self.assertRaisesRegex(ValueError, "does not support fp32"):
            load_models.load_vlm(
                "smolvlm2-2.2b",
                precision="fp32",
                torch_module=SimpleNamespace(float16="float16", float32="float32"),
                transformers_module=SimpleNamespace(),
            )

    def test_loader_requires_prepared_fp16_for_2_2b(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model_directory = root / "smolvlm2-2.2b"
            model_directory.mkdir()
            (model_directory / "config.json").touch()

            with patch.object(load_models, "SMALL_VLM_MODEL_DIRECTORY", root):
                with self.assertRaisesRegex(
                    FileNotFoundError,
                    "prepare_fp16_model.sh smolvlm2-2.2b",
                ):
                    load_models.load_vlm(
                        "smolvlm2-2.2b",
                        precision="fp16",
                        torch_module=SimpleNamespace(
                            float16="float16",
                            float32="float32",
                        ),
                        transformers_module=SimpleNamespace(),
                    )


if __name__ == "__main__":
    unittest.main()
