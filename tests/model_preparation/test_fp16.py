"""Tests for persistent FP16 checkpoint preparation."""

import json
import tempfile
import unittest
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from src.model_preparation.fp16 import convert_checkpoint_to_fp16, prepared_fp16_path


class Fp16PreparationTests(unittest.TestCase):
    def test_converts_shards_and_preserves_source(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "model"
            destination = source / "fp16"
            source.mkdir()
            (source / "config.json").write_text("{}\n", encoding="utf-8")
            save_file(
                {"layer.weight": torch.tensor([1.25, -2.5], dtype=torch.float32)},
                source / "model-00001-of-00001.safetensors",
            )
            (source / "model.safetensors.index.json").write_text(
                json.dumps(
                    {
                        "metadata": {"total_size": 8},
                        "weight_map": {
                            "layer.weight": "model-00001-of-00001.safetensors",
                        },
                    }
                ),
                encoding="utf-8",
            )

            result = convert_checkpoint_to_fp16(source, destination)

            self.assertEqual(destination, result)
            self.assertEqual(torch.float32, load_file(source / "model-00001-of-00001.safetensors")["layer.weight"].dtype)
            converted_file = next(destination.glob("model-fp16-*.safetensors"))
            self.assertEqual(torch.float16, load_file(converted_file)["layer.weight"].dtype)
            self.assertEqual(destination, prepared_fp16_path(source))
            index = json.loads((destination / "model.safetensors.index.json").read_text())
            self.assertEqual(4, index["metadata"]["total_size"])

    def test_rejects_existing_destination(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "model"
            destination = source / "fp16"
            destination.mkdir(parents=True)

            with self.assertRaisesRegex(FileExistsError, "already exists"):
                convert_checkpoint_to_fp16(source, destination)


if __name__ == "__main__":
    unittest.main()
