"""Offline tests for load-once inference sessions."""

import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from PIL import Image

from src.inference.vlm import VlmInferenceSession
from src.inference.yolo import YoloInferenceSession


class FakeTensor:
    def __init__(self, token_count: int) -> None:
        self.shape = (1, token_count)


class FakeOutput(FakeTensor):
    def __getitem__(self, key):
        start = key[1].start or 0
        return FakeOutput(self.shape[-1] - start)


class FakePixelTensor:
    shape = (1, 1, 3, 384, 512)


class FakeInputs(dict):
    def __init__(self) -> None:
        super().__init__(input_ids=FakeTensor(3), pixel_values=FakePixelTensor())

    def to(self, device: str):
        return self


class FakeProcessor:
    @staticmethod
    def apply_chat_template(messages, **kwargs):
        return "formatted prompt"

    @staticmethod
    def batch_decode(tokens, skip_special_tokens: bool):
        return ["synthetic description"]

    def __call__(self, **kwargs):
        return FakeInputs()


class InferenceSessionTests(unittest.TestCase):
    def test_vlm_loads_once_and_processes_multiple_images(self) -> None:
        torch = SimpleNamespace(inference_mode=nullcontext)
        model = Mock()
        model.generate.return_value = FakeOutput(5)
        loader = Mock(return_value=(model, FakeProcessor()))
        session = VlmInferenceSession(
            "smolvlm2-256m",
            torch_module=torch,
            model_loader=loader,
        )

        session.load()
        with Image.new("RGB", (1, 1)) as image:
            first = session.prepare(image)
            second = session.prepare(image)
            first_output = session.infer(first)
            second_output = session.infer(second)

        summary, generated_tokens = session.summarize(second_output, second)
        loader.assert_called_once()
        self.assertEqual(2, model.generate.call_count)
        self.assertEqual("synthetic description", summary)
        self.assertEqual(2, generated_tokens)
        self.assertEqual((512, 384), session.processed_image_size(second))

        session.close()
        self.assertIsNone(session.model)
        self.assertIsNone(session.processor)

    def test_yolo_loads_once_and_processes_multiple_images(self) -> None:
        model = Mock()
        model.predict.return_value = [SimpleNamespace(boxes=[object()])]
        yolo_class = Mock(return_value=model)
        with tempfile.TemporaryDirectory() as directory:
            model_directory = Path(directory)
            (model_directory / "yolo11n.pt").touch()
            session = YoloInferenceSession(
                "yolo11n",
                torch_module=SimpleNamespace(),
                model_directory=model_directory,
                yolo_class=yolo_class,
            )
            session.load()
            with Image.new("RGB", (1, 1)) as image:
                first = session.prepare(image)
                second = session.prepare(image)
                first_output = session.infer(first)
                second_output = session.infer(second)

        summary, generated_tokens = session.summarize(second_output, second)
        yolo_class.assert_called_once()
        self.assertEqual(2, model.predict.call_count)
        self.assertFalse(second["rect"])
        self.assertEqual((320, 320), session.processed_image_size(second))
        self.assertEqual("detections=1", summary)
        self.assertIsNone(generated_tokens)

        session.close()
        self.assertIsNone(session.model)

    def test_yolo_rejects_a_size_ultralytics_would_adjust(self) -> None:
        with self.assertRaisesRegex(ValueError, "positive multiple of 32"):
            YoloInferenceSession(
                "yolo11n",
                image_size=641,
                torch_module=SimpleNamespace(),
                yolo_class=Mock(),
            )


if __name__ == "__main__":
    unittest.main()
