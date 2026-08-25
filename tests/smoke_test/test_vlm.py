"""Offline tests for the VLM smoke-test adapter."""

import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from src.smoke_test.vlm import prepare_vlm_inputs, run_vlm_smoke_test


class FakeOutOfMemoryError(RuntimeError):
    pass


class FakeCuda:
    OutOfMemoryError = FakeOutOfMemoryError

    def __init__(self) -> None:
        self.empty_cache_calls = 0

    @staticmethod
    def is_available() -> bool:
        return True

    @staticmethod
    def get_device_name(index: int) -> str:
        return "Fake Orin"

    @staticmethod
    def synchronize() -> None:
        return None

    @staticmethod
    def reset_peak_memory_stats(device: str) -> None:
        return None

    @staticmethod
    def max_memory_allocated(device: str) -> int:
        return 1024 * 1024**2

    def empty_cache(self) -> None:
        self.empty_cache_calls += 1


class FakeTensor:
    def __init__(self, token_count: int) -> None:
        self.shape = (1, token_count)


class FakeOutput(FakeTensor):
    def __getitem__(self, key):
        token_slice = key[1]
        start = token_slice.start or 0
        return FakeOutput(self.shape[-1] - start)


class FakeInputs(dict):
    def __init__(self) -> None:
        super().__init__(input_ids=FakeTensor(3), pixel_values="pixels")
        self.device = None

    def to(self, device: str):
        self.device = device
        return self


class FakeProcessor:
    def __init__(self) -> None:
        self.inputs = FakeInputs()
        self.template_arguments = None
        self.processor_arguments = None

    def apply_chat_template(self, messages, **kwargs):
        self.template_arguments = (messages, kwargs)
        return "formatted prompt"

    def __call__(self, *args, **kwargs):
        self.processor_arguments = (args, kwargs)
        return self.inputs

    @staticmethod
    def batch_decode(tokens, skip_special_tokens: bool):
        return ["  synthetic description  "]


def make_fake_torch() -> SimpleNamespace:
    return SimpleNamespace(
        __version__="2.8.0",
        version=SimpleNamespace(cuda="12.6"),
        cuda=FakeCuda(),
        inference_mode=nullcontext,
    )


def write_test_image(path: Path) -> None:
    path.write_text("P3\n1 1\n255\n255 0 0\n", encoding="ascii")


class VlmInputPreparationTests(unittest.TestCase):
    def test_chat_template_models_use_shared_prompt_path(self) -> None:
        processor = FakeProcessor()

        inputs = prepare_vlm_inputs(
            "smolvlm2-256m",
            processor,
            SimpleNamespace(),
            "cuda:0",
        )

        messages, template_options = processor.template_arguments
        self.assertEqual("user", messages[0]["role"])
        self.assertEqual("Describe this image briefly.", messages[0]["content"][1]["text"])
        self.assertFalse(template_options["tokenize"])
        self.assertEqual({"num_frames": 1}, template_options["processor_kwargs"])
        self.assertEqual("formatted prompt", processor.processor_arguments[1]["text"])
        self.assertEqual(
            {"do_image_splitting": False},
            processor.processor_arguments[1]["images_kwargs"],
        )
        self.assertEqual("cuda:0", inputs.device)

    def test_phi_vision_uses_numbered_image_token(self) -> None:
        processor = FakeProcessor()

        prepare_vlm_inputs(
            "phi-3.5-vision",
            processor,
            SimpleNamespace(),
            "cuda:0",
        )

        prompt = processor.processor_arguments[0][0]
        self.assertIn("<|image_1|>", prompt)
        self.assertIn("Describe this image briefly.", prompt)
        self.assertIsNone(processor.template_arguments)


class VlmSmokeTestAdapterTests(unittest.TestCase):
    def test_runs_warmup_and_measured_generation(self) -> None:
        torch = make_fake_torch()
        processor = FakeProcessor()
        model = Mock()
        model.generate.return_value = FakeOutput(5)
        loader = Mock(return_value=(model, processor))
        clock = Mock(side_effect=(1.0, 1.5, 2.0, 2.4))

        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory) / "image.ppm"
            write_test_image(image)
            result = run_vlm_smoke_test(
                "smolvlm2-256m",
                image,
                torch_module=torch,
                model_loader=loader,
                clock=clock,
            )

        loader.assert_called_once_with(
            "smolvlm2-256m",
            device="cuda:0",
            precision="fp16",
            torch_module=torch,
        )
        self.assertEqual(2, model.generate.call_count)
        self.assertEqual(16, model.generate.call_args.kwargs["max_new_tokens"])
        self.assertFalse(model.generate.call_args.kwargs["do_sample"])
        self.assertEqual("passed", result.status)
        self.assertEqual("fp16", result.runtime_precision)
        self.assertEqual(2, result.generated_tokens)
        self.assertEqual("synthetic description", result.prediction_summary)
        self.assertAlmostEqual(0.5, result.load_time_seconds)
        self.assertAlmostEqual(0.4, result.inference_time_seconds)
        self.assertEqual(1024.0, result.peak_cuda_memory_mib)
        self.assertEqual(1, torch.cuda.empty_cache_calls)

    def test_cuda_out_of_memory_is_recorded(self) -> None:
        torch = make_fake_torch()
        model = Mock()
        model.generate.side_effect = FakeOutOfMemoryError("CUDA out of memory")

        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory) / "image.ppm"
            write_test_image(image)
            result = run_vlm_smoke_test(
                "qwen2.5-vl-3b",
                image,
                torch_module=torch,
                model_loader=Mock(return_value=(model, FakeProcessor())),
                clock=Mock(side_effect=(1.0, 1.2)),
            )

        self.assertEqual("failed", result.status)
        self.assertEqual("cuda_out_of_memory", result.error_type)
        self.assertEqual(1, torch.cuda.empty_cache_calls)


if __name__ == "__main__":
    unittest.main()
