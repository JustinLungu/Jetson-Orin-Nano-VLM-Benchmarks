"""Tests for the model smoke-test result contract."""

import json
import unittest

from src.smoke_results import SmokeTestResult


class SmokeTestResultTests(unittest.TestCase):
    def test_passed_result_is_json_serializable(self) -> None:
        result = SmokeTestResult(
            model="smolvlm2-256m",
            family="small-vlm",
            status="passed",
            device="cuda:0",
            runtime_versions={"torch": "2.8.0", "cuda": "12.6"},
            load_time_seconds=1.25,
            inference_time_seconds=0.5,
            peak_cuda_memory_mib=512.0,
            prediction_summary="A synthetic color grid.",
        )

        serialized = json.loads(json.dumps(result.to_dict()))

        self.assertEqual("passed", serialized["status"])
        self.assertEqual("smolvlm2-256m", serialized["model"])
        self.assertEqual(0.5, serialized["inference_time_seconds"])
        self.assertIsNone(serialized["error_type"])

    def test_failed_result_requires_an_error_type(self) -> None:
        with self.assertRaisesRegex(ValueError, "must include an error type"):
            SmokeTestResult(
                model="phi-3.5-vision",
                family="small-vlm",
                status="failed",
                device="cuda:0",
                runtime_versions={},
                error_message="Model loading failed",
            )

    def test_cuda_out_of_memory_failure_is_recorded(self) -> None:
        result = SmokeTestResult(
            model="qwen2.5-vl-3b",
            family="small-vlm",
            status="failed",
            device="cuda:0",
            runtime_versions={"torch": "2.8.0", "cuda": "12.6"},
            load_time_seconds=3.0,
            peak_cuda_memory_mib=7420.0,
            error_type="cuda_out_of_memory",
            error_message="CUDA out of memory",
        )

        serialized = result.to_dict()

        self.assertEqual("failed", serialized["status"])
        self.assertEqual("cuda_out_of_memory", serialized["error_type"])
        self.assertEqual("CUDA out of memory", serialized["error_message"])

    def test_passed_result_rejects_error_fields(self) -> None:
        with self.assertRaisesRegex(ValueError, "cannot contain an error"):
            SmokeTestResult(
                model="yolo11n",
                family="yolo",
                status="passed",
                device="cuda:0",
                runtime_versions={},
                error_type="unexpected_error",
            )

    def test_measurements_cannot_be_negative(self) -> None:
        with self.assertRaisesRegex(ValueError, "inference_time_seconds"):
            SmokeTestResult(
                model="yolo11n",
                family="yolo",
                status="passed",
                device="cuda:0",
                runtime_versions={},
                inference_time_seconds=-0.1,
            )


if __name__ == "__main__":
    unittest.main()
