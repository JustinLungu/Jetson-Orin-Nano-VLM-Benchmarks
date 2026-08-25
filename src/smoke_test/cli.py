"""Command-line orchestration for sequential model smoke tests."""

import argparse
import json
from collections.abc import Callable, Sequence
from datetime import datetime, timezone
from pathlib import Path

from src.constants import (
    MODEL_GROUPS,
    MODEL_SELECTORS,
    REPOSITORY_ROOT,
    YOLO_MODELS,
)
from src.smoke_test.result import SmokeTestResult
from src.smoke_test.vlm import run_vlm_smoke_test
from src.smoke_test.yolo import run_yolo_smoke_test
from src.utils import select_model_names

SMOKE_IMAGE = REPOSITORY_ROOT / "tests" / "fixtures" / "smoke_test.ppm"
SMOKE_RESULTS_DIRECTORY = REPOSITORY_ROOT / "results" / "smoke"
SmokeRunner = Callable[[str, Path], SmokeTestResult]


def select_smoke_models(arguments: list[str]) -> list[str]:
    """Expand model and family selectors in stable registry order."""
    return select_model_names(arguments, MODEL_GROUPS, MODEL_SELECTORS)


def run_selected_models(
    selectors: list[str],
    image_path: Path,
    *,
    runners: dict[str, SmokeRunner] | None = None,
) -> list[SmokeTestResult]:
    """Run selected models sequentially and isolate unexpected runner failures."""
    runners = runners or {
        "yolo": run_yolo_smoke_test,
        "small-vlm": run_vlm_smoke_test,
    }
    results = []
    total = len(selectors)
    for index, selector in enumerate(selectors, start=1):
        family = family_for_selector(selector)
        print(f"[{index}/{total}] {selector}: running")
        try:
            result = runners[family](selector, image_path)
        except Exception as error:
            result = SmokeTestResult(
                model=selector,
                family=family,
                status="failed",
                device="cuda:0",
                runtime_versions={},
                error_type="runner_error",
                error_message=str(error),
            )
        results.append(result)
        print(format_result_summary(result))
    return results


def family_for_selector(selector: str) -> str:
    """Return the adapter family for a configured selector."""
    return "yolo" if selector in YOLO_MODELS else "small-vlm"


def format_result_summary(result: SmokeTestResult) -> str:
    """Format one compact terminal status line."""
    if result.status == "failed":
        return f"  FAILED ({result.error_type}): {result.error_message}"
    details = [f"inference={result.inference_time_seconds:.3f}s"]
    if result.peak_cuda_memory_mib is not None:
        details.append(f"peak_cuda={result.peak_cuda_memory_mib:.1f}MiB")
    if result.generated_tokens is not None:
        details.append(f"tokens={result.generated_tokens}")
    return "  PASSED " + " ".join(details)


def write_smoke_report(
    results: list[SmokeTestResult],
    image_path: Path,
    output_directory: Path = SMOKE_RESULTS_DIRECTORY,
    *,
    created_at: datetime | None = None,
) -> Path:
    """Write a timestamped JSON report and return its path."""
    created_at = created_at or datetime.now(timezone.utc)
    timestamp = created_at.strftime("%Y%m%dT%H%M%SZ")
    output_directory.mkdir(parents=True, exist_ok=True)
    destination = output_directory / f"smoke-{timestamp}.json"
    report = {
        "schema_version": 1,
        "created_at": created_at.isoformat(),
        "image": str(image_path),
        "results": [result.to_dict() for result in results],
    }
    destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return destination


def main(
    argv: Sequence[str] | None = None,
    *,
    runners: dict[str, SmokeRunner] | None = None,
    output_directory: Path = SMOKE_RESULTS_DIRECTORY,
    created_at: datetime | None = None,
) -> int:
    parser = argparse.ArgumentParser(
        description="Run single-image CUDA inference smoke tests.",
    )
    parser.add_argument(
        "models",
        nargs="+",
        metavar="MODEL",
        help="model selector(s), 'yolo', 'small-vlm', or 'all'",
    )
    arguments = parser.parse_args(argv)
    try:
        selectors = select_smoke_models(arguments.models)
    except ValueError as error:
        parser.error(str(error))

    results = run_selected_models(selectors, SMOKE_IMAGE, runners=runners)
    report_path = write_smoke_report(
        results,
        SMOKE_IMAGE,
        output_directory,
        created_at=created_at,
    )
    passed = sum(result.status == "passed" for result in results)
    print(f"\nPassed: {passed}/{len(results)}")
    print(f"Report: {report_path}")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
