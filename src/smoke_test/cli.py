"""Command-line orchestration for sequential model smoke tests."""

import argparse
import json
import os
from collections.abc import Callable, Sequence
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

from src.constants import (
    MODEL_GROUPS,
    MODEL_SELECTORS,
    REPOSITORY_ROOT,
    YOLO_MODELS,
)
from src.smoke_test.result import SmokeTestResult
from src.smoke_test.runtime import TegrastatsMonitor
from src.smoke_test.vlm import run_vlm_smoke_test
from src.smoke_test.yolo import run_yolo_smoke_test
from src.utils import select_model_names

SMOKE_IMAGE = REPOSITORY_ROOT / "tests" / "fixtures" / "smoke_test.ppm"
SMOKE_RESULTS_DIRECTORY = REPOSITORY_ROOT / "results" / "smoke"
SmokeRunner = Callable[[str, Path], SmokeTestResult]
MonitorFactory = Callable[[], TegrastatsMonitor]
ResultCallback = Callable[[list[SmokeTestResult]], None]


def select_smoke_models(arguments: list[str]) -> list[str]:
    """Expand model and family selectors in stable registry order."""
    return select_model_names(arguments, MODEL_GROUPS, MODEL_SELECTORS)


def run_selected_models(
    selectors: list[str],
    image_path: Path,
    *,
    runners: dict[str, SmokeRunner] | None = None,
    monitor_factory: MonitorFactory = TegrastatsMonitor,
    result_callback: ResultCallback | None = None,
    vlm_precision: str = "fp16",
) -> list[SmokeTestResult]:
    """Run selected models sequentially and isolate unexpected runner failures."""
    runners = runners or {
        "yolo": run_yolo_smoke_test,
        "small-vlm": lambda selector, image: run_vlm_smoke_test(
            selector,
            image,
            precision=vlm_precision,
        ),
    }
    results = []
    total = len(selectors)
    for index, selector in enumerate(selectors, start=1):
        family = family_for_selector(selector)
        print(f"[{index}/{total}] {selector}: running")
        monitor = monitor_factory()
        monitor.start()
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
        finally:
            jetson_metrics = monitor.stop()
        result = replace(result, jetson_metrics=jetson_metrics)
        results.append(result)
        if result_callback is not None:
            result_callback(results)
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
    destination: Path | None = None,
    selected_models: list[str] | None = None,
    run_completed: bool = True,
) -> Path:
    """Atomically write a timestamped JSON report and return its path."""
    created_at = created_at or datetime.now(timezone.utc)
    output_directory.mkdir(parents=True, exist_ok=True)
    destination = destination or smoke_report_path(output_directory, created_at)
    report = {
        "schema_version": 1,
        "created_at": created_at.isoformat(),
        "image": str(image_path),
        "selected_models": selected_models or [result.model for result in results],
        "run_completed": run_completed,
        "results": [result.to_dict() for result in results],
    }
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, destination)
    return destination


def smoke_report_path(output_directory: Path, created_at: datetime) -> Path:
    """Return the stable destination used throughout one smoke-test run."""
    timestamp = created_at.strftime("%Y%m%dT%H%M%SZ")
    return output_directory / f"smoke-{timestamp}.json"


def main(
    argv: Sequence[str] | None = None,
    *,
    runners: dict[str, SmokeRunner] | None = None,
    output_directory: Path = SMOKE_RESULTS_DIRECTORY,
    created_at: datetime | None = None,
    monitor_factory: MonitorFactory = TegrastatsMonitor,
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
    parser.add_argument(
        "--precision",
        choices=("fp16", "fp32"),
        default="fp16",
        help="VLM runtime precision; FP32 is supported only by SmolVLM2-256M/500M",
    )
    arguments = parser.parse_args(argv)
    try:
        selectors = select_smoke_models(arguments.models)
    except ValueError as error:
        parser.error(str(error))

    report_created_at = created_at or datetime.now(timezone.utc)
    report_path = smoke_report_path(output_directory, report_created_at)

    def checkpoint_results(results: list[SmokeTestResult]) -> None:
        write_smoke_report(
            results,
            SMOKE_IMAGE,
            output_directory,
            created_at=report_created_at,
            destination=report_path,
            selected_models=selectors,
            run_completed=False,
        )

    checkpoint_results([])
    results = run_selected_models(
        selectors,
        SMOKE_IMAGE,
        runners=runners,
        monitor_factory=monitor_factory,
        result_callback=checkpoint_results,
        vlm_precision=arguments.precision,
    )
    write_smoke_report(
        results,
        SMOKE_IMAGE,
        output_directory,
        created_at=report_created_at,
        destination=report_path,
        selected_models=selectors,
        run_completed=True,
    )
    passed = sum(result.status == "passed" for result in results)
    print(f"\nPassed: {passed}/{len(results)}")
    print(f"Report: {report_path}")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
