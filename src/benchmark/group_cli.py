"""Sequential orchestration for the four reproducible benchmark groups."""

import argparse
from collections.abc import Callable, Sequence

from src.benchmark.cli import run_model_benchmark

YOLO_BENCHMARK_CONFIGURATIONS = (
    ("yolov8n", None),
    ("yolo11n", None),
    ("yolo26n", None),
)
SMOLVLM_BENCHMARK_CONFIGURATIONS = (
    ("smolvlm2-2.2b", "fp16"),
    ("smolvlm2-256m", "fp16"),
    ("smolvlm2-256m", "fp32"),
    ("smolvlm2-500m", "fp16"),
    ("smolvlm2-500m", "fp32"),
)
BenchmarkCommand = Callable[[str, str, str | None, int | None], int]


def benchmark_group_configurations(
    family: str,
) -> tuple[tuple[str, str | None], ...]:
    """Return the safe ordered configurations for one benchmark family."""
    if family == "yolo":
        return YOLO_BENCHMARK_CONFIGURATIONS
    if family == "smolvlm":
        return SMOLVLM_BENCHMARK_CONFIGURATIONS
    raise ValueError(f"Unknown benchmark family: {family}")


def main(
    argv: Sequence[str] | None = None,
    *,
    benchmark_command: BenchmarkCommand = run_model_benchmark,
) -> int:
    parser = argparse.ArgumentParser(
        prog="./scripts/run_benchmark_group.sh",
        description="Run one model family sequentially on one image dataset.",
    )
    parser.add_argument("family", choices=("yolo", "smolvlm"))
    parser.add_argument("dataset", choices=("coco", "imagenette"))
    parser.add_argument("--limit", type=int, help="development-only image limit")
    arguments = parser.parse_args(argv)

    if arguments.limit is not None and arguments.limit <= 0:
        parser.error("--limit must be a positive integer")

    configurations = benchmark_group_configurations(arguments.family)
    total = len(configurations)
    for index, (model, precision) in enumerate(configurations, start=1):
        print(f"\n[{index}/{total}] {model} {precision or 'fp16'} on {arguments.dataset}")
        exit_code = benchmark_command(
            model,
            arguments.dataset,
            precision,
            arguments.limit,
        )
        if exit_code != 0:
            print(f"Group stopped after {model} failed.")
            return exit_code

    print(f"\nCompleted benchmark group: {arguments.family} on {arguments.dataset}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
