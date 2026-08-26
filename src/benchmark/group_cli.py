"""Sequential orchestration for the four reproducible benchmark groups."""

import argparse
from collections.abc import Callable, Sequence

from src.benchmark.cli import desktop_is_active, main as run_single_benchmark

YOLO_BENCHMARK_CONFIGURATIONS = (
    ("yolov8n", None),
    ("yolo11n", None),
    ("yolo26n", None),
)
SMOLVLM_BENCHMARK_CONFIGURATIONS = (
    ("smolvlm2-256m", "fp16"),
    ("smolvlm2-256m", "fp32"),
    ("smolvlm2-500m", "fp16"),
    ("smolvlm2-500m", "fp32"),
    ("smolvlm2-2.2b", "fp16"),
)
BenchmarkCommand = Callable[[Sequence[str]], int]


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
    benchmark_command: BenchmarkCommand = run_single_benchmark,
    desktop_detector: Callable[[], bool] = desktop_is_active,
) -> int:
    parser = argparse.ArgumentParser(
        prog="./scripts/run_benchmark_group.sh",
        description="Run one model family sequentially on one image dataset.",
    )
    parser.add_argument("family", choices=("yolo", "smolvlm"))
    parser.add_argument("dataset", choices=("coco", "imagenette"))
    parser.add_argument("--warmup", type=int, help="excluded warm-up runs per model")
    parser.add_argument("--limit", type=int, help="development-only image limit")
    parser.add_argument("--image-size", type=int, help="YOLO square input size")
    parser.add_argument(
        "--allow-desktop-2.2b",
        dest="allow_desktop_2_2b",
        action="store_true",
        help="allow the memory-sensitive 2.2B run while the desktop is active",
    )
    arguments = parser.parse_args(argv)

    if arguments.warmup is not None and arguments.warmup < 0:
        parser.error("--warmup cannot be negative")
    if arguments.limit is not None and arguments.limit <= 0:
        parser.error("--limit must be a positive integer")
    if arguments.family == "smolvlm" and arguments.image_size is not None:
        parser.error("--image-size is only valid for the YOLO group")
    if arguments.family == "yolo" and arguments.allow_desktop_2_2b:
        parser.error("--allow-desktop-2.2b is only valid for the SmolVLM group")

    configurations = benchmark_group_configurations(arguments.family)
    if (
        arguments.family == "smolvlm"
        and desktop_detector()
        and not arguments.allow_desktop_2_2b
    ):
        print(
            "Benchmark group refused: SmolVLM2-2.2B requires a headless session. "
            "Stop the desktop or pass --allow-desktop-2.2b explicitly."
        )
        return 1

    total = len(configurations)
    for index, (model, precision) in enumerate(configurations, start=1):
        print(f"\n[{index}/{total}] {model} {precision or 'fp16'} on {arguments.dataset}")
        command = [model, arguments.dataset]
        if precision is not None:
            command.extend(("--precision", precision))
        if arguments.warmup is not None:
            command.extend(("--warmup", str(arguments.warmup)))
        if arguments.limit is not None:
            command.extend(("--limit", str(arguments.limit)))
        if arguments.image_size is not None:
            command.extend(("--image-size", str(arguments.image_size)))

        exit_code = benchmark_command(command)
        if exit_code != 0:
            print(f"Group stopped after {model} failed.")
            return exit_code

    print(f"\nCompleted benchmark group: {arguments.family} on {arguments.dataset}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
