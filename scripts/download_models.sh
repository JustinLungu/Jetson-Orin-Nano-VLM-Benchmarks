#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIRECTORY="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPOSITORY_ROOT="$(cd -- "${SCRIPT_DIRECTORY}/.." && pwd -P)"

if [[ "$#" -eq 0 ]]; then
    printf 'Usage: %s <model> [<model> ...] | yolo | small-vlm | all\n' "$0" >&2
    exit 2
fi

cd "${REPOSITORY_ROOT}"

UV_GROUPS=(--group download)
for selector in "$@"; do
    case "${selector}" in
        yolo|yolov8n|yolo11n|yolo26n|all)
            UV_GROUPS+=(--group yolo)
            break
            ;;
    esac
done

uv run --frozen "${UV_GROUPS[@]}" python -m src.load_models "$@"
