#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIRECTORY="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPOSITORY_ROOT="$(cd -- "${SCRIPT_DIRECTORY}/.." && pwd -P)"

if [[ "$#" -ne 1 ]]; then
    printf 'Usage: %s smolvlm2-2.2b\n' "$0" >&2
    exit 2
fi

cd "${REPOSITORY_ROOT}"
uv run --frozen --no-sync python -m src.model_preparation.fp16 "$1"
