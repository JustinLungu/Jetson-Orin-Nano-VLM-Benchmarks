# Jetson-Orin-Nano-VLM-Benchmarks
Setup, deployment, and benchmarking of compact vision-language and object-detection models on NVIDIA Jetson Orin Nano. The project will compare memory use, latency, throughput, quantization, and video inference performance under Jetson resource constraints.

## Supported models

The current benchmark targets are:

| Family | Selector | Model |
| --- | --- | --- |
| Small VLM | `smolvlm2-256m` | Hugging Face SmolVLM2 256M Video Instruct |
| Small VLM | `smolvlm2-500m` | Hugging Face SmolVLM2 500M Video Instruct |
| Small VLM | `smolvlm2-2.2b` | Hugging Face SmolVLM2 2.2B Instruct |
| Small VLM | `qwen2.5-vl-3b` | Qwen2.5-VL 3B Instruct |
| Small VLM | `phi-3.5-vision` | Phi-3.5 Vision Instruct |
| Object detection | `yolov8n` | YOLOv8 Nano |
| Object detection | `yolo11n` | YOLO11 Nano |
| Object detection | `yolo26n` | YOLO26 Nano |

## Development setup

The project uses Python 3.10 and [uv](https://docs.astral.sh/uv/) for dependency and virtual-environment management.

```bash
uv sync --locked
```

This creates a `.venv` from the committed `uv.lock` file. Run project commands with `uv run`, or activate the environment manually:

```bash
source .venv/bin/activate
```

When dependencies change, update and commit the lockfile:

```bash
uv lock
```

## Downloading models

Model downloads are kept in the ignored `models/` directory. Select individual models, a family, or the entire configured model set:

```bash
./scripts/download_models.sh smolvlm2-256m yolo11n
./scripts/download_models.sh small-vlm
./scripts/download_models.sh yolo
./scripts/download_models.sh all
```

The script installs its downloader dependencies from the locked `models` dependency group. VLM snapshots are pinned to their resolved Hugging Face commit and store that revision in `download_metadata.json` for reproducibility.
