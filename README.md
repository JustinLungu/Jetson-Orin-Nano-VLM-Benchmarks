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

The script installs Hugging Face download tooling from the locked `download` group. It adds the heavier `yolo` group only when a YOLO selector is requested. VLM snapshots are pinned to their resolved Hugging Face commit and store that revision in `download_metadata.json` for reproducibility.

Only files required for Transformers inference are downloaded. Alternative ONNX, GGUF, and framework checkpoints are excluded to avoid storing duplicate representations of the same model.

## Jetson-optimized inference

VLM checkpoints are always loaded as FP16 through the shared loader, regardless of the dtype used by the files on disk:

```python
from src.load_models import load_vlm_fp16

model, processor = load_vlm_fp16("smolvlm2-500m")
```

The loader uses local files only and fails if the resulting model parameters are not FP16. This makes memory comparisons consistent across model repositories that publish FP32 or BF16 checkpoints.

On the Jetson, install NVIDIA's JetPack-compatible CUDA build of PyTorch first. Do not replace it with the generic PyPI Torch wheel. Sync the remaining inference packages in inexact mode so uv preserves that manually installed Torch build:

```bash
uv sync --locked --group inference --inexact
```

The existing `smolvlm2-500m` checkpoint is correct. Its `onnx/` directory is not used by this benchmark path and can be deleted to recover roughly 5.4 GB; retain `model.safetensors` and the top-level configuration/tokenizer files.

YOLO `.pt` files are retained as the portable benchmark inputs, including custom-trained checkpoints. Runtime-specific exports such as TensorRT can be added later if backend comparison becomes an explicit benchmark requirement.

## Downloading datasets

Download COCO 2017 validation images and annotations for object-detection evaluation:

```bash
./scripts/download_datasets.sh coco
```

Download Imagenette-160 validation for a compact ImageNet-compatible classification dataset:

```bash
./scripts/download_datasets.sh imagenette
```

Generated dataset contents are stored under `datasets/` and ignored by Git. Only the layout and source documentation are committed. Imagenette contains ten ImageNet classes and is intended for lightweight development; it is not the full ImageNet-1K validation dataset.
