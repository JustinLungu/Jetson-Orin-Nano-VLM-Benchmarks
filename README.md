# Jetson-Orin-Nano-VLM-Benchmarks
Setup, deployment, and benchmarking of compact vision-language and object-detection models on NVIDIA Jetson Orin Nano. The project will compare memory use, latency, throughput, quantization, and video inference performance under Jetson resource constraints.

## Development setup

The project uses Python 3.10 and [uv](https://docs.astral.sh/uv/) for dependency and virtual-environment management.

On Ubuntu, install `uv` with its official standalone installer:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source "$HOME/.local/bin/env"
uv --version
```

If the installer does not create `~/.local/bin/env`, start a new shell or run:

```bash
export PATH="$HOME/.local/bin:$PATH"
```

The JetPack image already supplies a compatible Python 3.10, so do not ask `uv` to
download a different Python. From the repository root, create the base environment:

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

### Jetson prerequisites

This setup has been validated against JetPack 6.2.1 / L4T 36.4.7, Ubuntu 22.04,
Python 3.10, and CUDA 12.6. Check another Jetson before installing inference packages:

```bash
cat /etc/nv_tegra_release
python3 --version
/usr/local/cuda/bin/nvcc --version
```

`nvidia-smi` is not a reliable Jetson validation command and may report that it cannot
communicate with the driver even when the integrated GPU stack is installed correctly.

Install the JetPack 6.2 CUDA 12.6 builds of PyTorch and TorchVision into `.venv`. The
`+simple` suffix on the Jetson package index is required:

```bash
sudo apt install -y libopenblas-dev
uv pip install --reinstall \
  torch==2.8.0 torchvision==0.23.0 \
  --index-url https://pypi.jetson-ai-lab.io/jp6/cu126/+simple
```

Do not replace these with generic PyPI wheels. The Jetson wheels require NumPy 1.x;
the committed `yolo` dependency group and lockfile enforce that constraint.

After installing the NVIDIA wheel, verify it before adding Transformers:

```bash
./scripts/check_jetson_gpu.sh
```

Only continue when `CUDA available` is `True`. Then sync the remaining inference
packages in inexact mode so `uv` preserves the manually installed NVIDIA Torch build:

```bash
uv sync --locked --group inference --inexact
```

Do not sync the `yolo` group on a Jetson. Its resolved generic PyPI `torch` and
`torchvision` packages can replace the JetPack builds. After installing Jetson PyTorch,
install the locked Ultralytics version with pip-style dependency handling, which accepts
the existing compatible Torch installation:

```bash
uv pip install ultralytics==8.4.115
./scripts/check_jetson_gpu.sh
```

The existing `smolvlm2-500m` checkpoint is correct. Its `onnx/` directory is not used by this benchmark path and can be deleted to recover roughly 5.4 GB; retain `model.safetensors` and the top-level configuration/tokenizer files.

YOLO `.pt` files are retained as the portable benchmark inputs, including custom-trained checkpoints. Runtime-specific exports such as TensorRT can be added later if backend comparison becomes an explicit benchmark requirement.

## Model smoke tests

Smoke tests answer one question: can a downloaded model complete CUDA inference on this
Jetson? They do not evaluate prediction accuracy. Run one model, a family, or every
configured model from the repository root:

> [!WARNING]
> On the 8 GB Orin Nano, do not run `qwen2.5-vl-3b` or
> `phi-3.5-vision` with the current FP16 loader. Qwen2.5-VL-3B caused the
> tested Jetson to freeze and restart. Consequently, `small-vlm` and `all`
> are not safe selectors on this configuration because they include these models.

```bash
./scripts/smoke_test_models.sh smolvlm2-256m
./scripts/smoke_test_models.sh smolvlm2-500m
./scripts/smoke_test_models.sh yolo11n
./scripts/smoke_test_models.sh yolo
```

The runner processes models sequentially and releases model objects and cached CUDA
memory between them. A failure is recorded without stopping the remaining models. The
command exits nonzero when any selected model fails.

Every model uses the fixed synthetic RGB image at `tests/fixtures/smoke_test.ppm`.
VLMs use the prompt `Describe this image briefly.`, deterministic generation, and a
16-token output limit. YOLO smoke tests use FP16 and a 320-pixel input size. These
settings minimize unified-memory pressure and establish functionality; their timings
are not substitutes for the full-dataset benchmark.

Terminal output provides a compact status summary:

```text
[1/1] yolo11n: running
  PASSED inference=0.050s peak_cuda=40.0MiB

Passed: 1/1
Report: results/smoke/smoke-20260825T145230Z.json
```

Each run writes a timestamped JSON report under `results/smoke/` containing runtime
versions, load and inference times, peak CUDA memory, Jetson CPU/GPU/power/temperature
summaries, prediction summaries, and VLM generated-token counts. The report is created
before inference begins and atomically updated after every model, so completed results
survive a later model crashing or rebooting the Jetson. `selected_models` records the
intended run and `run_completed` remains `false` until every selected model finishes.
Reports are device-specific and ignored by Git.

### Observed results on the 8 GB Orin Nano

These observations were collected on 2026-08-25 with JetPack 6.2.1 / L4T 36.4.7,
CUDA 12.6, PyTorch 2.8.0, FP16 inference, and the graphical desktop running. Linux and
the desktop occupied approximately 1.7 GB of the 7.6 GB visible unified memory before
the VLM runs. They are smoke-test measurements from one synthetic image, not final
full-dataset benchmark results.

| Model | Result | Inference | PyTorch CUDA peak | Peak board power | Peak temperature |
| --- | --- | ---: | ---: | ---: | ---: |
| SmolVLM2-256M | Passed | 1.865 s | 533.1 MiB | 5.92 W | 48.41 C |
| SmolVLM2-500M | Passed | 1.830 s | 1025.0 MiB | 6.70 W | 48.25 C |

The PyTorch CUDA peak is only memory tracked by PyTorch's CUDA allocator. Jetson CPU and
GPU allocations share physical RAM, so it must not be interpreted as total system memory
required by a model.

Two consecutive YOLO family runs produced the following ranges. Variation between these
single-image runs is expected because they are functionality checks rather than repeated
performance trials.

| Model | Result | Inference range | PyTorch CUDA peak |
| --- | --- | ---: | ---: |
| YOLOv8n | Passed | 34.7-35.7 ms | 12.1 MiB |
| YOLO11n | Passed | 38.7-58.3 ms | 40.0 MiB |
| YOLO26n | Passed | 45.3-46.2 ms | 41.4 MiB |

The following larger-model outcomes are safety findings, including runs that could not
produce a JSON result:

| Model | Observed outcome | Current conclusion |
| --- | --- | --- |
| SmolVLM2-2.2B | Reached about 6.9/7.6 GB RAM and 2.5 GB swap, then failed near 85% of weight loading | Not yet validated; retry only by itself after reducing baseline memory and improving the loading path |
| Qwen2.5-VL-3B | Froze the device and caused an automatic restart | Unsafe with the current FP16 loader on the 8 GB device; do not retry |
| Phi-3.5 Vision | Not reached because Qwen restarted the device | Treat as unsafe in the same FP16 test sequence; do not run on this configuration |

L4T 36.4.7 also intermittently printed `NvMapMemAllocInternalTagged ... error 12` during
successful YOLO and VLM runs. Because inference sometimes continued, these messages did
not indicate the model's steady-state footprint. They are consistent with the known NvMap
allocation issue in this L4T release and make an upgrade part of any future large-model
retest.

The 2.2B result is not yet a universal hard limit. Freeing the roughly 1.7 GB desktop
baseline and upgrading to an L4T release containing NVIDIA's NvMap fix may make it possible.
Retry it individually rather than through `small-vlm` or `all`:

```bash
./scripts/smoke_test_models.sh smolvlm2-2.2b
```

The published SmolVLM2-2.2B checkpoint contains about 8.99 GB of FP32 tensors. The loader
requests FP16 and already uses `low_cpu_mem_usage=True`, producing approximately 4.5 GB of
FP16 parameters, but the on-load conversion and final runtime allocations still share the
same physical RAM as Linux. Before retrying, the safer optimization to investigate is a
separately preconverted local FP16 checkpoint, which avoids converting the 8.99 GB FP32
checkpoint during every load. This requires about 4.5 GB of additional storage and must be
validated separately before it becomes the default download path.

Quantization or a TensorRT engine is the appropriate next experiment for Qwen2.5-VL-3B and
Phi-3.5 Vision, but do not retry their current FP16 smoke tests on the 8 GB device. Results
from another precision or backend must be reported separately rather than compared directly
with FP16.

The shell entry point uses `uv run --no-sync` intentionally. Smoke tests must preserve
the validated JetPack-compatible Torch build instead of synchronizing generic PyPI
Torch from the cross-platform lockfile.
