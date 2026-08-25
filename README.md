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

## Jetson inference precision

Runtime precision is explicit. SmolVLM2-256M and SmolVLM2-500M support their published
FP32 precision and an optimized FP16 mode. SmolVLM2-2.2B is FP16-only on the 8 GB device
because its FP32 tensors alone exceed visible system memory:

```python
from src.load_models import load_vlm

model, processor = load_vlm("smolvlm2-500m", precision="fp32")
```

The loader uses local files only, validates every resulting parameter against the requested
precision, and rejects unsafe combinations such as 2.2B FP32 before loading weights. Reports
record `runtime_precision` so native and optimized measurements cannot be confused.

SmolVLM2-2.2B FP32 is infeasible by a parameter-size lower bound, not simply untested. Its
657 published FP32 tensors total 8,987,139,520 bytes (approximately 8.37 GiB), while Linux
can see only about 7.44 GiB of unified memory on the 8 GB Orin Nano. The weights therefore
cannot fit even in a hypothetical headless run using no memory for the operating system.
CUDA context, Transformers, inputs, activations, and generation buffers increase the actual
requirement further. Swap can assist CPU processes but cannot provide normal CUDA device
allocations, so attempting full-GPU FP32 would only produce an OOM or risk another restart.
The loader blocks this configuration and records FP16 as the supported 2.2B precision.

### Jetson prerequisites

This setup is currently validated against JetPack 6.2.2 / L4T 36.5.2, Ubuntu 22.04,
Python 3.10, and CUDA 12.6. L4T 36.4.7 was used during the initial experiments but its
NvMap allocation bug prevented reliable large-model CUDA placement. Check another Jetson
before installing inference packages:

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
> Do not use `small-vlm` or `all` on the 8 GB Orin Nano. Qwen2.5-VL-3B FP16
> restarted the device during weight loading on both L4T 36.4.7 and 36.5.2.
> Phi-3.5 Vision was intentionally not attempted because it is larger and the established
> capacity limit makes another restart predictable. Run only validated models individually.

```bash
./scripts/smoke_test_models.sh smolvlm2-256m
./scripts/smoke_test_models.sh smolvlm2-500m
./scripts/smoke_test_models.sh yolo11n
./scripts/smoke_test_models.sh yolo
```

Compare native FP32 and optimized FP16 for the two models that fit in either mode:

```bash
./scripts/smoke_test_models.sh --precision fp32 \
  smolvlm2-256m smolvlm2-500m

./scripts/smoke_test_models.sh --precision fp16 \
  smolvlm2-256m smolvlm2-500m
```

The default remains `--precision fp16` for backward compatibility. FP32 is intentionally
blocked for SmolVLM2-2.2B, Qwen2.5-VL-3B, and Phi-3.5 Vision until a safe native-precision
policy is implemented for those models.

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
versions, runtime precision, load and inference times, peak CUDA memory, Jetson
CPU/GPU/power/temperature summaries, prediction summaries, and VLM generated-token counts.
The report is created
before inference begins and atomically updated after every model, so completed results
survive a later model crashing or rebooting the Jetson. `selected_models` records the
intended run and `run_completed` remains `false` until every selected model finishes.
Reports are device-specific and ignored by Git.

### Observed results on the 8 GB Orin Nano

These observations were collected on 2026-08-25 with JetPack 6.2.2 / L4T 36.5.2,
CUDA 12.6, PyTorch 2.8.0, and the graphical desktop running. Each run used the same
synthetic image and 16-token generation limit. The FP16 models ran sequentially in one
process, followed by a separate sequential FP32 run of the two configurations that can
fit safely. The 2.2B checkpoint was persistently converted to FP16. These are smoke-test
functionality measurements, not final full-dataset benchmark results.

| Model | Precision | Result | Load time | Measured generation | PyTorch CUDA peak | Peak board power | Peak temperature |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| SmolVLM2-256M | FP16 | Passed | 22.810 s | 1.862 s | 533.1 MiB | 5.96 W | 47.88 C |
| SmolVLM2-256M | FP32 | Passed | 23.719 s | 2.114 s | 1059.1 MiB | 9.00 W | 49.19 C |
| SmolVLM2-500M | FP16 | Passed | 27.463 s | 1.878 s | 1026.1 MiB | 6.70 W | 47.81 C |
| SmolVLM2-500M | FP32 | Passed | 27.424 s | 2.216 s | 2013.7 MiB | 10.24 W | 49.41 C |
| SmolVLM2-2.2B | FP16 | Passed | 68.944 s | 1.449 s | 4336.5 MiB | 13.78 W | 49.44 C |

FP32 approximately doubled the PyTorch CUDA peak: 533.1 to 1059.1 MiB for 256M and
1026.1 to 2013.7 MiB for 500M. In these individual measurements, FP16 reduced measured
generation time by about 12% for 256M and 15% for 500M, while also reducing peak board
power. Both smaller models are therefore compatible with FP32, but FP16 is the more
memory- and power-efficient Jetson configuration. FP32 remains useful as a native-
precision reference for later output and numerical-quality comparisons.

The terminal's `inference=` value measures only the second, post-warm-up
`model.generate()` call. It excludes processor creation, checkpoint loading, CUDA
placement, input preparation, the warm-up generation, and cleanup. The Jetson metrics
span the wider smoke-test execution, so their average values are not inference-only
measurements.

The 2.2B model's lower single generation time does not establish that it is faster. It ran
last, reached 13.78 W while the smaller models peaked at or below 6.70 W, and may have benefited
from warmed CUDA state and higher dynamic clocks. Larger matrix operations can also use the
GPU more efficiently while small generations remain sensitive to Python, kernel-launch,
and synchronization overhead. Comparable performance claims require separate fresh
processes, fixed power/clocks, multiple warm-ups, repeated measurements, and median and p95
statistics.

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
| SmolVLM2-2.2B, original FP32 checkpoint on L4T 36.4.7 | Reached about 6.9/7.6 GB RAM and 2.5 GB swap, then failed during loading/CUDA placement | Failed configuration; preserved to show why checkpoint representation and L4T version matter |
| SmolVLM2-2.2B, prepared FP16 checkpoint on L4T 36.5.2 | Loaded all 657 tensors, reached about 7.2/7.6 GB total RAM, and generated all 16 tokens | Conditionally supported; use an individual run and prefer a headless environment for full benchmarks |
| Qwen2.5-VL-3B FP16 on L4T 36.4.7 | Froze the device and caused an automatic restart | Unsupported with the current Transformers FP16 loading path |
| Qwen2.5-VL-3B FP16 on L4T 36.5.2 | Reached about 53% of weight loading and 7.2-7.3/7.6 GB total RAM, then restarted the device; the checkpoint report remained incomplete with no model result | Confirms a unified-memory capacity failure; do not repeat this loading configuration |
| Phi-3.5 Vision FP16 | Intentionally not attempted after the smaller Qwen model exhausted the device | Not validated and excluded from the current supported set; a restart-prone test would add little information |

L4T 36.4.7 also intermittently printed `NvMapMemAllocInternalTagged ... error 12` during
successful YOLO and VLM runs. Because inference sometimes continued, these messages did
not indicate the model's steady-state footprint. They are consistent with the known NvMap
allocation issue in this L4T release and make an upgrade part of any future large-model
retest.

### Reproducing the SmolVLM2-2.2B result

The successful result required both a corrected Jetson software stack and a prepared
checkpoint. The exact environment was JetPack 6.2.2 / L4T 36.5.2, kernel
`5.15.199-tegra`, CUDA 12.6, PyTorch 2.8.0, and NumPy 1.26.4. The source snapshot was
`HuggingFaceTB/SmolVLM2-2.2B-Instruct` revision
`482adb537c021c86670beed01cd58990d01e72e4`.

The published checkpoint contains 657 FP32 tensors totaling 8,987,139,520 bytes. Prepare
the persistent FP16 representation once:

```bash
./scripts/prepare_fp16_model.sh smolvlm2-2.2b
```

The conversion must finish with `target_dtype: float16`, `tensor_count: 657`, and
`total_tensor_bytes: 4493569760` in `fp16/conversion_metadata.json`. The converter retains
the tensor names and model configuration, validates that every output tensor is FP16, and
leaves the original FP32 checkpoint untouched. The loader automatically prefers this
validated copy.

Close memory-heavy applications and run only this selector:

```bash
./scripts/smoke_test_models.sh smolvlm2-2.2b
```

The successful report loaded all 657 tensors, completed 16-token deterministic generation
in 1.481 seconds, recorded a 4334.0 MiB PyTorch CUDA peak, and produced coherent output.
Total Jetson RAM reached approximately 7.2/7.6 GB, so passing the smoke test does not imply
enough headroom for arbitrary prompts, image sizes, token limits, or concurrent processes.

### Headless experiments over SSH

SSH only saves memory when the graphical desktop is stopped; connecting through SSH while
GNOME, the display server, VS Code, and browsers remain active provides little benefit.
From an established SSH session, switch the Jetson to its non-graphical target before a
large-model experiment:

```bash
sudo systemctl isolate multi-user.target
```

Restore the desktop afterward:

```bash
sudo systemctl isolate graphical.target
```

Removing the graphical baseline should give the 2.2B full-dataset benchmark a safer memory
margin. It did not make Qwen2.5-VL-3B viable with the current FP16 Transformers loading
path: the model restarted the device at about 53% of weight loading. The runner's atomic
checkpoint preserved `results/smoke/smoke-20260825T223231Z.json` with
`run_completed: false`, which distinguishes the interrupted experiment from a completed
CUDA OOM result.

Quantization or a TensorRT engine is a separate future research direction for
Qwen2.5-VL-3B and Phi-3.5 Vision, not part of this smoke-test branch. Results from another
precision or backend must be reported separately rather than compared directly with FP16.

The shell entry point uses `uv run --no-sync` intentionally. Smoke tests must preserve
the validated JetPack-compatible Torch build instead of synchronizing generic PyPI
Torch from the cross-platform lockfile.
