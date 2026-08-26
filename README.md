# Jetson Orin Nano VLM Benchmarks

This repository measures the runtime performance of compact vision-language models and
YOLO detectors on an 8 GB NVIDIA Jetson Orin Nano. The experiment focuses on latency,
throughput, model-load time, unified RAM, PyTorch CUDA allocation, power, and temperature.
Prediction accuracy is intentionally outside its current scope.

## Experiment

The benchmark asks how model size, precision, and image dataset affect inference on a
normally operated edge device. The tested Jetson keeps its graphical desktop active; this
is the deployment scenario for this experiment, not an attempt to report the maximum
performance of an otherwise idle board. A headless user can repeat the same commands and
publish that environment as a separate comparison.

Every configuration uses batch size 1 and loads its model once. Three warm-up inferences
are excluded, followed by one synchronized latency measurement per image. Model loading
is timed separately. Labels are never read for evaluation, so both model families can
consume either image dataset:

| Family | Models | Dataset | Precision |
| --- | --- | --- | --- |
| Object detection | YOLOv8n, YOLO11n, YOLO26n | COCO or Imagenette validation | FP16 |
| Vision-language | SmolVLM2-256M | COCO or Imagenette validation | FP16, FP32 |
| Vision-language | SmolVLM2-500M | COCO or Imagenette validation | FP16, FP32 |
| Vision-language | SmolVLM2-2.2B | COCO or Imagenette validation | FP16 |

YOLO receives a fixed square `640x640` input and uses normal Ultralytics preprocessing
and postprocessing. SmolVLM uses its model-native image processor, the fixed prompt
`Describe this image briefly.`, deterministic decoding, and 16 generated tokens.

Timing scopes differ by family:

- YOLO latency measures `predict()`, including its internal preprocessing and
  postprocessing.
- SmolVLM latency measures `model.generate()` after processor input preparation.

The values are valid within each family and configuration. YOLO and SmolVLM latency must
not be ranked directly because their work and timing boundaries differ.

## Tested Jetson setup

The current results use:

| Control | Value |
| --- | --- |
| Device | NVIDIA Jetson Orin Nano, 8 GB unified memory |
| Operating system | Ubuntu 22.04 |
| JetPack / L4T | JetPack 6.2.2 / L4T 36.5.2 |
| Kernel | `5.15.199-tegra` |
| CUDA | 12.6 |
| Python | 3.10.12 |
| PyTorch | 2.8.0 Jetson build |
| Power mode | 25 W |
| Clocks | Dynamic, not locked |
| Desktop | Active |
| Batch size | 1 |

The active desktop and dynamic clocks introduce normal real-world variation. They are
part of this documented experiment. Results produced headless, with locked clocks, or
under another power mode describe a different setup and should be reported separately.

L4T 36.4.7 was used in early testing but intermittently produced NvMap allocation errors
and prevented reliable placement of larger models. The results below use L4T 36.5.2.

## Current findings

### Limited benchmark validation

Before full-dataset execution, all four benchmark groups were validated on deterministic
subsets: 20 images per YOLO configuration and 10 images per SmolVLM configuration. These
are pipeline-validation results, not final dataset-wide performance claims.

All 16 supported model/dataset/precision configurations produced completed reports. YOLO
no longer exhibited the earlier 250-400 ms changing-shape initialization spikes after
switching to a fixed `640x640` input.

YOLO median latency:

| Model | COCO | Imagenette |
| --- | ---: | ---: |
| YOLOv8n | 43.4 ms | 38.2 ms |
| YOLO11n | 50.2 ms | 45.4 ms |
| YOLO26n | 52.1 ms | 50.1 ms |

SmolVLM median generation latency:

| Model | Precision | COCO | Imagenette | PyTorch CUDA peak |
| --- | --- | ---: | ---: | ---: |
| SmolVLM2-256M | FP16 | 1.836 s | 1.820 s | 533.4 MiB |
| SmolVLM2-256M | FP32 | 2.106 s | 2.139 s | 1059.1 MiB |
| SmolVLM2-500M | FP16 | 1.889 s | 1.880 s | 1026.1 MiB |
| SmolVLM2-500M | FP32 | 2.219 s | 2.244 s | 2013.7 MiB |
| SmolVLM2-2.2B | FP16 | 1.491 s | 1.424 s | about 4335 MiB |

For 256M and 500M, FP16 reduced median latency by approximately 13-16% and roughly halved
PyTorch-tracked CUDA allocation relative to FP32. Peak board power across the validation
runs was about 13.7 W, and peak temperature remained around 53 C.

The 2.2B result was faster than the smaller variants for this fixed 16-token generation
workload on both datasets. This repeated observation is worth retaining, but the limited
runs do not yet establish general model throughput. Full-dataset repetitions are needed
before drawing a broader performance conclusion.

Model-load times varied substantially with filesystem cache, swap, and starting system
memory, so the limited runs validate that the field is collected but are not a controlled
load-time comparison.

### Capacity boundary

SmolVLM2-2.2B FP16 is conditionally supported. It completed on both datasets, but another
attempt failed under greater memory pressure. Successful runs used approximately 4.33 GiB
of PyTorch-tracked CUDA allocation and approached the board's unified-memory limit. Its
reliability depends on the desktop workload, starting RAM, swap, and memory fragmentation.

SmolVLM2-2.2B FP32 is blocked rather than merely untested. Its published FP32 tensors total
8,987,139,520 bytes (about 8.37 GiB), exceeding the memory visible to Linux before CUDA,
activations, inputs, and the operating system are considered.

Qwen2.5-VL-3B FP16 restarted the board during weight loading on both L4T 36.4.7 and
36.5.2. Phi-3.5 Vision was not attempted after that capacity boundary was established.
Both are excluded from grouped benchmarks. Quantization or another runtime backend is a
separate future experiment.

PyTorch CUDA allocation is not total Jetson memory. CPU and GPU share physical RAM, so
the report keeps CUDA allocation, unified RAM, and swap as separate measurements.

## Reproducing the experiment

### 1. Install the project environment

The project uses the JetPack-provided Python 3.10 and
[uv](https://docs.astral.sh/uv/) for dependency management. Install `uv` on Ubuntu:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source "$HOME/.local/bin/env"
uv --version
```

If that environment file is absent, start a new shell or run:

```bash
export PATH="$HOME/.local/bin:$PATH"
```

From the repository root, create `.venv` from the committed lockfile:

```bash
uv sync --locked
source .venv/bin/activate
```

Do not ask `uv` to download another Python version.

### 2. Download the datasets

```bash
./scripts/download_datasets.sh coco
./scripts/download_datasets.sh imagenette
```

COCO provides 5,000 validation images. Imagenette provides 3,925 validation images from
ten ImageNet classes. Annotations are downloaded as part of the dataset layouts but are
not used to calculate accuracy.

### 3. Download the supported models

```bash
./scripts/download_models.sh yolo
./scripts/download_models.sh \
  smolvlm2-256m smolvlm2-500m smolvlm2-2.2b
```

Model snapshots remain under the ignored `models/` directory. Hugging Face snapshots are
pinned to their resolved commit in `download_metadata.json`. Alternative ONNX, GGUF, and
duplicate framework checkpoints are excluded from the download.

Prepare the persistent 2.2B FP16 checkpoint once:

```bash
./scripts/prepare_fp16_model.sh smolvlm2-2.2b
```

The conversion should record `target_dtype: float16`, `tensor_count: 657`, and
`total_tensor_bytes: 4493569760` in `fp16/conversion_metadata.json`. It leaves the FP32
source untouched, and the loader automatically selects the validated FP16 copy.

### 4. Install the Jetson inference dependencies

Verify the platform before installing the CUDA packages:

```bash
cat /etc/nv_tegra_release
python3 --version
/usr/local/cuda/bin/nvcc --version
```

Install the JetPack 6.2 CUDA 12.6 builds of PyTorch and TorchVision into `.venv`:

```bash
sudo apt install -y libopenblas-dev
uv pip install --reinstall \
  torch==2.8.0 torchvision==0.23.0 \
  --index-url https://pypi.jetson-ai-lab.io/jp6/cu126/+simple
```

Do not replace these with generic PyPI wheels. Then verify CUDA:

```bash
./scripts/check_jetson_gpu.sh
```

Continue only when `CUDA available` is `True`. Preserve the manually installed Jetson
Torch build while installing the remaining inference packages:

```bash
uv sync --locked --group inference --inexact
uv pip install ultralytics==8.4.115
./scripts/check_jetson_gpu.sh
```

Do not sync the `yolo` group on the Jetson because its generic resolved Torch packages
can replace the JetPack-compatible builds. The shell entry points use
`uv run --frozen --no-sync` for the same reason.

### 5. Run smoke tests

Smoke tests answer only whether one downloaded model can complete CUDA inference:

```bash
./scripts/smoke_test_models.sh yolo

./scripts/smoke_test_models.sh --precision fp16 \
  smolvlm2-256m smolvlm2-500m

./scripts/smoke_test_models.sh --precision fp32 \
  smolvlm2-256m smolvlm2-500m

./scripts/smoke_test_models.sh smolvlm2-2.2b
```

> [!WARNING]
> Do not run the `small-vlm` or `all` smoke selectors on the 8 GB board. They include
> Qwen2.5-VL-3B and Phi-3.5 Vision. Use only the validated selectors above.

Smoke reports are written under `results/smoke/`.

### 6. Validate the benchmark pipeline

Run all four groups on small deterministic subsets:

```bash
./scripts/run_benchmark_group.sh yolo coco --limit 20
./scripts/run_benchmark_group.sh yolo imagenette --limit 20
./scripts/run_benchmark_group.sh smolvlm coco --limit 10
./scripts/run_benchmark_group.sh smolvlm imagenette --limit 10
```

Each SmolVLM group includes 256M FP16/FP32, 500M FP16/FP32, and 2.2B FP16. Limited reports
remain directly under `results/benchmarks/`.

### 7. Run the complete datasets

Omit `--limit` to execute the full experiment:

```bash
./scripts/run_benchmark_group.sh yolo coco
./scripts/run_benchmark_group.sh yolo imagenette
./scripts/run_benchmark_group.sh smolvlm coco
./scripts/run_benchmark_group.sh smolvlm imagenette
```

These four groups produce 16 independent reports: six YOLO reports and ten SmolVLM
reports. Full-dataset output is stored separately under
`results/benchmarks/full_run/`. A group stops when one configuration fails, while the
completed reports already written by earlier configurations remain valid.

## Reading benchmark reports

Each JSON file identifies the model, dataset, precision, software versions,
desktop state, fixed warm-up count, and whether the run was limited or full. It retains
one synchronized latency and status per image, plus compact summaries:

- mean, median, and nearest-rank p95 inference latency
- model-load and total run time
- images per second and, for VLMs, generated tokens per second
- RAM before and after model loading, peak RAM, and peak swap
- peak PyTorch CUDA allocation
- average and peak board power and peak temperature

Reports are atomically checkpointed after every image. `run_status` is `running`,
`completed`, `interrupted`, or `failed`; interrupted and failed reports retain one concise
error message. Generated results are device-specific and ignored by Git.
