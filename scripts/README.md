# Scripts

Run all scripts from the repository root.

## Dataset downloads

Download COCO 2017 validation images and annotations:

```bash
./scripts/download_datasets.sh coco
```

Download the labeled Imagenette-160 validation split:

```bash
./scripts/download_datasets.sh imagenette
```

Download both datasets:

```bash
./scripts/download_datasets.sh all
```

## Model downloads

Download individual models, a model family, or all configured models:

```bash
./scripts/download_models.sh smolvlm2-500m yolov8n
./scripts/download_models.sh small-vlm
./scripts/download_models.sh yolo
./scripts/download_models.sh all
```

Prepare the downloaded SmolVLM2-2.2B FP32 checkpoint as persistent FP16 without deleting
the source files:

```bash
./scripts/prepare_fp16_model.sh smolvlm2-2.2b
```

## Model smoke tests

Run single-image CUDA inference for one validated model or the YOLO family:

```bash
./scripts/smoke_test_models.sh smolvlm2-256m
./scripts/smoke_test_models.sh yolo11n
./scripts/smoke_test_models.sh yolo
```

Do not use the `small-vlm` or `all` selectors on the 8 GB Orin Nano. They include
Qwen2.5-VL-3B and Phi-3.5 Vision; Qwen exhausted unified memory and restarted the device,
and Phi was intentionally not attempted after that capacity boundary was established.

Run native FP32 or optimized FP16 for SmolVLM2-256M and SmolVLM2-500M:

```bash
./scripts/smoke_test_models.sh --precision fp32 smolvlm2-256m smolvlm2-500m
./scripts/smoke_test_models.sh --precision fp16 smolvlm2-256m smolvlm2-500m
```

Models run sequentially, and failures do not stop later selections. Results are written
to `results/smoke/`. The script exits nonzero if any selected model fails.

## Performance benchmarks

Run a short development benchmark before processing a complete dataset:

```bash
./scripts/run_benchmark.sh yolo11n coco --limit 10
./scripts/run_benchmark.sh smolvlm2-256m imagenette --precision fp16 --limit 10
```

Run the four complete experiment groups through the same single-model benchmark path:

```bash
./scripts/run_benchmark_group.sh yolo coco
./scripts/run_benchmark_group.sh yolo imagenette
./scripts/run_benchmark_group.sh smolvlm coco
./scripts/run_benchmark_group.sh smolvlm imagenette
```

Add `--limit` for validation runs. SmolVLM groups require a headless session because they
include SmolVLM2-2.2B FP16 and are refused while the desktop is active.

Omit `--limit` for the full dataset. VLMs require `--precision fp16` or
`--precision fp32`; YOLO uses FP16 and rejects the precision option. SmolVLM2-2.2B
supports FP16 only and should run alone in a headless session. Qwen2.5-VL-3B and
Phi-3.5 Vision are blocked for safety.

Reports are atomically checkpointed under `results/benchmarks/`. Use `--warmup` to change
the default three excluded warm-ups and `--output` to select an exact JSON path.
