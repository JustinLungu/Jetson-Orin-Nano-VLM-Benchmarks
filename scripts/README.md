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

## Performance benchmarks

Run the four experiment groups:

```bash
./scripts/run_benchmark_group.sh yolo coco
./scripts/run_benchmark_group.sh yolo imagenette
./scripts/run_benchmark_group.sh smolvlm coco
./scripts/run_benchmark_group.sh smolvlm imagenette
```

Add `--limit 1` for a quick inference check or a larger `--limit` for validation runs.
SmolVLM groups include SmolVLM2-2.2B FP16; running headless is recommended for its tighter
memory margin but is not enforced by the command.

Omit `--limit` for the full dataset. The model and precision combinations are fixed by
the group. Qwen2.5-VL-3B and Phi-3.5 Vision are excluded for safety.

Limited reports are atomically checkpointed under `results/benchmarks/`; full-dataset
reports go under `results/benchmarks/full_run/`. Every run uses three excluded warm-ups.
