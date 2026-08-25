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

## Model smoke tests

Run single-image CUDA inference for one model, a family, or all configured models:

```bash
./scripts/smoke_test_models.sh smolvlm2-256m
./scripts/smoke_test_models.sh yolo11n
./scripts/smoke_test_models.sh small-vlm
./scripts/smoke_test_models.sh yolo
./scripts/smoke_test_models.sh all
```

Models run sequentially, and failures do not stop later selections. Results are written
to `results/smoke/`. The script exits nonzero if any selected model fails.
