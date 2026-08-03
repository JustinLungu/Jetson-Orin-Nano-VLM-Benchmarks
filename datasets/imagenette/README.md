# Imagenette validation dataset

Imagenette is a 10-class subset of ImageNet created by fastai. This repository uses the 160-pixel labeled validation split as a compact ImageNet-compatible benchmark dataset.

Download and prepare it from the repository root:

```bash
./scripts/download_datasets.sh imagenette
```

The resulting layout is:

```text
datasets/imagenette/
├── README.md
├── validation/
│   └── images/
│       ├── n01440764/
│       └── ...
└── validation_labels.csv
```

The manifest maps every relative image path to its standard zero-based ImageNet-1K class index:

```csv
image_path,class_id
validation/images/n01440764/example.JPEG,0
```

Imagenette is useful for lightweight development and comparisons, but it is not a replacement for reporting full ImageNet-1K validation accuracy.

Sources:

- <https://github.com/fastai/imagenette>
- <https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz>
