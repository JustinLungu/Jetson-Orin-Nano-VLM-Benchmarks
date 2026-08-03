# COCO 2017 validation dataset

Download and prepare the validation images and annotations from the repository root:

```bash
./scripts/download_datasets.sh coco
```

The resulting layout is:

```text
datasets/coco/
├── README.md
├── images/
│   └── 000000000139.jpg
└── annotations/
    ├── instances_val2017.json
    ├── captions_val2017.json
    └── person_keypoints_val2017.json
```

COCO validation contains 5,000 images. Dataset contents are ignored by Git.

Source: <https://cocodataset.org/#download>
