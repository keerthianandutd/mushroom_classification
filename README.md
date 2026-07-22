# Mushroom Classification — ML Projects

Two machine learning projects built around mushroom data: a tabular classifier
that predicts edibility, and an image classifier that identifies mushroom
genus from photos.

## Structure

```
.
├── mushroom-edibility-classifier/       Edible vs. poisonous classifier (tabular data)
│   └── README.md                        Full details: approach, model comparison, usage
│
├── mushroom-genus-image-classifier/     Mushroom genus classifier (images, MobileNetV2)
│   └── README.md                        Full details: approach, training results, usage
│
└── data/
    ├── mushroom_mixed_50000.csv         Tabular training data
    ├── mushroom_mixed_test.csv          Tabular holdout/test data
    └── mushroom_images/                 Labeled mushroom photos
        ├── Agaricus/
        └── Amanita/
```

Each project directory has its own README with full details on approach,
models, results, and usage — see the links above.

## Large files

Model weights and the image dataset are tracked with **Git LFS**
(`*.pkl`, `*.keras`, `*.jpg`, `*.jpeg`, `*.png`, `*.pdf`, `*.docx`). Make sure
you have [Git LFS](https://git-lfs.com) installed before cloning:

```bash
git lfs install
git clone <this-repo-url>
```
