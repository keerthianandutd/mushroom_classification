# Mushroom Classification — ML Projects

Two machine learning projects built around mushroom data:

- **[mushroom-edibility-classifier](./mushroom-edibility-classifier)** — predicts
  whether a mushroom is edible or poisonous from 20 physical attributes
  (cap shape, gill traits, stem measurements, habitat, season, etc.) using
  classic tabular models (Decision Tree, Random Forest, k-NN). Best model:
  Random Forest at ~99.98% accuracy.

- **[mushroom-genus-image-classifier](./mushroom-genus-image-classifier)** —
  identifies a mushroom's genus (Agaricus, Amanita, Boletus, Cortinarius,
  Entoloma, Hygrocybe, Lactarius, Russula, Suillus) from a photo using a
  fine-tuned MobileNetV2, reaching ~69.6% validation accuracy.

Training data lives in `data/`: `mushroom_mixed_50000.csv` and
`mushroom_mixed_test.csv` for the tabular model, and `mushroom_images/`
(currently Agaricus and Amanita only) for the image model.

Each project folder has its own README with full details on approach,
models, results, and usage.

## Large files

Model weights and the image dataset are tracked with **Git LFS**
(`*.pkl`, `*.keras`, `*.jpg`, `*.jpeg`, `*.png`, `*.pdf`, `*.docx`). Make sure
you have [Git LFS](https://git-lfs.com) installed before cloning:

```bash
git lfs install
git clone <this-repo-url>
```
