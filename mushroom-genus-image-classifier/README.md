# Mushroom Genus Image Classifier

Classifies a photo of a mushroom into its **genus** using a fine-tuned
MobileNetV2. Unlike the tabular edibility classifier, this project works
directly from images — no manual attribute measurements needed.

## Classes

Defined in `class_indices.json`:

```
Agaricus, Amanita, Boletus, Cortinarius, Entoloma,
Hygrocybe, Lactarius, Russula, Suillus
```

> **Note:** the `../data/mushroom_images/` folder in this repo currently
> only contains photos for **Agaricus** and **Amanita**. The other 7 genera
> the model was trained on aren't included in this snapshot of the dataset.

## Approach

- **Base model** — `MobileNetV2` pretrained on ImageNet
  (`include_top=False`, 224×224×3 input), initially frozen for feature
  extraction with a custom classification head on top.
- **Data augmentation** — `ImageDataGenerator` with rescaling, rotation
  (±30°), width/height shift, shear, zoom, and horizontal flip; 80/20
  train/validation split.
- **Training** — two stages:
  1. **Feature extraction** (base frozen, `rmsprop` optimizer): validation
     accuracy climbed from ~57% to ~65% over 10 epochs.
  2. **Fine-tuning** (base unfrozen, `rmsprop` at a lower learning rate of
     `1e-5`): validation accuracy improved further, reaching **~69.6%** by
     the final epoch.
  - `ReduceLROnPlateau`, `EarlyStopping`, and `ModelCheckpoint` (on
    `val_accuracy`) were used throughout.
- Two checkpoints are included: `mobilenet_v2.keras` (best checkpoint during
  training) and `mobilenet_v2_final.keras` (end-of-training weights).

## Files

| File | Description |
|---|---|
| `Project2_Training.ipynb` | Full training notebook (data augmentation, MobileNetV2 setup, two-stage training, accuracy/loss plots) |
| `proj2_test.py` | Standalone script to evaluate the model on new images |
| `mobilenet_v2.keras` | Best checkpoint (highest validation accuracy during training) |
| `mobilenet_v2_final.keras` | Final model weights after fine-tuning |
| `class_indices.json` | Maps class name → model output index |
| `kxa240000_ML_Project2.pdf` | Written report |

## Usage

```bash
cd mushroom-genus-image-classifier
python proj2_test.py
```

Requires: `tensorflow`/`keras`, `numpy`, `Pillow`.
