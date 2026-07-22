# Mushroom Classification — ML Projects

Two machine learning projects built around mushroom data: a tabular classifier
that predicts edibility, and an image classifier that identifies mushroom
genus from photos.

## Structure

```
.
├── project1-tabular-classification/   Edible vs. poisonous classifier (tabular data)
│   ├── AttributeInfo.txt              Description of each input feature
│   ├── ModelTraining.ipynb            Training / experimentation notebook
│   ├── proj1_evaluate.py              Script to evaluate a saved model on new data
│   ├── proj1_chosen_model.pkl         Final selected model
│   ├── decision_tree_model.pkl
│   ├── knn_model.pkl
│   ├── random_forest_model.pkl
│   ├── feature_selector.pkl
│   ├── selected_features.pkl
│   ├── label_encoders.pkl
│   ├── kxa240000_ML_Project1.pdf      Write-up
│   └── kxa240000_ML_Project1.docx     Write-up (source)
│
├── project2-image-classification/     Mushroom genus classifier (images, MobileNetV2)
│   ├── Project2_Training.ipynb        Training / experimentation notebook
│   ├── proj2_test.py                  Script to evaluate the model on new images
│   ├── mobilenet_v2.keras
│   ├── mobilenet_v2_final.keras
│   ├── class_indices.json             Class label → index mapping
│   └── kxa240000_ML_Project2.pdf      Write-up
│
└── data/
    ├── mushroom_mixed_50000.csv       Tabular training data (Project 1)
    ├── mushroom_mixed_test.csv        Tabular holdout/test data (Project 1)
    └── mushroom_images/               Labeled mushroom photos (Project 2)
        ├── Agaricus/
        └── Amanita/
```

## Project 1 — Tabular Edibility Classifier

Predicts whether a mushroom is edible (`e`) or poisonous (`p`) from 20
physical attributes (cap shape/color/surface, gill traits, stem measurements,
habitat, season, etc. — see `AttributeInfo.txt` for the full attribute
legend). Several models were trained and compared (decision tree, k-NN,
random forest); the best performer is saved as `proj1_chosen_model.pkl`.

Run evaluation:
```bash
cd project1-tabular-classification
python proj1_evaluate.py --data ../data/mushroom_mixed_test.csv --model proj1_chosen_model.pkl
```

## Project 2 — Image Genus Classifier

Classifies a mushroom photo into a genus using a fine-tuned MobileNetV2.
`class_indices.json` lists the genera the model was trained on: Agaricus,
Amanita, Boletus, Cortinarius, Entoloma, Hygrocybe, Lactarius, Russula,
Suillus.

> **Note:** `data/mushroom_images/` in this repo currently only contains
> images for **Agaricus** and **Amanita** — the other 7 genera referenced in
> `class_indices.json` aren't included in this snapshot of the dataset.

Run evaluation:
```bash
cd project2-image-classification
python proj2_test.py
```

## Large files

Model weights and the image dataset are tracked with **Git LFS**
(`*.pkl`, `*.keras`, `*.jpg`, `*.jpeg`, `*.png`). Make sure you have
[Git LFS](https://git-lfs.com) installed before cloning:

```bash
git lfs install
git clone <this-repo-url>
```
