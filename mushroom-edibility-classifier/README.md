# Mushroom Edibility Classifier (Tabular)

Predicts whether a mushroom is **edible (`e`)** or **poisonous (`p`)** from
20 physical attributes — cap shape/color/surface, gill traits, stem
measurements, veil/ring type, habitat, and season. No images involved; this
is a classic tabular classification problem.

## Data

Trained on `../data/mushroom_mixed_50000.csv` (50,000 rows) and evaluated
against `../data/mushroom_mixed_test.csv`. See `AttributeInfo.txt` for the
full attribute legend (what each code like `cap-shape=x` or `habitat=d`
means). Missing values in the raw data are denoted `?`.

## Approach

1. **Preprocessing** — missing-value imputation (`SimpleImputer`), label
   encoding of categorical features (`LabelEncoder`, saved as
   `label_encoders.pkl`), and feature scaling.
2. **Feature selection** — `SelectKBest` with mutual information
   (`feature_selector.pkl`, `selected_features.pkl`) to keep the most
   informative attributes.
3. **Model comparison** — three classifiers were trained and tuned with
   `GridSearchCV` (5-fold cross-validation):

   | Model | Accuracy (pre-tuning) | Accuracy (after GridSearchCV) |
   |---|---|---|
   | Decision Tree | 99.73% | 78.83%* |
   | Random Forest | 99.99% | 99.98% |
   | k-Nearest Neighbors | 99.90% | 99.91% |

   \* The tuned decision tree's grid search favored heavier pruning
   (`ccp_alpha=0.01`, `max_depth=10`), which traded raw accuracy for a
   simpler, less overfit tree.

4. **Final model** — **Random Forest** (`n_estimators=100, max_depth=20,
   max_features='sqrt', min_samples_leaf=2, min_samples_split=10`) was
   selected as the best performer and saved as `proj1_chosen_model.pkl`.

## Files

| File | Description |
|---|---|
| `ModelTraining.ipynb` | Full training/experimentation notebook (preprocessing, feature selection, model comparison, tuning) |
| `proj1_evaluate.py` | Standalone script to load a saved model and evaluate it on new data |
| `proj1_chosen_model.pkl` | Final selected model (Random Forest) |
| `decision_tree_model.pkl` / `knn_model.pkl` / `random_forest_model.pkl` | Individual trained models kept for comparison |
| `feature_selector.pkl` / `selected_features.pkl` | Fitted `SelectKBest` selector and the feature names it kept |
| `label_encoders.pkl` | Fitted `LabelEncoder`s for each categorical column |
| `AttributeInfo.txt` | Legend for every attribute code used in the dataset |
| `kxa240000_ML_Project1.pdf` / `.docx` | Written report |

## Usage

```bash
cd mushroom-edibility-classifier
python proj1_evaluate.py --data ../data/mushroom_mixed_test.csv --model proj1_chosen_model.pkl
```

Requires: `pandas`, `numpy`, `scikit-learn`, `joblib`.
