# Fabric Defect Classification — PyTorch Project

A three-notebook project for classifying fabric defects into four categories (**hole**, **horizontal**, **stain**, **vertical**) using the *Fabric Defects Dataset* from Kaggle.

## Project Overview

This project builds and compares two CNN architectures:

| Model | Architecture | Key Features |
|-------|--------------|--------------|
| **Baseline CNN** | 3-block CNN with heavy FC head | Baseline for comparison |
| **Improved CNN** | 4-block CNN with BatchNorm + Global Average Pooling | Cosine LR schedule, early stopping, ColorJitter augmentation |

Both models are trained on the same fixed, stratified split (70% train / 15% val / 15% test) with strict data leakage prevention controls.

## Quick Start

### 1. Setup

Install dependencies:
```bash
pip install torch torchvision scikit-learn matplotlib seaborn pandas pillow tqdm
```

Download the dataset from Kaggle and place it in the workspace:
```
https://www.kaggle.com/datasets/nexuswho/fabric-defects-dataset
```

After unzipping, your structure should look like:
```
fabric_defects_raw/
    hole/          (images only, masks auto-filtered)
    horizontal/
    vertical/
    stain/
```

### 2. Run Notebooks in Order

Run these notebooks **sequentially** on a clean start:

| Step | Notebook | Purpose |
|------|----------|---------|
| 1️⃣ | `00_Data_Preparation_and_EDA.ipynb` | Creates train/val/test split, runs EDA |
| 2️⃣ | `01_Baseline_CNN.ipynb` | Trains baseline model |
| 3️⃣ | `02_Improved_CNN.ipynb` | Trains improved model, compares with baseline |

⚠️ **Important**: Notebook 00 creates `fabric_split/` which all other notebooks depend on. Re-running it will invalidate saved test metrics.

## Data Leakage Prevention

This project implements strict controls to prevent data leakage:

1. **Mask images filtered out** — Binary segmentation masks are removed before preprocessing. Detected by filename suffix (`_mask`) or folder name (`mask/`, `gt/`, `masks/`).
2. **Single deterministic split** — Fixed seed (42) ensures reproducibility. All notebooks read from the same `fabric_split/` folders.
3. **Stratified split** — Class proportions preserved across train (70%), val (15%), and test (15%).
4. **Train-only augmentation** — Validation and test use deterministic transforms (resize → tensor → normalize). No augmentation, shuffling, or random crops.
5. **Train-set class weights** — Loss function weights computed from training labels only.
6. **Test set evaluated once** — Test set loaded exactly once per notebook at the end, never used for model selection.

No-overlap assertions in Notebook 00 verify these controls.

## Output Files

### Model Weights
```
models/
  baseline_cnn/
    baseline_cnn_weights.pth
    baseline_history.json
    baseline_metrics.json
  improved_cnn/
    improved_cnn_weights.pth
    improved_history.json
    improved_metrics.json
```

### Results & Visualizations
All visualizations and comparison results are saved to `visualizations/` and `results/`:

**Notebook 00 — Data Preparation & EDA:**
- `eda_class_distribution.png` — Class counts (bar + pie charts)
- `eda_image_dimensions.png` — Image size distributions
- `eda_channel_statistics.png` — Per-channel statistics by class
- `eda_sample_images.png` — 5 samples per defect class
- `eda_split_stratification.png` — Train/val/test class proportions

**Notebook 01 — Baseline CNN:**
- `baseline_augmented_batch.png` — Sample augmented images
- `baseline_training_curves.png` — Loss + accuracy over epochs
- `baseline_overfitting_gap.png` — Train-val accuracy gap
- `baseline_confusion_matrix.png` — Confusion matrix (raw + normalized)
- `baseline_per_class_metrics.png` — Precision/recall/F1 by class
- `baseline_roc_curves.png` — One-vs-rest ROC curves
- `baseline_correct.png`, `baseline_wrong.png` — Sample predictions

**Notebook 02 — Improved CNN:**
- `improved_param_count.png` — Parameter count vs baseline
- `improved_training_curves.png` — Loss + accuracy + LR schedule
- `improved_vs_baseline_curves.png` — Side-by-side comparison
- `improved_confusion_matrix.png`, `improved_per_class_metrics.png`
- `improved_roc_curves.png`, `improved_correct.png`, `improved_wrong.png`
