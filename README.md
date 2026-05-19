# Fabric Defect Classification — Project Files

A four-notebook PyTorch project for classifying fabric defects into four
categories — **hole**, **horizontal**, **stain**, **vertical** — using the
*Fabric Defects Dataset* from Kaggle (`nexuswho/fabric-defects-dataset`).

The project compares three models trained on the same fixed split:

1. **Baseline CNN** — 3-block CNN with a heavy FC head. Floor of the comparison.
2. **Improved CNN** — 4-block CNN with BatchNorm, Global Average Pooling,
   cosine LR schedule, early stopping, and ColorJitter augmentation.
3. **Tuned CNN** — the Improved CNN architecture, with hyperparameters chosen
   by a 216-configuration grid search over learning rate, batch size, dropout,
   optimiser, weight decay, and LR schedule.

## File list

| File | What it does |
|---|---|
| `00_Data_Preparation_and_EDA.ipynb` | One-time, deterministic train/val/test split. Excludes mask images. Runs EDA (class distribution, image dimensions, channel statistics, sample grids). Writes the split to `fabric_split/`. |
| `01_Baseline_CNN.ipynb` | Trains the Baseline CNN. Saves `baseline_cnn_weights.pth`, `baseline_history.json`, `baseline_metrics.json`. |
| `02_Improved_CNN.ipynb` | Trains the Improved CNN. Saves `improved_cnn_weights.pth`, `improved_history.json`, `improved_metrics.json`. Overlays curves against the baseline if available. |
| `03_Hyperparameter_Tuning.ipynb` | Grid search over the Improved CNN, full retraining of the winner, final three-way comparison. Saves `tuned_cnn_weights.pth`, `grid_search_results.json`, `tuned_metrics.json`, `final_comparison.csv`. |

## Run order

The notebooks **must** be run in numerical order on a clean start.

```
00 → 01 → 02 → 03
```

Notebook 00 produces the `fabric_split/` directory that the other three
read from. Re-running notebook 00 after training will invalidate any saved
test metrics because the split changes — only re-run it if you intend to
discard all trained models.

## Setup

1. Download the Kaggle dataset and unzip it next to these notebooks:
   <https://www.kaggle.com/datasets/nexuswho/fabric-defects-dataset>

   After unzipping you should have something like:
   ```
   fabric_defects_raw/
       hole/          (images + maybe a mask/ subfolder, all auto-filtered)
       horizontal/
       vertical/
       stain/
   ```
   The folder names in the raw dataset can vary slightly across Kaggle
   versions — notebook 00 auto-detects common variants. If your folder is
   somewhere else, edit `RAW_DATA_DIR` in cell 2 of notebook 00.

2. Install dependencies (Python 3.10+):
   ```
   pip install torch torchvision scikit-learn matplotlib seaborn pandas pillow tqdm
   ```

3. Run notebook 00 once. It creates `fabric_split/{train,val,test}/{hole,horizontal,stain,vertical}/`.

4. Run notebooks 01, 02, 03 in order.

## How data leakage is prevented

Five controls, applied in this order:

1. **Mask images filtered out** before anything else. The Kaggle dataset
   ships binary segmentation masks alongside each defect image — these would
   train a mask-vs-image classifier rather than a defect classifier if left in.
   Notebook 00 detects masks by filename suffix (`_mask`) and by parent-folder
   name (`mask/`, `gt/`, `masks/`).
2. **Single deterministic split** with a fixed seed (42), written to disk.
   Every training notebook reads from the same `fabric_split/` folders.
3. **Stratified split** so class proportions are preserved across train, val,
   and test (70 / 15 / 15).
4. **Augmentation is train-only.** Val and test transforms are deterministic
   (resize → tensor → normalise). No augmentation, no shuffling, no random
   crops on evaluation data.
5. **Class weights** for the loss function are computed from training labels
   only. Computing them on the full dataset would smuggle val/test class
   frequencies into the training signal.
6. **Validation set selects the model. Test set evaluates it.** The test set
   is loaded exactly once per notebook, at the very end. The hyperparameter
   grid search never sees the test set during its 216 runs — it ranks
   configurations by validation accuracy and the winner is retrained from
   scratch before the single test evaluation.

The explicit no-overlap assertion in notebook 00, section 7, would fail loudly
if any of this were violated.

## Visualisations produced (for the report)

Notebook 00 — EDA:
- `eda_class_distribution.png` — bar + pie chart of class counts
- `eda_image_dimensions.png` — width / height / aspect ratio distributions
- `eda_channel_statistics.png` — per-channel pixel mean and std per class
- `eda_sample_images.png` — 5 samples per defect class
- `eda_split_stratification.png` — class proportions across train/val/test

Notebook 01 — Baseline:
- `baseline_augmented_batch.png` — what augmentation looks like
- `baseline_training_curves.png` — loss + accuracy over epochs
- `baseline_overfitting_gap.png` — train-val accuracy gap diagnostic
- `baseline_confusion_matrix.png` — counts + row-normalised
- `baseline_per_class_metrics.png` — precision/recall/F1 bars
- `baseline_roc_curves.png` — one-vs-rest ROC
- `baseline_correct.png`, `baseline_wrong.png` — sample predictions

Notebook 02 — Improved:
- `improved_param_count.png` — parameter count vs baseline
- `improved_training_curves.png` — loss + accuracy + LR schedule
- `improved_vs_baseline_curves.png` — overlay comparison
- `improved_confusion_matrix.png`, `improved_per_class_metrics.png`,
  `improved_roc_curves.png`, `improved_correct.png`, `improved_wrong.png`

Notebook 03 — Hyperparameter Tuning:
- `hp_sensitivity.png` — bar chart of mean/max accuracy per hyperparameter
- `hp_boxplots.png` — distribution of val accuracy per hyperparameter value
- `hp_lr_optimiser_interaction.png` — heatmap of LR × optimiser interaction
- `hp_dropout_wd_interaction.png` — heatmap of dropout × weight-decay
- `hp_val_acc_histogram.png` — distribution across all 216 runs
- `hp_best_grid_curves.png` — curves of best grid-search run
- `hp_full_retrain_curves.png` — curves of full 25-epoch retraining
- `hp_tuned_confusion_matrix.png`, `hp_tuned_per_class_metrics.png`,
  `hp_tuned_roc_curves.png`
- `final_comparison_bars.png` — the headline three-model bar chart
- `final_comparison.csv` — machine-readable summary

## Files saved between notebooks

- `fabric_split/` — the canonical data split (produced by 00, read by 01/02/03)
- `baseline_metrics.json`, `improved_metrics.json`, `tuned_metrics.json` —
  per-model summary metrics, loaded by later notebooks for cross-comparison
- `baseline_history.json`, `improved_history.json` — full per-epoch training
  history, used for overlay plots
- `grid_search_results.json` — every grid-search run's config + history;
  enables resuming after interruption
- `*.pth` — best model weights for each of the three models
