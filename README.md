# Retina-Care AI: Diabetic Retinopathy Severity Grader

Streamlit app that grades diabetic retinopathy (0-4) from a retinal fundus photo, using EfficientNet-B0 (and optionally a ResNet-50 ensemble) trained on APTOS 2019 at 160 px, with Grad-CAM / LIME / SHAP explanations and a PDF report.

> **Not a medical device.** Research and education only. Not for diagnosis or clinical decisions.

## Status

- Every number below is produced by a script in `scripts/` and stored in `results/`. Nothing is hard-coded.
- The earlier version of this project displayed a quadratic weighted kappa of 0.8042. **That value was not reproduced.** The measured EfficientNet-B0 validation QWK is in the table below, and it is lower.
- The Evaluation page reads `results/` only. If the files are missing it shows no numbers.
- Evaluation is on the validation split, which was also used for early stopping and temperature fitting. There is no separate held-out test set yet.
- Measured with newer library versions than `requirements.txt` pins (see "How to reproduce"). Whether this explains any gap to the old number is not tested.

## Results

<!-- RESULTS:START -->
### Validation-split performance (n = 733)

| Variant | QWK | 95% CI (bootstrap, 1000 resamples) | Accuracy |
|---|---|---|---|
| efficientnet_b0 | 0.7473 | 0.7040 to 0.7868 | 61.8% |
| resnet50 | 0.7580 | 0.7199 to 0.7964 | 55.3% |
| ensemble | 0.7765 | 0.7404 to 0.8134 | 60.4% |
| efficientnet_b0+tta | 0.7809 | 0.7459 to 0.8155 | 61.5% |
| ensemble+tta | 0.7897 | 0.7532 to 0.8232 | 62.1% |

Measured 2026-10-18 at commit `73c0e75a`. Default model (`config.yaml`): `efficientnet_b0`. Per-grade metrics and confusion matrices: `results/val_metrics.json`, `results/confusion_matrix*.png`. Error breakdown: [results/ERROR_ANALYSIS.md](results/ERROR_ANALYSIS.md).

### Calibration (15-bin ECE, NLL)

| Model | ECE before | ECE after (cross-fitted T) | NLL before | NLL after (cross-fitted T) | ECE with shipped T (in-sample) |
|---|---|---|---|---|---|
| efficientnet_b0 | 0.1005 | 0.0919 | 1.0394 | 0.9813 | 0.0902 |
| resnet50 | 0.1039 | 0.1006 | 1.1680 | 1.1485 | 0.1220 |
| ensemble | 0.0821 | 0.1049 | 0.9862 | 0.9928 | 0.1037 |

T is fitted on one half of the validation split and evaluated on the other half (2-fold cross-fitting, seed 0). Cross-fitted temperatures: efficientnet_b0 1.481 / 1.527, resnet50 1.234 / 1.369. Shipped temperatures (`model/*_temp.pt`): efficientnet_b0 1.5000, resnet50 1.4035. Plot: `results/reliability_diagrams.png`.

### Robustness (QWK under perturbation, validation split, no retraining)

| Perturbation | Severity | Parameter | EfficientNet-B0 QWK | Ensemble QWK |
|---|---|---|---|---|
| clean | 0 | none | 0.7488 | 0.7862 |
| blur | 1 | sigma=0.5% of width | 0.7165 | 0.7677 |
| blur | 2 | sigma=1% of width | 0.2237 | 0.3257 |
| blur | 3 | sigma=2% of width | 0.0362 | 0.0461 |
| dark | 1 | x0.7 | 0.7480 | 0.7708 |
| dark | 2 | x0.5 | 0.7230 | 0.7658 |
| dark | 3 | x0.3 | 0.6495 | 0.7072 |
| bright | 1 | x1.3 | 0.7298 | 0.7866 |
| bright | 2 | x1.6 | 0.6924 | 0.7556 |
| bright | 3 | x2.0 | 0.5968 | 0.7126 |
| jpeg | 1 | quality=80 | 0.7741 | 0.7921 |
| jpeg | 2 | quality=50 | 0.7467 | 0.7793 |
| jpeg | 3 | quality=20 | 0.3631 | 0.4125 |
| noise | 1 | sigma=5 | 0.7209 | 0.7575 |
| noise | 2 | sigma=15 | 0.0812 | 0.1385 |
| noise | 3 | sigma=30 | 0.0780 | 0.1123 |

Worst case per model: efficientnet_b0 0.0362 (blur, sigma=2% of width); ensemble 0.0461 (blur, sigma=2% of width). Perturbations are applied to raw images downscaled to a 640 px long side (see `scripts/robustness.py`), so the clean row can differ slightly from the table above. Plot: `results/robustness.png`.
<!-- RESULTS:END -->

## Method

- **Data:** APTOS 2019 Blindness Detection training set (3,662 images, grades 0-4). Stratified 80/20 split with seed 42 (`train.py`): 2,929 training and 733 validation images (`data/_train_split.csv`, `data/_val_split.csv`). The 733-image split is called the *validation split* throughout; it is not a test set.
- **Preprocessing:** crop to the retinal circle, resize to 160 x 160, Ben Graham local-contrast enhancement (`utils/preprocess.py`), ImageNet normalisation.
- **Models:** EfficientNet-B0 and ResNet-50 backbones (timm) with a small classification head (Dropout, Linear 256, ReLU, Dropout, Linear 5).
- **Loss and sampling:** focal loss (gamma 1.5) with per-class weights, plus a weighted random sampler for class imbalance.
- **Training:** AdamW, cosine annealing with warm restarts, early stopping on validation QWK, CPU-friendly (160 px, batch size 8). Note that `train.py` hard-codes several hyperparameters, so `config.yaml` does not fully describe the training run.
- **Calibration:** one temperature per model (`model/*_temp.pt`), applied before softmax.
- **Inference options in the app:** single EfficientNet-B0 (default), a 0.6 / 0.4 soft-voting ensemble with ResNet-50 (the weights are hand-set, not tuned), and 5-view test-time augmentation (identity, two flips, two rotations).

## Calibration

See the calibration table above. ECE uses 15 equal-width bins. The temperature is never evaluated on the images it was fitted on: the reported "after" numbers use 2-fold cross-fitting. The shipped temperatures were fitted on the whole validation split, so the column marked in-sample is optimistic.

## Robustness

See the robustness table above. The perturbations are applied to raw images, and the models are not retrained or adapted. Perturbation strengths are defined in `scripts/robustness.py`.

## Limitations

- Single dataset (APTOS 2019); no external validation on other cameras, clinics or populations.
- One train/validation split with one seed; no cross-validation across splits.
- The validation split was reused for early stopping and temperature fitting, so the reported numbers are optimistic relative to a clean test set.
- 160 px images: small lesions such as microaneurysms may not be resolvable at this size.
- The ensemble weights (0.6 / 0.4) are hand-set.
- The QWK confidence interval reflects image-level sampling only, not training variance.
- The image-quality indicators on the Analysis page are heuristic, and several of them show a fixed status; they are not validated measurements.
- Not a medical device and not clinically validated.

## How to reproduce

Needs the APTOS 2019 images at `data/train_images/` (download from Kaggle; not in the repo) and the committed checkpoints in `model/`.

```bash
pip install -r requirements.txt pytest
python scripts/evaluate.py           # results/val_metrics.json, val_predictions.csv, confusion matrices
python scripts/calibration_report.py # results/calibration.json, reliability_diagrams.png
python scripts/robustness.py         # results/robustness.csv, robustness.png (about 15 min on CPU)
python scripts/error_analysis.py     # results/ERROR_ANALYSIS.md
python scripts/build_readme_tables.py  # refreshes the Results section of this README
pytest tests
streamlit run app.py
```

The results in this repo were produced with Python 3.10, torch 2.14.1 (CPU), timm 0.9.12, albumentations 1.3.1, OpenCV 5.0.0 and numpy 1.26.4 (full list in `results/val_metrics.json`). `requirements.txt` pins older versions (e.g. torch 2.1.0, opencv-python-headless 4.8.1.78) that were not used for these measurements.

To retrain: `python train.py --data_dir data --model efficientnet_b0` (and `--model resnet50`), then `python -m calibration.temperature_scaling --model efficientnet_b0`. Retraining overwrites `model/*_dr.pth`.

### Demo mode

`DR_DEMO_MOCK=1 streamlit run app.py` returns random predictions for UI demos when no weights are present. A red banner says so. Without the variable, missing weights raise an error.

## Project structure

```text
app.py                  landing page
pages/                  1_Analysis (inference + XAI + PDF), 2_Evaluation (reads results/), 3_About
model/                  architecture, predict, ensemble, checkpoints, temperature files
utils/                  preprocessing, dataset, losses, metrics, report, UI
calibration/            temperature scaling
explainability/         Grad-CAM, LIME, SHAP
scripts/                evaluate, calibration_report, robustness, error_analysis, build_readme_tables
results/                measured outputs
tests/                  pytest tests (metrics, preprocessing)
data/                   split CSVs (images are not in the repo)
```

## Dataset

[APTOS 2019 Blindness Detection](https://www.kaggle.com/competitions/aptos2019-blindness-detection/data) (Kaggle competition data; check the competition rules for your use).
