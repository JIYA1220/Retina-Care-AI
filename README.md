# Diabetic Retinopathy Severity Grader

> An AI-powered retinal image analysis tool with **Grad-CAM + LIME explainability** and **clinical PDF report generation**.

---

## Project Structure

```
retina_grader/
├── app.py                        # Streamlit web app (main entry point)
├── train.py                      # Training script
├── requirements.txt
├── README.md
│
├── model/
│   ├── model.py                  # EfficientNet-B0 architecture
│   ├── predict.py                # Inference pipeline
│   └── efficientnet_dr.pth       # saved weights (created after training)
│
├── explainability/
│   ├── gradcam.py                # Grad-CAM heatmap generation
│   └── lime_explain.py           # LIME superpixel explanations
│
├── utils/
│   ├── preprocess.py             # Ben Graham preprocessing
│   ├── dataset.py                # PyTorch Dataset + WeightedSampler
│   └── report.py                 # PDF report generator
│
└── data/                         # place Kaggle data here
    ├── train.csv
    └── train_images/
        ├── 0a4e1a29ffff.png
        └── ...
```

---

## Dataset

**APTOS 2019 Blindness Detection**
https://www.kaggle.com/competitions/aptos2019-blindness-detection/data

- 3,662 retinal fundus images
- Labels: 0 (No DR) -> 4 (Proliferative DR)
- Collected from Aravind Eye Hospital, India

### Download via Kaggle CLI:
```bash
# Install kaggle CLI
pip install kaggle

# Set up your API key from https://www.kaggle.com/settings
mkdir ~/.kaggle
cp kaggle.json ~/.kaggle/

# Download
kaggle competitions download -c aptos2019-blindness-detection -p data/
cd data && unzip aptos2019-blindness-detection.zip
```

---

## Setup

### 1. Create a virtual environment (recommended)
```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

---

## Training

```bash
python train.py --data_dir data --epochs 15 --batch_size 8
```

| Argument | Default | Description |
|---|---|---|
| `--data_dir` | `data` | Folder containing `train.csv` + `train_images/` |
| `--epochs` | `15` | Number of training epochs |
| `--batch_size` | `8` | Batch size (keep <=8 for CPU) |
| `--lr` | `1e-4` | Learning rate |

Training on CPU for 15 epochs takes ~3-6 hours depending on your machine.
Tip: Start with `--epochs 5` to verify everything works.

After training, weights are saved to `model/efficientnet_dr.pth`.

---

## 🌡️ Confidence Calibration (Optional but Recommended)

Neural networks can be overconfident in their predictions. We use **Temperature Scaling** to calibrate the model's probabilities, ensuring that a 90% confidence score actually corresponds to 90% accuracy.

### Run Calibration:
```bash
python -m calibration.temperature_scaling --weights model/efficientnet_dr.pth --data_dir data
```

- **Output:** Saves `model/temperature.pt`.
- **Impact:** The Streamlit app automatically detects this file and applies it to all future inferences to provide more reliable and "calibrated" probability distributions.

---

## 🚀 Running the App

```bash
streamlit run app.py
```

Open your browser at **http://localhost:8501**

### Features:
- Upload any retinal fundus image (PNG/JPG)
- Get DR severity grade (0-4) with confidence
- Per-class probability bar chart
- Grad-CAM heatmap (what the model looks at)
- LIME superpixel explanation
- Download a clinical-style PDF report

---

## Model Details

| Component | Details |
|---|---|
| **Backbone** | EfficientNet-B0 (timm, ImageNet pretrained) |
| **Classifier** | Dropout -> Linear(1280->256) -> ReLU -> Linear(256->5) |
| **Loss** | Weighted CrossEntropy (handles class imbalance) |
| **Sampling** | WeightedRandomSampler (handles imbalance at batch level) |
| **Preprocessing** | Ben Graham: circle crop + Gaussian subtraction |
| **Augmentation** | Albumentations (flip, rotate, color jitter, noise) |
| **Metric** | Quadratic Weighted Kappa (competition standard) |

---

## Explainability

### Grad-CAM
Hooks into the last convolutional block of EfficientNet-B0.
Computes gradient-weighted activations to highlight important spatial regions.

### LIME
Perturbs the image into superpixels and trains a local linear model.
Identifies which regions push the model toward the predicted grade.

---

## Deployment (Streamlit Cloud)

1. Push this repo to GitHub
2. Go to https://share.streamlit.io
3. Connect your repo -> set `app.py` as entry point
4. Add `model/efficientnet_dr.pth` to the repo (use Git LFS for large files)
5. Deploy!

---

## Resume Talking Points

- Implemented **Ben Graham preprocessing** (domain-specific retinal image enhancement)
- Handled **severe class imbalance** (1808 Grade-0 vs 193 Grade-3) using weighted sampling + weighted loss
- Integrated **Grad-CAM** for spatial attribution and **LIME** for superpixel-level XAI
- Built end-to-end **Streamlit web app** with clinical PDF report generation
- Achieved competitive **Quadratic Weighted Kappa** (target: >0.80)

---

## Disclaimer

This tool is for **research and educational purposes only**.
It is NOT a substitute for professional ophthalmological diagnosis.
