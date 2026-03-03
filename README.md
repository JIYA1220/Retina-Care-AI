# Retina-Care AI — Diabetic Retinopathy Severity Grader (Grad-CAM + LIME + PDF Report)

Retina-Care AI is a Streamlit-based application that analyzes retinal fundus images and predicts **Diabetic Retinopathy (DR) severity (Grade 0–4)**. It also provides **model explainability** using **Grad-CAM** and **LIME**, and generates a downloadable **clinical-style PDF report** to summarize results.

> **Disclaimer:** This project is for research/education only. It is **not** a medical device and must **not** be used as a substitute for professional diagnosis.

---

## Features

- Upload a retinal fundus image (**PNG/JPG**) and predict **DR grade (0–4)**
- Shows **confidence + per-class probability distribution**
- **Grad-CAM heatmap** to visualize important regions
- **LIME superpixel explanation** for local interpretability
- Download a **clinical-style PDF report**
- Optional **temperature scaling calibration** for more reliable probabilities
- Includes training script for the APTOS 2019 dataset

---

## Project Structure

```text
retina_grader/
├── app.py                        # Streamlit web app (entry point)
├── train.py                      # Training script
├── requirements.txt
├── README.md
│
├── model/
│   ├── model.py                  # EfficientNet-B0 architecture
│   ├── predict.py                # Inference pipeline
│   └── efficientnet_dr.pth       # Saved weights (after training)
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
└── data/                         # Kaggle dataset goes here
    ├── train.csv
    └── train_images/
        ├── 0a4e1a29ffff.png
        └── ...
```

---

## Dataset

This project uses the **APTOS 2019 Blindness Detection** dataset:  
https://www.kaggle.com/competitions/aptos2019-blindness-detection/data

- ~**3,662** retinal fundus images
- Labels: **0 (No DR)** → **4 (Proliferative DR)**

### Download via Kaggle CLI

```bash
# Install Kaggle CLI
pip install kaggle

# Add your Kaggle API key from https://www.kaggle.com/settings
mkdir ~/.kaggle
cp kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json  # recommended on macOS/Linux

# Download dataset
kaggle competitions download -c aptos2019-blindness-detection -p data/

# Unzip
cd data && unzip aptos2019-blindness-detection.zip
```

---

## Setup

### 1) Create a virtual environment (recommended)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

### 2) Install dependencies

```bash
pip install -r requirements.txt
```

---

## Training

Train the model using APTOS 2019 images and labels:

```bash
python train.py --data_dir data --epochs 15 --batch_size 8
```

| Argument | Default | Description |
|---|---:|---|
| `--data_dir` | `data` | Folder containing `train.csv` and `train_images/` |
| `--epochs` | `15` | Number of training epochs |
| `--batch_size` | `8` | Batch size (keep ≤ 8 for CPU) |
| `--lr` | `1e-4` | Learning rate |

After training, weights are saved to:

```text
model/efficientnet_dr.pth
```

**Tip:** Start with fewer epochs first (e.g., `--epochs 3` or `--epochs 5`) to confirm everything runs correctly.

---

## Confidence Calibration (Optional but Recommended)

Neural networks can be overconfident. This project supports **Temperature Scaling** to calibrate predicted probabilities (so confidence values are more realistic).

```bash
python -m calibration.temperature_scaling --weights model/efficientnet_dr.pth --data_dir data
```

- Output file: `model/temperature.pt`
- If this file exists, the app automatically applies calibration during inference.

---

## Run the App (Streamlit)

```bash
streamlit run app.py
```

Then open:

```text
http://localhost:8501
```

### What you’ll see in the app
- Predicted **DR grade (0–4)** and confidence
- **Probability chart** for all classes
- **Grad-CAM** visualization
- **LIME** visualization
- Button to download a **PDF report**

---

## Model Overview

- **Backbone:** EfficientNet-B0 (ImageNet pretrained via `timm`)
- **Classifier head:** Dropout → Linear(1280→256) → ReLU → Linear(256→5)
- **Loss:** Weighted Cross Entropy (helps with imbalance)
- **Sampling:** WeightedRandomSampler
- **Preprocessing:** Ben Graham style enhancement (retinal image normalization)
- **Augmentation:** Albumentations transforms (flip/rotate/color/noise)
- **Evaluation metric:** Quadratic Weighted Kappa

---

## Explainability

### Grad-CAM
Grad-CAM highlights image regions that most strongly influence the model’s decision by combining gradients with convolutional feature maps.

### LIME
LIME creates superpixels and perturbs them to learn a local surrogate model, showing which regions push the prediction toward a specific grade.

---

## Deployment (Streamlit Community Cloud)

1. Push your repository to GitHub
2. Go to https://share.streamlit.io
3. Select your repo and set **`app.py`** as the entry point
4. Add `model/efficientnet_dr.pth` to your repo (use **Git LFS** if needed)
5. Deploy

---

## Disclaimer

This tool is provided for **research and educational purposes only**.  
It is **not** intended for real-world clinical diagnosis and should not be used for medical decision-making.
