# 🧪 Fingerprint-Based Blood Group Detection

A deep learning system that predicts blood groups (A+, A−, B+, B−, AB+, AB−, O+, O−) from fingerprint images using an ensemble of CNN models, complete with Explainable AI (XAI) visualizations via a Streamlit web app.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Models](#models)
- [XAI Methods](#xai-methods)
- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Results](#results)
- [License](#license)

---

## Overview

This project explores the hypothesis that fingerprint ridge patterns carry biometric correlations with blood groups. Four CNN architectures are trained and combined into an ensemble predictor. A Streamlit-based dashboard lets users upload a fingerprint image and instantly see:

- Individual predictions from each model with confidence scores
- An ensemble (weighted voting or probability averaging) final prediction
- Grad-CAM, Occlusion Sensitivity, and SmoothGrad visualizations that explain *why* the model made its prediction

---

## ✨ Features

- **Multi-model ensemble** — AlexNet, LeNet, ResNet-34, and VGG-16, combined for robust predictions
- **Two ensemble strategies** — weighted voting and probability averaging
- **Explainable AI (XAI)** — three distinct visualization methods per model
- **Interactive Streamlit UI** — drag-and-drop image upload, live confidence charts, radar plots, and detailed result tables
- **Per-model performance metrics** — parameter counts, layer counts, input/output shapes

---

## 📁 Project Structure

```
Fingerprint-Based-Blood-Group-Detection/
│
├── Dataset/                        # Training and evaluation fingerprint images
│   └── ...
│
├── fingerprint_detection/          # Jupyter notebooks for model training
│   └── ...
│
├── ridge_analysis/                 # Ridge feature extraction notebooks/scripts
│   └── ...
│
├── app_1.py                        # Main Streamlit application (UI + XAI)
├── ensemble_prediction.py          # EnsemblePredictor class (weighted voting & averaging)
├── xai.py                          # Standalone XAI analysis module
├── LICENSE                         # MIT License
└── README.md
```

---

## 🤖 Models

| Model | Input Size | Architecture | Description |
|-------|-----------|--------------|-------------|
| **AlexNet** | 227 × 227 | 8-layer deep CNN | Pioneer deep network adapted for fingerprint features |
| **LeNet** | 224 × 224 | Classic CNN | Lightweight, fast baseline |
| **ResNet-34** | 256 × 256 | 34-layer residual network | Deep architecture with skip connections to prevent vanishing gradients |
| **VGG-16** | 256 × 256 | 16-layer very deep CNN | Strong spatial feature extractor with uniform 3×3 convolutions |

Trained model files (`.keras` / `.h5`) are expected at the paths configured in `MODEL_PATHS` inside `app_1.py`. Update these paths to match your local setup before running.

**Supported blood groups:** `A+`, `A−`, `B+`, `B−`, `AB+`, `AB−`, `O+`, `O−`

---

## 🔍 XAI Methods

The `XAIAnalyzer` class (in `xai.py` and integrated into `app_1.py`) provides three explanation techniques:

| Method | What it shows |
|--------|--------------|
| **Grad-CAM** | Highlights the fingerprint regions that most strongly activated the final convolutional layer for the predicted class |
| **Occlusion Sensitivity** | Systematically blocks patches of the image and measures how much the prediction confidence drops — brighter areas are more critical |
| **SmoothGrad** | Averages gradients over multiple noise-perturbed copies of the image to reduce visual noise and reveal stable feature attributions |

Each method produces a colour-mapped overlay (JET / HOT / VIRIDIS colormaps respectively) that is displayed alongside the original fingerprint in the app.

---

## ⚙️ Installation

**Prerequisites:** Python 3.8+

```bash
# 1. Clone the repository
git clone https://github.com/ShivamMitra/Fingerprint-Based-Blood-Group-Detection.git
cd Fingerprint-Based-Blood-Group-Detection

# 2. (Recommended) Create a virtual environment
python -m venv venv
source venv/bin/activate        # Linux/macOS
venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install streamlit tensorflow numpy opencv-python pillow plotly seaborn matplotlib
```

> **Note:** TensorFlow GPU is recommended for faster XAI generation. CPU inference works but SmoothGrad and Occlusion maps may take a minute or two per model.

---

## 🚀 Usage

### Running the Streamlit App

```bash
streamlit run app_1.py
```

Then open `http://localhost:8501` in your browser.

**Workflow:**
1. Ensure the trained model files are placed at the paths defined in `MODEL_PATHS` inside `app_1.py`.
2. Upload a fingerprint image (JPG, PNG, BMP).
3. View per-model predictions, confidence bar charts, and radar comparison plots.
4. Expand the XAI section to explore Grad-CAM, Occlusion, and SmoothGrad heatmaps for each model.

### Using the Ensemble Predictor Programmatically

```python
from ensemble_prediction import EnsemblePredictor
import tensorflow as tf
import numpy as np
from PIL import Image

# Load your models
models = {
    'ResNet-34': (tf.keras.models.load_model('path/to/resnet.h5'), 'resnet'),
    'VGG-16':    (tf.keras.models.load_model('path/to/vgg16.h5'),  'vgg'),
}

predictor = EnsemblePredictor(models)

image = np.array(Image.open('fingerprint.png').convert('RGB'))

# Weighted voting ensemble
result = predictor.predict_ensemble(image, method='weighted_voting')
print(result['ensemble_prediction'], result['ensemble_confidence'])

# Probability averaging ensemble
result = predictor.predict_ensemble(image, method='averaging')
print(result['ensemble_prediction'], result['average_probabilities'])
```

---

## 📊 Dataset

The `Dataset/` folder contains fingerprint images organised by blood group label. The `fingerprint_detection/` notebooks walk through data loading, augmentation, and model training. The `ridge_analysis/` directory contains scripts for fingerprint ridge feature extraction used during preprocessing.

---

## 📈 Results

The Streamlit dashboard displays:

- **Confidence bar chart** — per-class probability for each model side-by-side
- **Radar chart** — multi-model comparison across all 8 blood group classes
- **Results table** — prediction, confidence score, and a traffic-light performance indicator (🟢 High / 🟡 Medium / 🔴 Low) for each model

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
