# 🫁 Chest X-Ray AI Diagnostic System with DenseNet-121 & Explainable Grad-CAM

<p align="center">
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" />
  <img src="https://img.shields.io/badge/CUDA-000000?style=for-the-badge&logo=nvidia&logoColor=76B900" />
  <img src="https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white" />
  <img src="https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white" />
  <img src="https://img.shields.io/badge/Tailwind_CSS-38B2AC?style=for-the-badge&logo=tailwind-css&logoColor=white" />
  <img src="https://img.shields.io/badge/Chart.js-FF6384?style=for-the-badge&logo=chartdotjs&logoColor=white" />
  <img src="https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge" />
</p>

An end-to-end Deep Learning Clinical Decision Support System (CDSS) for automated detection and differential diagnosis of **Pneumonia** (Normal vs. Bacterial vs. Viral) from Chest Radiographs (CXR). The pipeline integrates **DenseNet-121**, Automatic Mixed Precision (AMP), **Explainable AI (Grad-CAM)** heatmaps, and a high-throughput **FastAPI** backend served to a modern glassmorphic web dashboard.

---

## 📌 Performance & Validation Metrics

Evaluated on an unseen test cohort of **624 clinical chest radiographs**:

- **Overall Test Accuracy:** `74.84%`
- **Macro F1-Score:** `0.7349`
- **Weighted F1-Score:** `0.7463`

### Diagnostic Classification Breakdown

| Condition | Precision | Recall (Sensitivity) | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **NORMAL** | **1.0000** | 0.4957 | 0.6629 | 234 |
| **BACTERIAL** | 0.8577 | **0.8967** | **0.8768** | 242 |
| **VIRAL** | 0.5255 | **0.9054** | 0.6650 | 148 |
| **Macro Average** | **0.7944** | **0.7659** | **0.7349** | **624** |

### Confusion Matrix

```text
               NORMAL    BACTERIAL     VIRAL
 NORMAL          116          22          96
 BACTERIAL         0         217          25
 VIRAL             0          14         134
```

🧠 Architecture & Methodology
Input CXR (224x224 RGB)
       │
       ▼
 [CLAHE Contrast Enhancement] (clipLimit=2.0, tileGrid=(8,8))
       │
       ▼
 [DenseNet-121 Backbone] (Feature Reuse Across Dense Blocks 1-4)
       │
       ├──► [Global Average Pooling + Dropout(0.3) + Linear(3)] ──► Probabilities (Softmax)
       │
       └──► [Grad-CAM Layer: features.denseblock4.denselayer16.conv2] ──► ColorMap JET Overlay



📁 Repository Structure
chest-xray-densenet121-gradcam/
├── backend/
│   ├── app.py                   # FastAPI endpoints & static frontend mounting
│   └── infer.py                 # DenseNet-121 inference & Grad-CAM extraction
├── frontend/
│   └── index.html               # Tailwind CSS & Chart.js interactive dashboard
├── models/
│   └── densenet121_xray_best.pt # Saved model checkpoint weights
├── src/
│   ├── dataset.py               # CLAHE preprocessing & DataLoader pipelines
│   ├── model.py                 # DenseNet-121 architectural definition
│   └── train.py                 # AMP-accelerated training loop & evaluation
├── chest_xray/                  # Chest Radiograph dataset directory
├── requirements.txt             # Global project dependencies
└── README.md                    # Project technical documentation


🚀 Quickstart & Execution
# 1. Install dependencies
pip install torch torchvision fastapi uvicorn opencv-python pillow numpy scikit-learn python-multipart

# 2. Launch FastAPI Server
python -m uvicorn backend.app:app --host 127.0.0.1 --port 8000 --reload

Navigate to http://127.0.0.1:8000 to upload radiographs and inspect Grad-CAM visualizations.


👨‍💻 Developer 
Yaarob Alhamamreh


GitHub: @yaarob-alhamamreh0


Focus: Artificial Intelligence & Deep Learning Engineering

