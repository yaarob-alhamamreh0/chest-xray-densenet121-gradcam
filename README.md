# 🫁 Chest X-Ray AI Diagnostic System with DenseNet-121 & Explainable Grad-CAM

<p align="center">
  <img src="[https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)" />
  <img src="[https://img.shields.io/badge/CUDA-000000?style=for-the-badge&logo=nvidia&logoColor=76B900](https://img.shields.io/badge/CUDA-000000?style=for-the-badge&logo=nvidia&logoColor=76B900)" />
  <img src="[https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)" />
  <img src="[https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white](https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)" />
  <img src="[https://img.shields.io/badge/Tailwind_CSS-38B2AC?style=for-the-badge&logo=tailwind-css&logoColor=white](https://img.shields.io/badge/Tailwind_CSS-38B2AC?style=for-the-badge&logo=tailwind-css&logoColor=white)" />
  <img src="[https://img.shields.io/badge/Chart.js-FF6384?style=for-the-badge&logo=chartdotjs&logoColor=white](https://img.shields.io/badge/Chart.js-FF6384?style=for-the-badge&logo=chartdotjs&logoColor=white)" />
  <img src="[https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)" />
</p>

An end-to-end Deep Learning Clinical Decision Support System (CDSS) for automated detection and differential diagnosis of **Pneumonia** (Normal vs. Bacterial vs. Viral) from Chest Radiographs (CXR). The pipeline integrates **DenseNet-121**, Automatic Mixed Precision (AMP), **Explainable AI (Grad-CAM)** heatmaps, and a high-throughput **FastAPI** backend served to a modern glassmorphic web dashboard.

---

## 📌 Performance & Validation Metrics

Evaluated on an unseen test cohort of **624 clinical chest radiographs**:

* **Overall Test Accuracy:** `74.84%`
* **Macro F1-Score:** `0.7349`
* **Weighted F1-Score:** `0.7463`

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

### 🧠 Architecture & Methodology
```

Input CXR (224x224 RGB)
       │
       ▼
[CLAHE Preprocessing] (clipLimit=2.0, tileGrid=(8,8))
       │
       ▼
[DenseNet-121 Feature Extractor]
       │
       ├──► [Adaptive AvgPool + Dropout(0.3) + Linear(3)] ──► Softmax Probabilities
       │
       └──► [Grad-CAM Hook: denseblock4.denselayer16.conv2] ──► ColorMap JET Overlay
```

###📁 Repository Structure
```
chest-xray-densenet121-gradcam/
├── backend/
│   ├── app.py                     # FastAPI backend & web server
│   └── infer.py                   # PyTorch inference & Grad-CAM pipeline
├── frontend/
│   └── index.html                 # Tailwind CSS & Chart.js interface
├── models/
│   └── densenet121_xray_best.pt   # DenseNet-121 trained checkpoint
├── src/
│   ├── dataset.py                 # PyTorch Dataset & CLAHE transforms
│   ├── model.py                   # Network architectural setup
│   └── train.py                   # AMP-accelerated training pipeline
├── requirements.txt               # Environment dependencies
└── README.md                      # Technical documentation
```
###🚀 Quickstart & Execution
```
1. Install dependencies
pip install torch torchvision fastapi uvicorn opencv-python pillow numpy scikit-learn python-multipart
2. Launch FastAPI Server
python -m uvicorn backend.app:app --host 127.0.0.1 --port 8000 --reload

Navigate to [http://127.0.0.1:8000](http://127.0.0.1:8000) to upload radiographs and inspect Grad-CAM visualizations.
```
### 👨‍💻Developer
Yaarob Alhamamreh
GitHub: @yaarob-alhamamreh0

Focus: Artificial Intelligence & Deep Learning Engineering[cite: 1, 2]
