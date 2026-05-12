# 🫁 AG-UNet: Lung Tumor Segmentation

![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red?logo=pytorch&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)
![Notebook](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter&logoColor=white)
![Task](https://img.shields.io/badge/Task-Medical%20Image%20Segmentation-blueviolet)

> Attention Gate UNet with a custom Hybrid Dice-Focal-Boundary Loss for accurate lung tumor segmentation under severe class imbalance.

---

## 🧠 What This Does

Lung tumor segmentation in CT scans is extremely challenging due to:
- Tiny tumor regions vs. massive background (severe class imbalance)
- Blurry, irregular tumor boundaries
- High variation in tumor shape and size

This project implements an **Attention Gate UNet (AG-UNet)** that addresses all three problems:
- **Attention Gates** force the model to focus only on relevant tumor regions, suppressing background noise
- **Hybrid Loss** = Dice Loss + Focal Loss + Boundary Loss working together to handle imbalance and sharp edges
- **Evaluation notebook** provides full metric analysis with visual predictions

---

## 🏗️ Model Architecture

```
Input CT Slice
     │
  Encoder (Contracting Path)
  ├── Conv Block 1 → MaxPool
  ├── Conv Block 2 → MaxPool
  ├── Conv Block 3 → MaxPool
  └── Conv Block 4 → MaxPool (Bottleneck)
     │
  Attention Gates ← skip connections with gating signal
     │
  Decoder (Expanding Path)
  ├── UpConv + Attention-weighted skip + Conv Block
  ├── UpConv + Attention-weighted skip + Conv Block
  ├── UpConv + Attention-weighted skip + Conv Block
  └── UpConv + Attention-weighted skip + Conv Block
     │
  Output Segmentation Map (sigmoid)
```

---

## 🔥 Hybrid Loss Function

The model is trained with a custom combined loss:

```
Total Loss = Dice Loss + Focal Loss + Boundary Loss
```

| Loss Component | Purpose |
|---|---|
| **Dice Loss** | Handles class imbalance by optimizing overlap directly |
| **Focal Loss** | Down-weights easy negatives, focuses on hard tumor pixels |
| **Boundary Loss** | Penalizes errors near tumor edges for sharper segmentation |

---

## 📁 Project Structure

```
ag-unet-lung-tumor-segmentation/
│
├── ag_unet_lung_tumor.ipynb     # Model definition, training pipeline
├── ag_unet_evaluation.ipynb     # Evaluation, metrics, visualizations
├── outputs/                     # Saved predictions and output masks
├── requirements.txt             # Python dependencies
└── README.md
```

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| Python 3.8+ | Core language |
| PyTorch | Model building and training |
| NumPy | Array operations |
| OpenCV / PIL | Image preprocessing |
| Matplotlib | Visualization of predictions |
| scikit-learn | Metric computation |
| Jupyter Notebook | Interactive experimentation |

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/sunilraj180805/ag-unet-lung-tumor-segmentation.git
cd ag-unet-lung-tumor-segmentation
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Prepare Your Dataset

This project is designed for lung CT segmentation datasets (e.g., LUNA16, LIDC-IDRI, or similar).

Organize your data as:
```
data/
├── images/       # CT scan slices (.png or .npy)
└── masks/        # Corresponding binary tumor masks
```

Update the dataset path in `ag_unet_lung_tumor.ipynb` before running.

### 4. Train the Model
```bash
jupyter notebook ag_unet_lung_tumor.ipynb
```
Run all cells to train the AG-UNet with the hybrid loss.

### 5. Evaluate the Model
```bash
jupyter notebook ag_unet_evaluation.ipynb
```
Generates predictions, overlays, and all evaluation metrics.

---

## 📊 Evaluation Metrics

The evaluation notebook reports:

| Metric | Description |
|---|---|
| **Dice Score** | Primary segmentation overlap metric |
| **IoU (Jaccard)** | Intersection over Union |
| **Precision** | How many predicted pixels are correct |
| **Recall** | How many actual tumor pixels are caught |
| **Boundary F1** | Accuracy at tumor edges specifically |

---

## ⚠️ Limitations

- Designed for 2D CT slice segmentation — 3D volumetric extension is a future scope
- Performance depends heavily on dataset quality and annotation accuracy
- Requires GPU for reasonable training time (tested on NVIDIA hardware)
- Dataset not included in this repo due to size/licensing — use a publicly available lung CT dataset

---

## 🔮 Future Scope

- Extend to 3D volumetric segmentation (3D UNet / V-Net)
- Add transformer-based attention (TransUNet, Swin-UNet)
- Test on LIDC-IDRI benchmark for standardized comparison
- Deploy as a lightweight inference API

---

## 📜 License

This project is licensed under the [MIT License](LICENSE).

---

## 🙋 Author

**Sunil Raj**  
[GitHub](https://github.com/sunilraj180805)
