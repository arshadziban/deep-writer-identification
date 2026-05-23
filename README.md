# Deep Writer Identification

A handwriting-based writer identification system that benchmarks traditional machine learning and deep learning approaches on a dataset of 47 writers. Built for DIP 424 — Digital Image Processing Lab.

---

## Overview

This project implements an end-to-end pipeline for **writer identification from handwriting samples**: raw image collection, preprocessing, augmentation, feature extraction, and model evaluation. The goal is to correctly attribute an unlabeled handwriting sample to one of 47 known writers.

| Stage | Details |
|---|---|
| Writers | 47 |
| Raw images | 238 |
| Processed images | 2,380 (10× augmentation) |
| Task | 47-class classification |
| Best traditional model | k-NN — **59.24% accuracy** |
| Best deep learning model | ConvNeXt — **55.67% accuracy** |

---

## Repository Structure

```
deep-writer-identification/
├── fine_tune_dataset.ipynb    # Data preprocessing & augmentation pipeline
├── apply_model.ipynb          # Model training & evaluation (ML + DL)
├── requirements.txt           # Python dependencies
├── augmented_examples.png     # Sample augmented images
├── processed_examples.png     # Sample processed/filtered images
├── dataset/                   # Raw handwriting images (47 writers, ~238 images)
├── dataset_v2/                # Processed & augmented dataset (2,380 images)
└── DIP 424 Lab Assignment Guideline.pdf
```

> `dataset/`, `dataset_v2/`, and `.venv/` are excluded from version control via `.gitignore`.

---

## Pipeline

### 1. Data Preprocessing (`fine_tune_dataset.ipynb`)

Raw images in mixed formats (`.jpg`, `.jpeg`, `.heic`, `.dng`) are standardized and expanded into a rich training set.

**Image Filters (5 per image):**

| Filter | Purpose |
|---|---|
| Laplacian | Edge sharpness detection |
| Gabor | Stroke orientation & texture |
| Morphological Opening | Noise removal |
| Morphological Gradient | Stroke boundary extraction |
| Skeletonization | Pen stroke thinning |

**Augmentations (4 per image):**

| Augmentation | Parameters |
|---|---|
| Random Rotation | ±30° |
| Random Translation | ±20% width |
| Gaussian Noise | σ = 25 |
| Sharpness Enhancement | 0.5× – 2.0× |

**Output:** 238 raw → **2,380 processed images** (1 original + 5 filters + 4 augmentations per image)

---

### 2. Model Training & Evaluation (`apply_model.ipynb`)

#### Traditional ML (HOG Features)

Images are resized to **128×128** and HOG features are extracted (9 orientations, 8×8 px/cell, 2×2 cells/block → 8,100 features). Four classifiers are trained on an 80/20 stratified split.

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| k-NN (k=5) | **59.24%** | 72.96% | 59.24% | 61.97% |
| Random Forest (100 trees) | 58.61% | 65.98% | 58.61% | 58.18% |
| SVM (RBF kernel) | 57.77% | 83.64% | 57.77% | 62.73% |
| XGBoost | 55.88% | 57.43% | 55.88% | 55.38% |

#### Deep Learning (Transfer Learning)

Images are resized to **299×299**. Five pretrained CNN backbones are fine-tuned with frozen base weights and a custom classification head (Dropout → FC-256 → ReLU → FC-47). Training uses Adam (lr=0.0001) for 10 epochs with batch size 16.

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| **ConvNeXt** | **55.67%** | 60.46% | 55.67% | 52.99% |
| ResNet50 | 52.94% | 64.01% | 52.94% | 52.06% |
| GoogLeNet | 46.22% | 58.38% | 46.22% | 43.62% |
| EfficientNetV2 | 46.64% | 57.88% | 46.64% | 44.61% |
| InceptionV3 | 40.13% | 58.08% | 40.13% | 38.72% |

---

## Setup

### Prerequisites

- Python 3.9+
- CUDA-capable GPU recommended for deep learning models

### Installation

```bash
git clone https://github.com/arshadziban/deep-writer-identification.git
cd deep-writer-identification

python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### Usage

1. Place raw handwriting images in `dataset/` — one subdirectory per writer.
2. Run `fine_tune_dataset.ipynb` to generate the processed dataset in `dataset_v2/`.
3. Run `apply_model.ipynb` to train and evaluate all models.

---

## Dependencies

| Category | Libraries |
|---|---|
| Image Processing | OpenCV, Pillow, scikit-image, albumentations |
| Traditional ML | scikit-learn, XGBoost |
| Deep Learning | PyTorch, torchvision |
| Visualization | Matplotlib, Seaborn |
| Utilities | NumPy, pandas, tqdm |

See [requirements.txt](requirements.txt) for pinned versions.

---

## Results Summary

Traditional ML and deep learning models achieve comparable performance (~55–59% accuracy) on this 47-class problem. The limited dataset size (238 raw samples) is the primary constraint — augmentation (10× expansion) and transfer learning partially mitigate this.

- **k-NN** edges out other traditional models, benefiting from the high-dimensional HOG feature space.
- **ConvNeXt** outperforms other CNN backbones, likely due to its modern architecture with stronger inductive biases for texture and structure.
- **SVM** achieves the highest precision (83.64%) at the cost of lower recall, suggesting it is conservative but accurate when it does predict.

---

## License

This project was developed as part of DIP 424 — Digital Image Processing Lab coursework.
