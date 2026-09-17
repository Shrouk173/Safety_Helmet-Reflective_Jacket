# 🦺 Construction Site PPE Compliance Monitor & Multi-Architecture Benchmark

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=flat\&logo=python\&logoColor=white)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Framework-Ultralytics%20YOLO%20%7C%20PyTorch%20Torchvision-EE4C2C?style=flat\&logo=pytorch\&logoColor=white)](https://github.com/ultralytics/ultralytics)
[![UI](https://img.shields.io/badge/Deployment-Streamlit-FF4B4B?style=flat\&logo=streamlit\&logoColor=white)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
# WORKING LINK LIVE
# https://safety-helmet-reflective-jacket.vercel.app/
An automated Computer Vision and edge-ready deep learning system designed to monitor workplace occupational health and safety standards in real time.

The system detects construction workers and verifies compliance with mandatory **Personal Protective Equipment (PPE)** requirements, specifically:

* 🪖 Safety helmets / hardhats
* 🦺 High-visibility reflective safety vests / jackets

Non-compliance is identified through visual detection, worker-level compliance logic, real-time alerts, and an automated audit logger.

The project also provides a comprehensive benchmark comparing three object detection architectures:

**YOLO11s · YOLOv8n · RetinaNet**

Designed for automated visual auditing across construction sites, civil infrastructure projects, manufacturing plants, and heavy industrial facilities.

---

## 📌 Table of Contents

1. [Project Overview](#-project-overview)
2. [Key Features](#-key-features)
3. [System Architecture & Compliance Logic](#-system-architecture--compliance-logic)
4. [Architectural Comparison](#-architectural-comparison)
5. [Theoretical Foundations](#-theoretical-foundations)
6. [Empirical Evaluation & Benchmarking](#-empirical-evaluation--benchmarking)
7. [Trade-Off Analysis & Deployment Strategy](#-trade-off-analysis--deployment-strategy)
8. [Dataset Specification](#-dataset-specification)
9. [Negative Class Supervision](#-negative-class-supervision)
10. [Repository Structure](#-repository-structure)
11. [Installation & Environment Setup](#-installation--environment-setup)
12. [Pipeline Execution](#-pipeline-execution)
14. [Model Weights](#-model-weights)
15. [Engineering Team](#-engineering-team)
16. [License & Acknowledgments](#-license--acknowledgments)

---

# 📌 Project Overview

## Problem Statement

Construction sites contain multiple safety hazards, and manually monitoring PPE compliance is time-consuming and prone to human error.

Traditional object detectors trained only on affirmative classes such as `helmet` and `vest` may leave uncertainty when protective equipment is absent, occluded, or outside the field of view.

This project addresses the problem through:

1. Explicit positive and negative PPE classes.
2. Multi-architecture object detection benchmarking.
3. Spatial matching between workers and detected PPE.
4. Deterministic worker-level compliance evaluation.
5. Real-time visual alerts and audit logging.

## Objectives

* Detect construction workers in complex visual environments.
* Identify whether each worker is wearing a safety helmet.
* Identify whether each worker is wearing a reflective safety vest.
* Detect explicit PPE violations.
* Compare lightweight and classical object detection architectures.
* Evaluate accuracy, recall, computational complexity, and inference speed.
* Provide an interactive Streamlit dashboard for inspection and benchmarking.
* Support edge-oriented deployment scenarios.

> **Important:** This system is designed as an automated visual auditing and decision-support tool. Safety-critical entry restrictions or operational shutdowns require appropriate human and organizational verification.

---

# 🚀 Key Features

### Computer Vision

* Multi-class PPE object detection.
* Worker localization.
* Helmet and reflective vest detection.
* Explicit `no_helmet` and `no_vest` classes.
* Spatial geometry matching between workers and PPE.

### Model Benchmarking

* YOLO11s attention-enhanced detector.
* YOLOv8n lightweight edge baseline.
* RetinaNet ResNet-50-FPN classical baseline.
* Identical dataset splits and evaluation resolution.
* mAP, precision, recall, FLOPs, parameter count, and FPS benchmarking.

### Interactive Dashboard

* Image upload and webcam input.
* Adjustable worker and gear confidence thresholds.
* Real-time compliance alerts.
* Per-worker compliance table.
* CSV audit report export.
* Side-by-side model comparison.
* Diagnostic curves and confusion matrices.

---

# 🧠 System Architecture & Compliance Logic

## Detection Pipeline

```text
Input Image / Webcam
        │
        ▼
Preprocessing
        │
        ▼
Object Detection Model
(YOLO11s / YOLOv8n / RetinaNet)
        │
        ▼
Detect PPE and Workers
        │
        ▼
Spatial Geometry Matching
        │
        ▼
Worker-Level Compliance Evaluation
        │
        ▼
Compliance State
        │
        ├── Visual Alert Banner
        ├── Annotated Image
        └── Audit Log / CSV Export
```

## Five-Class Detection Schema

The model is trained using five target classes:

| Class ID | Class Name  | Description                                         |
| -------- | ----------- | --------------------------------------------------- |
| 0        | `helmet`    | Safety hardhat present on worker's head.            |
| 1        | `no_helmet` | Worker's head exposed without protective headgear.  |
| 2        | `no_vest`   | Worker's torso missing high-visibility safety vest. |
| 3        | `person`    | Human worker bounding frame.                        |
| 4        | `vest`      | High-visibility reflective vest/jacket present.     |

## Spatial Matching Algorithm

During inference, the system determines whether detected PPE belongs to a worker by checking whether the center of the PPE bounding box lies inside the worker bounding box.

For a bounding box:

$$
B = (x_1, y_1, x_2, y_2)
$$

The center is:

$$
\text{Center}(B) =
\left(
\frac{x_1+x_2}{2},
\frac{y_1+y_2}{2}
\right)
$$

The PPE center is considered associated with a worker when:

$$
\text{Center}(B_{\text{gear}})
\in B_{\text{person}}
$$

This enables worker-level safety evaluation rather than simply counting detected helmets and vests.

## Compliance State Machine

Each worker is evaluated using a deterministic three-tier decision matrix.

| Helmet    | Vest      | Compliance State      | Visual Border | UI Alert               |
| --------- | --------- | --------------------- | ------------- | ---------------------- |
| ✅ Present | ✅ Present | 🟢 Compliant          | Solid Green   | Site Compliant         |
| ✅ Present | ❌ Missing | 🟠 Warning            | Amber Orange  | Safety Warning         |
| ❌ Missing | ✅ Present | 🟠 Warning            | Amber Orange  | Safety Warning         |
| ❌ Missing | ❌ Missing | 🔴 Critical Violation | Heavy Red     | Critical Pulsing Alert |

### Enforcement Logic

| State                 | Description                          |
| --------------------- | ------------------------------------ |
| 🟢 Compliant          | Worker has both required PPE items.  |
| 🟠 Warning            | One required PPE item is missing.    |
| 🔴 Critical Violation | Both required PPE items are missing. |

**Note:** The color values and enforcement behavior are application-level settings. Actual workplace enforcement must follow site safety procedures.

---

# 🏗️ Architectural Comparison

Three distinct object detection paradigms were trained and evaluated on identical data splits to study parameter scaling, computational complexity, and inference latency.

| Architecture | Detection Paradigm             | Backbone / Main Modules | Parameters | GFLOPs |
| ------------ | ------------------------------ | ----------------------- | ---------: | -----: |
| RetinaNet    | Dense anchor-based             | ResNet-50 + FPN         |      34.0M |   90.0 |
| YOLOv8n      | Anchor-free                    | CSPDarknet + C2f        |       3.2M |    8.7 |
| YOLO11s      | Attention-enhanced anchor-free | C3k2 + C2PSA            |       9.4M |   21.4 |

---

# 🔬 Theoretical Foundations

## 1. RetinaNet — Classical Dense Baseline

### Backbone & Neck

RetinaNet uses:

* ResNet-50 residual backbone.
* Top-Down Feature Pyramid Network (FPN).
* Lateral connections for multi-scale semantic feature maps.
* Feature pyramid levels P3 through P7.

### Detection Mechanism

RetinaNet is a single-stage, dense, anchor-based detector designed to handle severe foreground-background class imbalance.

It uses **Focal Loss**:

$$
FL(p_t) =
-\alpha_t(1-p_t)^\gamma \log(p_t)
$$

Where:

* $p_t$ is the predicted probability of the target class.
* $\alpha_t$ is the class weighting factor.
* $\gamma = 2.0$ is the focusing parameter.

Focal Loss downweights easy, well-classified background examples and focuses training on difficult targets.

### Limitations

* Relies on dense, hand-crafted anchor grids.
* Requires multiple anchor scales and aspect ratios.
* ResNet-50 introduces additional computational overhead.
* Higher memory and compute requirements for edge deployment.

---

## 2. YOLOv8n — Lightweight Edge Baseline

### Backbone & Neck

YOLOv8n uses a modified CSPDarknet backbone with **C2f (Cross-Stage Partial with two convolutions)** blocks.

The C2f module:

* Splits low-level channel features.
* Uses residual bottlenecks.
* Merges feature information across stages.
* Improves gradient flow while reducing parameter redundancy.

### Detection Mechanism

YOLOv8n eliminates predefined anchor boxes through an anchor-free detection approach.

Key components:

* Task-Aligned Assignment (TAL).
* Distribution Focal Loss (DFL).
* CIoU Loss.
* Decoupled classification and localization heads.

### Strengths

* Small model size.
* Low computational requirements.
* High inference throughput.
* Suitable for constrained edge devices.

---

## 3. YOLO11s — Proposed Attention-Enhanced Model

### Backbone & Neck

YOLO11s introduces:

* C3k2 processing blocks.
* C2PSA (Cross-Stage Partial Spatial Attention) modules.
* High-level semantic attention at stages such as P5.

### Attention Mechanism

The model incorporates multi-head spatial self-attention within residual cross-stage structures.

$$
\text{Attention}(Q,K,V)
=
\text{softmax}
\left(
\frac{QK^T}{\sqrt{d_k}}
\right)V
$$

Where:

* $Q$ = Query.
* $K$ = Key.
* $V$ = Value.
* $d_k$ = Key dimensionality.

Spatial self-attention allows the model to capture long-range contextual relationships across feature maps.

This can help identify subtle visual details such as:

* Hardhat brims.
* Thin reflective vest straps.
* Small PPE objects.
* Workers in complex or overexposed backgrounds.

---

# 📊 Empirical Evaluation & Benchmarking

All models were evaluated under identical conditions:

| Evaluation Setting | Value                                          |
| ------------------ | ---------------------------------------------- |
| Dataset            | PPE Dataset                                    |
| Image Resolution   | 640 × 640                                      |
| Evaluation Split   | Held-out test partition                        |
| Frameworks         | Ultralytics / Torchvision                      |
| Benchmark Metrics  | mAP, Precision, Recall, FPS, FLOPs, Parameters |

## Overall Model Performance

| Architecture  | Model File               | Framework   | Parameters |   GFLOPs |   mAP@0.5 | mAP@0.5:0.95 | Precision |    Recall |  CPU FPS | Primary Role              |
| ------------- | ------------------------ | ----------- | ---------: | -------: | --------: | -----------: | --------: | --------: | -------: | ------------------------- |
| **YOLO11s**   | `best.pt`                | Ultralytics |   **9.4M** | **21.4** | **92.0%** |    **59.8%** | **87.5%** | **89.5%** |      6.2 | Proposed Production Model |
| **YOLOv8n**   | `best_8n.pt`             | Ultralytics |   **3.2M** |  **8.7** | **90.0%** |    **57.0%** | **86.0%** | **88.0%** | **13.2** | Lightweight Edge Baseline |
| **RetinaNet** | `retinanet_ppe_best.pth` | Torchvision |  **34.0M** | **90.0** | **81.0%** |    **47.6%** | **85.0%** | **83.0%** |     12.0 | Classical Baseline        |

> Benchmark values are based on the supplied project evaluation results.

## Per-Class Detection Breakdown — YOLO11s

| Class           | Targets Evaluated | Precision |    Recall |   mAP@0.5 | mAP@0.5:0.95 |
| --------------- | ----------------: | --------: | --------: | --------: | -----------: |
| `person`        |               239 |     0.892 |     0.935 |     0.941 |        0.652 |
| `vest`          |               171 |     0.884 |     0.912 |     0.932 |        0.648 |
| `helmet`        |               201 |     0.871 |     0.889 |     0.918 |        0.584 |
| `no_helmet`     |                45 |     0.865 |     0.872 |     0.905 |        0.551 |
| `no_vest`       |                56 |     0.863 |     0.867 |     0.904 |        0.555 |
| **All Classes** |           **712** | **0.875** | **0.895** | **0.920** |    **0.598** |

### Detection Analysis

| Class       | Analysis                                                  |
| ----------- | --------------------------------------------------------- |
| `person`    | Robust full-body localization under heavy visual clutter. |
| `vest`      | Strong detection driven by neon contrast and texture.     |
| `helmet`    | High accuracy on rounded hardhat structures.              |
| `no_helmet` | Distinguishes bare heads/hair from safety gear.           |
| `no_vest`   | Distinguishes standard shirts/jackets from safety gear.   |

---

# ⚖️ Trade-Off Analysis & Deployment Strategy

## YOLO11s — Proposed Production Model

YOLO11s is the attention-enhanced model used for the primary production configuration.

### Characteristics

* 92.0% mAP@0.5.
* 59.8% mAP@0.5:0.95.
* 89.5% Recall.
* 9.4M parameters.
* 21.4 GFLOPs.

### Design Rationale

In workplace safety auditing, recall is an important metric because missed violations may create safety risks.

YOLO11s provides the highest recall among the evaluated models in the supplied benchmark, while retaining a relatively lightweight architecture.

The C2PSA attention mechanism is intended to help capture subtle PPE features and small objects in visually complex scenes.

---

## YOLOv8n — Edge Deployment Alternative

YOLOv8n is the lightweight baseline for resource-constrained devices.

### Characteristics

* 13.2 FPS on CPU.
* 3.2M parameters.
* 8.7 GFLOPs.
* 90.0% mAP@0.5.

### Potential Deployment Targets

* Raspberry Pi 5.
* NVIDIA Jetson Nano.
* Battery-powered smart cameras.
* Other constrained edge processors.

Its low parameter count and compute requirements make it a suitable candidate when inference speed and memory usage are important.

---

## RetinaNet — Classical Baseline

RetinaNet provides a classical dense anchor-based reference point.

### Characteristics

* ResNet-50-FPN backbone.
* 34.0M parameters.
* 90.0 GFLOPs.
* 81.0% mAP@0.5.
* 47.6% mAP@0.5:0.95.

### Benchmark Role

RetinaNet demonstrates the computational and performance trade-offs associated with a classical anchor-based architecture compared with the evaluated YOLO models.

---

# 📂 Dataset Specification

## Dataset

**Personal-Protective-Equipment (PPE) Dataset**

* Curated by: ndomalau.
* Hosted on: Kaggle.
* Dataset size: 4,060 annotated images.
* Domain: Construction site PPE detection.
* Conditions: Various construction lighting and weather conditions.

### Dataset Link

[Personal Protective Equipment (PPE) Dataset — Kaggle](https://www.kaggle.com/datasets/ndomalau/personal-protective-equipment-ppe-dataset)

## Annotation Format

The dataset uses YOLO normalized bounding box coordinates:

```text
class_id x_center y_center width height
```

All bounding box coordinates are normalized to the range:

```text
[0, 1]
```

## Dataset Split

| Split      | Percentage |
| ---------- | ---------: |
| Training   |        70% |
| Validation |        15% |
| Testing    |        15% |

## Target Classes

```text
0: helmet
1: no_helmet
2: no_vest
3: person
4: vest
```

## Programmatic Dataset Download

```python
import kagglehub

dataset_path = kagglehub.dataset_download(
    "ndomalau/personal-protective-equipment-ppe-dataset"
)

print(f"Dataset mounted at: {dataset_path}")
```

---

# 🧩 Negative Class Supervision

## Why Explicit Negative Classes Matter

Standard PPE datasets may annotate only positive classes such as:

* `helmet`
* `vest`

If a worker is not wearing protective equipment, the model may simply produce no detection in that region.

This creates uncertainty:

* Is the equipment missing?
* Is the worker occluded?
* Is the gear outside the camera view?
* Did the detector miss the object?

## Proposed Solution

The system explicitly supervises:

```text
no_helmet
no_vest
```

This teaches the network to recognize visual patterns associated with non-compliance, such as:

* Exposed hair.
* Exposed skin.
* Standard civilian clothing.
* Shirts and jackets without reflective safety gear.

### Benefit

The downstream compliance system receives an explicit signal of potential non-compliance rather than relying solely on the absence of a positive detection.

---

# 📁 Repository Structure

```text
Safety_Helmet-Reflective_Jacket/
│
├── data/
│   └── ppe_data.yaml
│       # Dataset split directories and class mappings
│
├── models/
│   ├── best.pt
│   │   # Fine-tuned YOLO11s weights (18.4 MB)
│   ├── best_8n.pt
│   │   # Fine-tuned YOLOv8n weights (6.5 MB)
│   └── retinanet_ppe_best.pth
│       # Trained RetinaNet weights (136 MB)
│
├── docs/
│   ├── evaluation_output/
│   │   └── test_metrics/
│   │       # YOLO11s diagnostic plots
│   ├── evaluation8n_output/
│   │   └── test_metrics/
│   │       # YOLOv8n diagnostic plots
│   └── evaluation_retinanet_output/
│       └── test_metrics/
│           # RetinaNet diagnostic plots
│
├── notebooks/
│   ├── 01_data_preprocessing_eda.ipynb
│   │   # Dataset auditing and verification
│   └── 02_model_training_kaggle.ipynb
│       # Model training and ablation experiments
│
├── src/
│   ├── train.py
│   │   # Standalone training script
│   └── evaluate.py
│       # Benchmarking and evaluation script
│
├── app/
│   └── app.py
│       # Streamlit dashboard
│
├── .gitignore
├── requirements.txt
├── LICENSE
└── README.md
```

---

# ⚙️ Installation & Environment Setup

## Prerequisites

* Python 3.9, 3.10, or 3.11.
* Git.
* CUDA 11.8+ or 12.0+ compatible GPU (optional).
* CPU execution is supported.

## 1. Clone the Repository

```bash
git clone https://github.com/mariammmo/Safety_Helmet-Reflective_Jacket.git

cd Safety_Helmet-Reflective_Jacket
```

## 2. Create a Virtual Environment

```bash
python -m venv venv
```

## 3. Activate the Environment

### Linux / macOS

```bash
source venv/bin/activate
```

### Windows PowerShell

```powershell
.\venv\Scripts\activate
```

## 4. Upgrade pip

```bash
pip install --upgrade pip
```

## 5. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# 🚀 Pipeline Execution

## 1. Model Training

The training pipeline supports automated data augmentation, including:

* HSV shift.
* Rotation.
* Shear.
* Scale.
* Horizontal flipping.
* Mosaic augmentation.

### Train YOLO11s

```bash
python src/train.py \
    --model yolo11s.pt \
    --epochs 50 \
    --batch 16 \
    --imgsz 640 \
    --name YOLO11_PPE
```

### Train YOLOv8n

```bash
python src/train.py \
    --model yolov8n.pt \
    --epochs 50 \
    --batch 16 \
    --imgsz 640 \
    --name YOLOv8n_PPE
```

## 2. Quantitative Evaluation

Run the evaluation suite on the held-out test split.

The evaluation script computes:

* mAP@0.5.
* mAP@0.5:0.95.
* Per-class precision and recall.
* Parameter count.
* Inference latency.
* FPS.
* FLOPs.
* Diagnostic plots.

```bash
python src/evaluate.py \
    --weights models/best.pt \
    --data data/ppe_data.yaml \
    --device cpu
```

## 3. Launch the Streamlit Application

```bash
cd app

streamlit run app.py
```

The web interface will initialize at:

```text
http://localhost:8501
```

---

# 🖥️ Streamlit Application

The application provides three functional views.

## 1. Live Compliance Inspector

### Features

* Image upload.
* Webcam input.
* Dual-confidence sliders.
* Real-time compliance alert banners.
* Per-worker status logging.
* Individual confidence scores.
* CSV audit report export.

### Confidence Thresholds

| Detection Type   | Default Threshold | Purpose                                                     |
| ---------------- | ----------------: | ----------------------------------------------------------- |
| Worker Detection |              0.60 | Reduce false detections from background objects or animals. |
| Gear Detection   |              0.35 | Improve detection of small hardhats and distant vests.      |

### Dynamic Alert Strip

The top banner updates based on the compliance state:

* 🟢 Solid Green — Full compliance.
* 🟠 Amber — Missing PPE item.
* 🔴 Pulsing Red — Complete absence of PPE.

---

## 2. Head-to-Head Architecture Arena

The Model Arena allows simultaneous inference using two selected models on the same input frame.

### Supported Comparisons

* YOLO11s vs. YOLOv8n.
* YOLO11s vs. RetinaNet.
* YOLOv8n vs. RetinaNet.

### Comparison Features

* Side-by-side detection results.
* Bounding box alignment comparison.
* False positive comparison.
* Inference speed comparison.
* Metric comparison tables.
* Interactive Plotly radar charts.

---

## 3. Evaluation Artifacts & Diagnostic Viewer

The application provides access to evaluation artifacts such as:

* Training loss curves (`results.png`).
* Normalized confusion matrices.
* Precision-recall curves.
* Model diagnostic plots.
* Evaluation metrics.

These artifacts help inspect model behavior and compare training and evaluation performance.

---

# 📦 Model Weights

Pre-trained checkpoints are available in the repository.

| Target Model | Architecture                     | File                            |    Size |
| ------------ | -------------------------------- | ------------------------------- | ------: |
| YOLO11s      | Ultralytics Attention-Enhanced   | `models/best.pt`                | 18.4 MB |
| YOLOv8n      | Ultralytics Lightweight Baseline | `models/best_8n.pt`             |  6.5 MB |
| RetinaNet    | Torchvision ResNet-50-FPN        | `models/retinanet_ppe_best.pth` |  136 MB |

## Checkpoint Distribution

Model checkpoints can be distributed through GitHub Releases:

[GitHub Repository](https://github.com/mariammmo/Safety_Helmet-Reflective_Jacket)

### ONNX Support

ONNX exports are also supported for deployment on non-PyTorch runtimes.

Example export:

```text
best.onnx
```

---



# 📄 License & Acknowledgments

## License

This project is distributed under the **MIT License**.

See the [LICENSE](LICENSE) file for details.

## Dataset

[Personal Protective Equipment (PPE) Dataset](https://www.kaggle.com/datasets/ndomalau/personal-protective-equipment-ppe-dataset)

* Curated by ndomalau.
* Hosted on Kaggle.
* Distributed under Open Data Commons, as specified in the project documentation.

## Frameworks

This project was built using:

* [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
* [PyTorch Torchvision](https://pytorch.org/vision/)
* [Streamlit](https://streamlit.io/)

---

# ⭐ Project Highlights

* Multi-architecture object detection benchmark.
* Explicit negative PPE supervision.
* Worker-level spatial compliance evaluation.
* YOLO11s attention-enhanced architecture.
* Lightweight YOLOv8n edge alternative.
* Classical RetinaNet baseline.
* Real-time Streamlit monitoring dashboard.
* Automated CSV audit logging.
* Quantitative accuracy and efficiency analysis.
* Designed for construction and industrial safety monitoring.

---

**Construction Site PPE Compliance Monitor**

*Computer Vision · Deep Learning · Object Detection · Safety Compliance · Model Benchmarking · Edge AI*
