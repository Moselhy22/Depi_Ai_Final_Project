# 🚗 AI-Based Driver Monitoring System

> Real-time driver distraction detection and face authentication using YOLOv8 and ArcFace.

---

## 📋 Table of Contents

**Part 1 — Distraction Detection**
- [Project Overview](#project-overview)
- [Features](#features)
- [System Architecture](#system-architecture)
- [Target Classes](#target-classes)
- [Distraction Model Performance](#distraction-model-performance)
- [Distraction Datasets](#distraction-datasets)

**Part 2 — Face Recognition & Authentication**
- [Face Recognition Overview](#face-recognition-overview)
- [ArcFace vs AdaFace Evaluation](#arcface-vs-adaface-evaluation)

**General**
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Installation & Setup](#installation--setup)
- [Usage](#usage)
- [Future Improvements](#future-improvements)

---

# Part 1 — Distraction Detection

## Project Overview

An intelligent **Driver Monitoring System** that uses Computer Vision and Deep Learning to detect distracted driving behaviors in real time through a camera feed.

The system combines two main components:

- **Distraction Detection** — a custom-trained YOLOv8n model that identifies dangerous objects (phone, cigarette, vape, bottle) in the driver's view.
- **Face Authentication** — an ArcFace-based recognition system with anti-spoofing protection to verify driver identity before starting a session.

---

## Features

- 🔍 Real-time object detection via YOLOv8
- 👤 Face registration and login with ArcFace embeddings
- 🛡️ Lightweight anti-spoofing (blur, brightness, texture, edge, color checks)
- 📹 Live MJPEG stream with annotated bounding boxes
- 📝 Automatic distraction logging per driver session
- 📊 Activity and distraction history logs
- 🌐 Flask REST API backend with CORS support

---

## System Architecture

```
Camera Feed
     │
     └──► Distraction Detection (YOLOv8n)
               ├── Object Detection (Phone, Cigarette, Vape, Bottle)
               ├── Sustained Frame Confirmation (6 frames)
               ├── Annotated Frame Output
               └── Auto Distraction Logging
```

---

## Target Classes

| Class ID | Class Name | Description |
|----------|------------|-------------|
| 0 | Bottle | Beverage containers |
| 1 | Cigarette | Smoking while driving |
| 2 | Phone | Mobile phone usage |
| 3 | Vape | Vaping device |

---

## Distraction Model Performance

### ✅ Final Model — YOLOv8n (Custom Fine-Tuned)

Trained on a clean, balanced dataset (~5,500 images) for 70 epochs.

| Class | Precision | Recall | mAP50 | mAP50-95 |
|-------|-----------|--------|-------|----------|
| Bottle | 0.907 | 0.870 | 0.941 | 0.806 |
| Phone | 0.809 | 0.868 | 0.889 | 0.695 |
| Vape | 0.672 | 0.571 | 0.636 | 0.314 |
| Cigarette | 0.575 | 0.349 | 0.420 | 0.139 |
| **Overall** | **0.741** | **0.665** | **0.722** | **0.489** |

> Cigarette detection is the weakest class due to small object size, occlusions, and motion blur. Dataset expansion is planned.

---

### What We Tried Before

#### Approach 1 — EfficientNetB3 Classification

Our first attempt used **EfficientNetB3** fine-tuned on the [State Farm Distracted Driver Detection](https://www.kaggle.com/datasets/rightway11/state-farm-distracted-driver-detection) dataset as a **classification** model (not detection).

**Training config:** 30 epochs · batch 8 · image size 300 · lr 0.0003

| Metric | Value |
|--------|-------|
| Train Accuracy | 99.73% |
| Validation Accuracy | 99.53% |

**Per-class results on held-out test set:**

| Class | Precision | Recall | F1 |
|-------|-----------|--------|----|
| Safe Driving | 1.000 | 1.000 | 1.000 |
| Texting Right | 1.000 | 1.000 | 1.000 |
| Smoking | 0.986 | 1.000 | 0.993 |
| Phone Call Right | 1.000 | 1.000 | 1.000 |
| Texting Left | 1.000 | 1.000 | 1.000 |
| Phone Call Left | 1.000 | 1.000 | 1.000 |
| Headphones | 1.000 | 1.000 | 1.000 |
| Drinking | 1.000 | 0.981 | 0.990 |
| Talking Passenger | 1.000 | 1.000 | 1.000 |
| **Overall** | **0.998** | **0.998** | **0.998** |

Despite near-perfect validation accuracy, the model **failed completely on real webcam input** — it was trained on a classification dataset with fixed camera angles and posed images, making it unable to generalize to live driving conditions.

---

#### Approach 2 — YOLOv8n (First YOLO Run)

We switched to detection with YOLOv8n, trained on a larger merged Roboflow dataset (~44K images) covering 3 classes, then resumed training from a checkpoint.

**Training config:** 50 epochs (resumed) · batch 4 · image size 512 · AdamW · patience 7

| Metric | Value |
|--------|-------|
| mAP50 | 0.704 |
| mAP50-95 | 0.546 |
| Precision | 0.745 |
| Recall | 0.644 |

The dataset was too noisy and imbalanced — similar-looking classes caused confusion. This led us to rebuild the dataset from scratch with cleaner, more focused sources.

---

## Distraction Datasets

### Final Model Datasets

| Class | Dataset |
|-------|---------|
| Bottle | [Beverage Containers](https://universe.roboflow.com/roboflow-universe-projects/beverage-containers-3atxb) |
| Cigarette | [Smoker YOLO](https://universe.roboflow.com/cigaretteple-7m0hn/smoker-yolo/dataset/1) · [Cigarette Detection](https://universe.roboflow.com/cigarette-lqptw/cigarette-gpofi/dataset/1) |
| Vape | [Vaping Detection](https://universe.roboflow.com/tiara-fb7pp/vaping-ulrul/dataset/1) |
| Phone | [Mobile Phone Detection](https://universe.roboflow.com/exam-detection-a9bsf/mobile-phone-detection-mtsje/dataset/1) |

**Final dataset distribution:**

| Class | Images |
|-------|--------|
| Bottle | ~1,000 |
| Cigarette | ~2,500 |
| Phone | ~1,000 |
| Vape | ~1,000 |

All datasets were cleaned, balanced, and split 80/10/10 (train/val/test).

---

### First YOLO Run Datasets

| Class | Dataset |
|-------|---------|
| Bottle & Cups | [Bottles and Cups Detection](https://universe.roboflow.com/ai-object-detection/bottles-and-cups-detection-6u8tg) |
| Cigarette & Vape | [Cigarette Vape Detection](https://universe.roboflow.com/takoyati/cigarette-vape-detection) |
| Phone | [Phone Dataset](https://universe.roboflow.com/cap-ybmfa/phone-p9wzw) · [Phone Detection](https://universe.roboflow.com/phone-detection-dmdak/phone-f7l35) |
| Headphones | [Headphones Dataset](https://universe.roboflow.com/workspace-w5tce/headphones-746cn) |
| Food | [NutraCal Food Detection](https://universe.roboflow.com/object-detection-vpvcm/nutracal-food-detection) |

---

### EfficientNet Dataset

| Dataset | Source |
|---------|--------|
| State Farm Distracted Driver Detection | [Kaggle](https://www.kaggle.com/datasets/rightway11/state-farm-distracted-driver-detection) |

---

# Part 2 — Face Recognition & Authentication

## Face Recognition Overview

The authentication system uses **ArcFace** (via InsightFace `buffalo_l`) to verify driver identity before each session.

### Authentication Architecture

```
Camera Feed
     │
     └──► Face Authentication (InsightFace / ArcFace)
               ├── Face Detection (YOLOv8m-face)
               ├── Anti-Spoof Check
               ├── Embedding Extraction (ArcFace)
               └── Cosine Similarity Match
```

### Registration Pipeline

1. Capture user image via webcam
2. Detect the largest face using YOLOv8m-face
3. Crop and extract facial embedding using ArcFace
4. Save embedding as `.npy` and store metadata in `users_db.json`

### Login Pipeline

1. Capture live image
2. Detect face
3. Run anti-spoofing checks (blur · brightness · texture · edge density · color variance)
4. Extract ArcFace embedding
5. Compute cosine similarity against stored embedding
6. Authenticate if similarity > 0.6

---

## ArcFace vs AdaFace Evaluation

We compared both models on a held-out face dataset using balanced genuine/impostor pairs and cosine similarity scoring. The evaluation script is at `Src/Final Evaluation ArcFace vs AdaFace.py`.

### Evaluation Metrics

| Metric | Description |
|--------|-------------|
| AUC | Area under the ROC curve (higher = better) |
| EER | Equal Error Rate (lower = better) |
| Best Threshold | Cosine similarity cutoff for maximum accuracy |

### Results Summary

| Model | AUC | Result |
|-------|-----|--------|
| **ArcFace** | **~0.99** | ✅ Excellent separability — model works correctly |
| AdaFace | ~0.55 | ❌ Near random — failed to learn class 0, threshold unreliable |

> AdaFace's poor AUC does not mean it is inherently worse — the result reflects that it was not properly configured or trained for our specific setup. ArcFace gave stable, reliable embeddings in our real-world tests.

### Why ArcFace Was Selected

- Higher embedding separability (AUC ~0.99)
- More stable cosine similarity distribution between genuine and impostor pairs
- Better robustness under varied lighting and face angles
- Reliable real-time performance with `det_size=320` on CPU

---

## Tech Stack

| Category | Tools |
|----------|-------|
| Language | Python 3.11.5 |
| Detection | YOLOv8 (Ultralytics) |
| Face Recognition | InsightFace (ArcFace / buffalo_l) |
| Face Detection | YOLOv8m-face |
| Computer Vision | OpenCV, Pillow |
| Backend | Flask, Flask-CORS |
| Experiment Tracking | MLflow 2.12.1 |
| Data | NumPy, PyYAML, scikit-learn |
| Hardware | NVIDIA GTX 1650 (CUDA) |

---

## Project Structure

```
DEPI GP/
│
├── backend/
│   ├── app.py                  # Flask backend (API + MJPEG stream)
│   ├── distraction.py          # YOLOv8 detection module
│   ├── recognition.py          # Face registration & login
│   └── spoof.py                # Anti-spoofing checks
│
├── frontend/
│   └── index.html              # Frontend UI
│
├── Src/
│   ├── YoloTrain3.py                              # YOLO training script
│   ├── divide_data.py                             # Dataset splitting utility
│   ├── Face_Detection_Yolo.py                     # Face cropping using YOLOv8m-face
│   └── Final Evaluation ArcFace vs AdaFace.py    # Face model comparison
│
├── models/
│   ├── detection_model/
│   │   └── yolov8m-face.pt
│   └── recognistion_models/
│       └── Arcface Model/
│           └── models/
│               └── buffalo_l/
│
├── data/
│   ├── users/                  # Per-user face embeddings & images
│   ├── users_db.json           # Registered users database
│   ├── activity_log.json
│   └── distraction_log.json
│
├── requirements.txt
├── README.md
└── mlruns/                     # MLflow experiment runs
```

---

## Installation & Setup

### Prerequisites

- Python 3.11.5
- CUDA-compatible GPU (optional, CPU works too)
- Webcam

### 1. Clone the repository

```bash
git clone https://github.com/Moselhy22/Depi_Ai_Final_Project.git
cd Depi_Ai_Final_Project
git checkout Aya
```

### 2. Create a virtual environment

```bash
python -m venv A_env
A_env\Scripts\activate        # Windows
# source A_env/bin/activate   # Linux/Mac
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download the ArcFace model

Download the `buffalo_l` model and place it at:

```
models/recognistion_models/Arcface Model/models/buffalo_l/
```

### 5. Update model paths

In `backend/distraction.py`:
```python
YOLO_PATH = r"path\to\models\yolov8n\yolov8n_final_ft_resume\best.pt"
```

In `backend/recognition.py`:
```python
LOCAL_MODEL = r"path\to\models\recognistion_models\Arcface Model"
```

### 6. Run the server

```bash
python backend/app.py
```

The server starts at `http://localhost:5000`.

---

## Usage

### Register a Driver

```
POST /register
Body: { "name": "driver_name", "image": "<base64_image>" }
```

### Login

```
POST /login
Body: { "name": "driver_name", "image": "<base64_image>" }
```

### Live Distraction Stream

```
GET /distraction_stream
```

### Detect Distraction (single frame)

```
POST /detect_distraction
Body: { "image": "<base64_image>" }
```

### Get Logs

```
GET /get_activity_log
GET /get_distraction_log
```

---

## Future Improvements

- [ ] Seatbelt detection
- [ ] Deep learning anti-spoofing model
- [ ] Night driving dataset augmentation
- [ ] Edge device deployment (Raspberry Pi / Jetson)
- [ ] Mobile app support

---

## License

This project was developed as a graduation project. All rights reserved.
