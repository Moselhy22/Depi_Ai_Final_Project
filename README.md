# In-Cabin Driver Drowsiness Detection System — V2

A real-time **Driver Monitoring System (DMS)** that detects driver drowsiness using a dashboard-mounted webcam. Built with **YOLO11n** and **MediaPipe FaceMesh** running in a hybrid ensemble pipeline for maximum robustness across different lighting conditions, face angles, and with or without glasses.

---

## Features

- **Real-time detection** at 30+ FPS using NVIDIA GPU (CUDA)
- **Hybrid ensemble pipeline** — YOLO11n + MediaPipe geometric analysis running simultaneously
- **3-tier alert system** with audio + visual alerts:
  - `MICROSLEEP` — Eyes closed for > 1.5 seconds
  - `DROWSY` — PERCLOS > 70% over a rolling 3-second window
  - `FATIGUED` — 3+ yawns detected within 60 seconds
- **Glasses-bias correction** — OR-gate ensemble prevents false negatives when not wearing glasses
- **SQLite logging** — All alert events saved locally for post-session analysis
- **Windows-optimized** — Handles paging file and VRAM constraints on consumer hardware

---

## Model Performance

Trained on 6,803 images (5,796 train / 509 val / 498 test) from the CTU Drowsiness Dataset.

| Metric | Score |
|---|---|
| **mAP@50** | **98.4%** |
| **mAP@50-95** | **83.0%** |
| **Precision** | **95.9%** |
| **Recall** | **95.0%** |
| **Epochs trained** | 112 / 300 (early stopping) |
| **Inference speed** | ~2.8 ms/frame |

---

## Project Structure

```
New_Project/
│
├── detect.py               ← Real-time webcam detection engine
├── requirements.txt        ← Python dependencies
├── .gitignore
│
├── scripts/
│   ├── train.py            ← YOLO11n training (300 epochs, BCE loss)
│   ├── evaluate.py         ← Evaluation on held-out test split
│   └── download_dataset.py ← Roboflow dataset downloader
│
├── data/                   ← Dataset (NOT included in repo — see below)
│   ├── data.yaml           ← YOLO dataset config (included)
│   ├── train/images/       ← 5,796 training images
│   ├── valid/images/       ← 509 validation images
│   └── test/images/        ← 498 test images
│
├── models/                 ← Trained weights (NOT included — see below)
│   └── best.pt             ← YOLO11n fine-tuned weights (15.2 MB)
│
├── logs/
│   └── drowsiness_log.db   ← SQLite alert event log (auto-created)
│
├── run.bat                 ← Windows: launch live detection
├── train.bat               ← Windows: start training
├── evaluate.bat            ← Windows: evaluate on test split
├── download.bat            ← Windows: download dataset
└── install.bat             ← Windows: install all dependencies
```

---

## Setup & Installation

### Requirements
- Windows 10/11
- Python 3.11.x
- NVIDIA GPU with CUDA 12.1 (for GPU acceleration)
- ~600 MB free disk space for the dataset

### Step 1 — Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/drowsiness-detection-v2.git
cd drowsiness-detection-v2
```

### Step 2 — Create a Virtual Environment
```bash
python -m venv venv
venv\Scripts\activate
```

### Step 3 — Install Dependencies

**Option A (Windows one-click):**
```
Double-click install.bat
```

**Option B (Manual):**
```bash
# Install PyTorch with CUDA 12.1 support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install remaining packages
pip install -r requirements.txt
```

> **CPU-only systems:** Replace the PyTorch install line with:
> ```bash
> pip install torch torchvision torchaudio
> ```
> Detection will still work but will be slower (~10 FPS).

---

## Loading the Dataset

The dataset is **not included in the repository** due to its size (523 MB). You must download it before training.

### Dataset Details
- **Name:** CTU Drowsiness Detection
- **Source:** [Roboflow Universe](https://universe.roboflow.com/ctufinalthesis/drowsinessdetectionyolov8-test2/dataset/6)
- **License:** CC BY 4.0
- **Classes:** `Awake` (0), `Drowsy` (1)
- **Format:** YOLOv11 (pre-formatted — no conversion needed)

### Download Instructions

**Option A — Automatic (requires a free Roboflow account):**

1. Sign up for a free account at [roboflow.com](https://roboflow.com)
2. Go to your [Account Settings → API Key](https://app.roboflow.com/settings/api)
3. Copy your API key
4. Run:

```bash
python scripts/download_dataset.py --api-key YOUR_ROBOFLOW_API_KEY
```

Or on Windows:
```
Double-click download.bat
# (edit download.bat to add your API key first)
```

**Option B — Manual download:**

1. Visit the dataset URL: https://universe.roboflow.com/ctufinalthesis/drowsinessdetectionyolov8-test2/dataset/6
2. Click **Export Dataset** → Choose format **YOLOv11**
3. Download and extract into the `data/` folder
4. Ensure the structure matches:
   ```
   data/
   ├── data.yaml
   ├── train/images/  and  train/labels/
   ├── valid/images/  and  valid/labels/
   └── test/images/   and  test/labels/
   ```

---

## Loading the Trained Weights

The pre-trained weights (`models/best.pt`) are **not included** in the repository because binary `.pt` files are not ideal for version control.

### Option A — Download Pre-trained Weights
> *(Upload your `best.pt` to a GitHub Release or Google Drive and link it here)*
```bash
# Example if hosted on GitHub Releases:
# Download best.pt from the Releases page and place it at:
# models/best.pt
```

### Option B — Train from Scratch
If you have the dataset downloaded, simply run:
```
Double-click train.bat
```
Or:
```bash
python scripts/train.py
```
Training takes approximately **1.5–3 hours** on an NVIDIA GTX 1660 Super. Best weights are automatically saved to `models/best.pt`.

---

## Running the System

### Live Webcam Detection
```
Double-click run.bat
```
Or:
```bash
python detect.py                   # Default camera (index 0)
python detect.py --source 1        # Secondary/USB camera
python detect.py --no-sound        # Disable audio alerts
python detect.py --conf 0.4        # Higher confidence threshold
```

### Evaluate on Test Split
```
Double-click evaluate.bat
```
Or:
```bash
python scripts/evaluate.py
```
Reports Precision, Recall, mAP@50, mAP@50-95, and saves confusion matrix + PR curves to `runs/detect/evaluation/`.

---

## How It Works

### Detection Pipeline
```
Webcam Frame
     │
     ├──► YOLO11n ────────────────► Bounding boxes (Awake / Drowsy)
     │
     └──► MediaPipe FaceMesh ─────► EAR + MAR (geometric eye/mouth ratio)
                                          │
                               Ensemble OR Gate
                             (YOLO closed OR MediaPipe closed)
                                          │
                                  PERCLOS Buffer
                              (90-frame rolling window)
                                          │
                          ┌───────────────┼──────────────┐
                     MICROSLEEP        DROWSY         FATIGUED
                     EAR<0.24         PERCLOS>70%    ≥3 yawns/60s
                     for >1.5s        over 3 sec
```

### Key Metrics
| Metric | Formula | Threshold |
|---|---|---|
| **EAR** (Eye Aspect Ratio) | `(A+B) / (2×C)` | < 0.24 = closed |
| **MAR** (Mouth Aspect Ratio) | `(A+B+C) / (2×D)` | > 0.55 = yawning |
| **PERCLOS** | Mean of last 90 frames | > 70% = drowsy |

### Why Two Models?
YOLO11n can develop dataset bias (e.g., faces with glasses look different). MediaPipe uses pure 3D geometry — immune to bias. The ensemble OR-gate means neither model alone can cause a false negative.

---

## Training Details

| Parameter | Value |
|---|---|
| Base model | `yolo11n.pt` (COCO pre-trained) |
| Epochs | 300 max (stopped at 112 via early stopping) |
| Patience | 25 |
| Batch size | 8 |
| Image size | 640×640 |
| Optimizer | AdamW |
| Classification loss | Binary Cross-Entropy (BCE, `cls=0.5`) |
| Box loss | `box=7.5` |
| DFL loss | `dfl=1.5` |
| `workers` | 0 (required on Windows to prevent WinError 1455) |

---

## Alert Log Schema

All alerts are stored in `logs/drowsiness_log.db` (SQLite):

```sql
CREATE TABLE events (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp  TEXT,     -- e.g. "2026-05-19 21:09:10"
    alert_type TEXT,     -- "MICROSLEEP", "DROWSY", or "FATIGUED"
    duration_s REAL,     -- Duration of the event in seconds
    notes      TEXT      -- Full alert message string
);
```

Query example:
```sql
SELECT timestamp, alert_type, duration_s
FROM events
ORDER BY timestamp DESC
LIMIT 20;
```

---

## Hardware Recommendations

| Component | Minimum | Recommended |
|---|---|---|
| GPU | Any NVIDIA with CUDA | GTX 1660 Super or better |
| VRAM | 4 GB | 6 GB |
| RAM | 8 GB | 16 GB |
| Camera | Any USB webcam | 1080p at 30fps |

---

## License

Dataset: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) — CTUFinalThesis via Roboflow Universe

Code: MIT License

---

## Acknowledgements

- [Ultralytics YOLO11](https://github.com/ultralytics/ultralytics)
- [MediaPipe FaceMesh](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker)
- [CTU Drowsiness Dataset](https://universe.roboflow.com/ctufinalthesis/drowsinessdetectionyolov8-test2)
