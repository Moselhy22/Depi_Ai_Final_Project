# 🏋️ Training Guide

This guide explains how to train the models used in the **In-Car Violence Detection System**, including setup, execution, monitoring, and troubleshooting.

---

## ⚙️ Prerequisites

Before starting training, ensure the following:

### 1. Activate Environment

```bash
conda activate in-car-violence-detection
````

---

### 2. Preprocess Data

```bash
./scripts/run_preprocessing.sh
```

---

### 3. Verify GPU Availability

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

✅ Expected output:

```
True
```

---

## 🚨 Violence Detection Training

### ▶️ Quick Start

```bash
python -m src.training.train_violence \
    --train-dir data/processed/violence_in_car/train/clips \
    --val-dir data/processed/violence_in_car/val/clips
```

---

### ⚙️ Configuration

Modify training parameters in:

```
configs/train_violence.yaml
```

Example:

```yaml
training:
  epochs: 100
  batch_size: 4
  lr: 0.0001
  
  early_stopping:
    patience: 15
    
  loss:
    class_weights: [1.0, 2.0]
```

---

### 📊 Monitoring & Outputs

| Item       | Path                                         |
| ---------- | -------------------------------------------- |
| Logs       | `logs/violence/`                             |
| Best Model | `models/checkpoints/violence/best_model.pth` |

---

### 📈 Example Training Output

```
Epoch 1/100
  Batch 0/50, Loss: 0.6931
  Batch 10/50, Loss: 0.6123
...
Train - Loss: 0.5234, Acc: 0.7123, F1: 0.6543
Val   - Loss: 0.4891, Acc: 0.7456, F1: 0.7012
      - Precision: 0.7234, Recall: 0.6890
      - F1: 0.7012, AUC: 0.8123
✅ New best model saved!
```

---

## 🔫 Weapon Detection Training (YOLOv8)

### ▶️ Quick Start

```bash
python -m src.training.train_weapon
```

---

### ⚙️ Configuration

Edit:

```
configs/train_weapon.yaml
```

Example:

```yaml
model:
  variant: "n"   # Options: n, s, m, l

training:
  epochs: 100
  batch_size: 16
  imgsz: 640
```

---

### 📊 Monitoring & Outputs

YOLOv8 automatically logs results to:

```
models/checkpoints/weapon/
```

---

### 📈 Example Output

```
Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
100/100      2.1G      1.234      0.567      1.123         42        640

metrics/mAP50(B)      metrics/mAP50-95(B)
      0.8234              0.6543
```

---

## ⚡ Performance Optimization Tips

### 🧠 Memory Issues (CUDA OOM)

| Model    | Solution                          |
| -------- | --------------------------------- |
| Violence | Reduce `batch_size` to 2 or 1     |
| Violence | Reduce clip length (e.g., 16 → 8) |
| Weapon   | Reduce `batch_size` to 8          |
| Both     | Use smaller model variant         |

---

### 🚀 Slow Training

* Reduce `num_workers` in DataLoader
* Ensure **SSD storage** (avoid HDD)
* Use **mixed precision** (enabled by default)

---

### 📉 Overfitting

* Increase **dropout**
* Add more **data augmentation**
* Reduce model complexity
* Use **early stopping**

---

## 🔄 Resuming Training

### Violence Detection

⚠️ Not implemented yet
→ Save checkpoints manually

---

### Weapon Detection (YOLOv8)

```bash
yolo detect train model=path/to/last.pt resume
```

---

## 🖥️ Multi-GPU Training

### Violence Detection

❌ Not supported yet

---

### Weapon Detection

```bash
yolo detect train data=dataset.yaml model=yolov8n.pt device=0,1
```

---

## 🛠️ Troubleshooting

| Issue              | Solution              |
| ------------------ | --------------------- |
| CUDA out of memory | Reduce batch size     |
| No clips found     | Run preprocessing     |
| NaN loss           | Reduce learning rate  |
| No improvement     | Check dataset quality |

---

## ✅ Post-Training Steps

### 1. Evaluate Models

```bash
./scripts/run_evaluation.sh
```

---

### 2. Run Inference Test

```bash
python -m src.inference.real_time_detection --source 0
```

---

### 3. Save Progress

```bash
git add .
git commit -m "Training complete: violence + weapon models"
```

---

## 🎯 Summary

| Model              | Command          | Output           |
| ------------------ | ---------------- | ---------------- |
| Violence Detection | `train_violence` | `.pth` model     |
| Weapon Detection   | `train_weapon`   | YOLO `.pt` model |


