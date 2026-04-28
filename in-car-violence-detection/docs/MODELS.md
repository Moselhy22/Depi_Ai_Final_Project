# 🧠 Model Architectures

This document provides a detailed overview of the machine learning models used in the **In-Car Violence Detection System**, including architecture design, configurations, and training strategies.

---

## 🚨 Violence Detection Model

### 🏗️ Architecture: `InCarViolenceDetector`

A hybrid deep learning architecture combining:
- **Convolutional Neural Networks (CNNs)** for spatial feature extraction  
- **Long Short-Term Memory (LSTM)** for temporal modeling  
- **Attention mechanisms** for improved focus on relevant spatial and temporal features  

---

### 🔩 Model Components

#### 1. 🖼️ CNN Backbone (ResNet50)

| Property | Value |
|----------|------|
| Input | (3, 224, 224) |
| Output | (2048, 7, 7) |
| Pretrained | ImageNet |
| Modification | Final pooling and FC layers removed |

✔ Extracts high-level spatial features from each frame.

---

#### 2. 🎯 Spatial Attention Module

- Focuses on **important regions** within each frame.

**Architecture:**
```python id="spatial_att"
Conv2d(2048 → 512, kernel=1)
→ ReLU
→ Conv2d(512 → 1, kernel=1)
→ Sigmoid
````

| Input        | Output                      |
| ------------ | --------------------------- |
| (2048, 7, 7) | Attention-weighted features |

---

#### 3. 🔄 Bidirectional LSTM

* Captures **temporal dependencies** across video frames.

| Parameter   | Value        |
| ----------- | ------------ |
| Input       | (B, T, 2048) |
| Output      | (B, T, 1024) |
| Hidden size | 512          |
| Layers      | 2            |
| Dropout     | 0.5          |

✔ Processes frame sequences in both forward and backward directions.

---

#### 4. ⏱️ Temporal Attention Module

* Identifies the most **important time steps** in a sequence.

**Architecture:**

```python id="temporal_att"
Linear(1024 → 128)
→ Tanh
→ Linear(128 → 1)
→ Softmax
```

| Input        | Output                   |
| ------------ | ------------------------ |
| (B, T, 1024) | Context vector (B, 1024) |

---

#### 5. 🧮 Classification Head

**Architecture:**

```python id="classifier_head"
Linear(1024 → 512)
→ ReLU → Dropout(0.5)
→ Linear(512 → 256)
→ ReLU → Dropout(0.5)
→ Linear(256 → 2)
```

✔ Produces final **violence vs non-violence prediction**.

---

### 📥 Input / 📤 Output

| Component          | Shape                  | Description          |
| ------------------ | ---------------------- | -------------------- |
| Input              | (B, 16, 3, 224, 224)   | 16-frame video clips |
| Output             | (B, 2)                 | Class probabilities  |
| Temporal Attention | (B, 16, 1)             | Frame importance     |
| Spatial Attention  | (B, 1, 7, 7) per frame | Region importance    |

---

### 📊 Model Size

* **Total Parameters**: ~42 Million
* **Trainable**: 100%

---

## 🔫 Weapon Detection Model

### ⚡ Architecture: YOLOv8

Real-time object detection using the **Ultralytics YOLOv8 framework**.

---

### 📦 Model Variants

| Variant | Parameters | Speed     | Use Case         |
| ------- | ---------- | --------- | ---------------- |
| YOLOv8n | 3.2M       | ⚡ Fastest | Edge devices     |
| YOLOv8s | 11.2M      | 🚀 Fast   | Balanced         |
| YOLOv8m | 25.9M      | ⚖️ Medium | Higher accuracy  |
| YOLOv8l | 43.7M      | 🐢 Slower | Maximum accuracy |

---

### ⚙️ Current Configuration (YOLOv8n)

| Parameter            | Value                  |
| -------------------- | ---------------------- |
| Input Size           | 640 × 640              |
| Classes              | gun, knife, background |
| Confidence Threshold | 0.5                    |
| IoU (NMS)            | 0.7                    |

---

### 🏋️ Training Configuration

| Parameter     | Value  |
| ------------- | ------ |
| Epochs        | 100    |
| Batch Size    | 16     |
| Optimizer     | Adam   |
| Learning Rate | 0.001  |
| Weight Decay  | 0.0005 |

---

### 🎨 Data Augmentation

* Mosaic: 1.0
* MixUp: 0.2
* Rotation: ±15°
* Scale: 0.5 – 1.5
* HSV Augmentation:

  * Hue: ±1.5%
  * Saturation: ±70%
  * Value: ±40%

---

## ⚖️ Model Comparison

| Feature      | Violence Detection      | Weapon Detection        |
| ------------ | ----------------------- | ----------------------- |
| Task         | Action Recognition      | Object Detection        |
| Input        | Video clips (16 frames) | Single frames           |
| Architecture | CNN + LSTM + Attention  | YOLOv8                  |
| Output       | Binary classification   | Bounding boxes + labels |
| Speed        | ~10 FPS                 | ~30+ FPS                |
| Parameters   | ~42M                    | 3M – 44M                |

---

## 🧪 Training Best Practices

### 🚨 Violence Detection

* Use **class weights** → `[1.0, 2.0]`
* Apply **gradient clipping** → `max_norm = 1.0`
* Use **OneCycleLR scheduler**
* Enable **early stopping** based on validation F1-score

---

### 🔫 Weapon Detection

* Start with **YOLOv8n** for faster experimentation
* Apply strong **data augmentation**
* Ensure accurate **bounding box annotations**
* Validate under different:

  * Lighting conditions
  * Camera angles
  * Resolutions

---

## 🚀 Future Improvements

* [ ] Integrate pose estimation (ST-GCN)
* [ ] Multi-camera fusion system
* [ ] Audio analysis (e.g., scream detection)
* [ ] Edge optimization using TensorRT
* [ ] Model quantization (INT8)

---

## ✅ Summary

| Model                 | Purpose                | Strength               |
| --------------------- | ---------------------- | ---------------------- |
| InCarViolenceDetector | Detect violent actions | Temporal understanding |
| YOLOv8                | Detect weapons         | Real-time performance  |

---

```

---


