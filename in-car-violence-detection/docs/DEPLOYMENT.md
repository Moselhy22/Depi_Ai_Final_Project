# 🚀 Deployment Guide

This guide explains how to deploy the **In-Car Violence Detection System** in real-world scenarios, including setup, optimization, and system integration.

---

## 🖥️ System Requirements

| Component | Minimum | Recommended |
|----------|--------|-------------|
| GPU | NVIDIA GTX 1650 | RTX 3060+ |
| RAM | 8 GB | 16 GB |
| Storage | 10 GB SSD | 50 GB SSD |
| Camera | 720p webcam | 1080p IP camera |
| OS | Ubuntu 20.04 | Ubuntu 22.04 |

---

## ⚙️ Deployment Modes

### 💻 Option 1: Laptop / Development Mode

Run real-time detection using a webcam:

```bash
conda activate in-car-violence-detection

python -m src.inference.real_time_detection \
    --source 0 \
    --violence-model models/checkpoints/violence/best_model.pth \
    --weapon-model models/checkpoints/weapon/best.pt
````

---

### 🎥 Option 2: Video File (Testing Mode)

```bash id="video_test"
python -m src.inference.real_time_detection \
    --source path/to/video.mp4 \
    --violence-model models/checkpoints/violence/best_model.pth \
    --weapon-model models/checkpoints/weapon/best.pt \
    --output results/output.mp4
```

---

### 📷 Option 3: Multi-Camera Deployment

```bash id="multi_cam"
python -m src.inference.real_time_detection \
    --source 0 2 4 \
    --violence-model models/checkpoints/violence/best_model.pth \
    --weapon-model models/checkpoints/weapon/best.pt
```

---

## 📦 Model Export

### 🔄 Export to ONNX (Cross-Platform)

```python id="onnx_export"
from ultralytics import YOLO

model = YOLO("models/checkpoints/weapon/best.pt")
model.export(format="onnx")
```

⚠️ Note: Violence model requires custom export implementation.

---

### ⚡ Export to TensorRT (NVIDIA GPUs)

```python id="tensorrt_export"
model = YOLO("models/checkpoints/weapon/best.pt")
model.export(format="engine")
```

✔ Significantly improves inference speed on NVIDIA hardware.

---

## 🚨 Alert System

### Threat Levels

| Level | Status      | Trigger                     | Action                     |
| ----- | ----------- | --------------------------- | -------------------------- |
| 0     | 🟢 Normal   | No detection                | Continue monitoring        |
| 1     | 🟡 Watch    | Suspicious activity         | Log event                  |
| 2     | 🟠 Warning  | Violence OR weapon detected | Audio alert + recording    |
| 3     | 🔴 Critical | Violence + weapon           | Alarm + emergency response |

---

### ⚙️ Custom Alerts

Modify alert behavior in:

```id="alert_path"
src/inference/real_time_detection.py
```

Example:

```python id="custom_alert"
if level == self.THREAT_CRITICAL:
    send_emergency_call()
    activate_hazard_lights()
    lock_vehicle_doors()
```

---

## ⚡ Performance Optimization

### 1. Reduce Input Resolution

```python id="reduce_res"
self.frame_size = (160, 160)
```

---

### 2. Reduce Clip Length

```yaml id="clip_reduce"
clip_length: 8
```

---

### 3. Use Lightweight Model

```yaml id="yolo_fast"
model:
  variant: "n"
```

---

### 4. Frame Skipping

```python id="frame_skip"
if frame_count % 2 == 0:
    process_frame(frame)
```

---

## 🚗 Vehicle Integration (Advanced)

### 🔌 CAN Bus Communication

```python id="can_bus"
import can

bus = can.interface.Bus(channel='can0', bustype='socketcan')
msg = can.Message(arbitration_id=0x123, data=[0x01, 0x02])
bus.send(msg)
```

---

### 📍 GPS Integration

```python id="gps_code"
import gps

location = gps.get_current_location()
send_alert(f"Emergency at: {location}")
```

---

## 📊 Monitoring & Logging

### 🔍 View Logs

```bash id="logs_live"
tail -f logs/violence/*.log
```

```bash id="logs_specific"
cat logs/violence/IntegratedDetection_*.log
```

---

### 📈 TensorBoard (Training Metrics)

```bash id="tensorboard_run"
tensorboard --logdir models/checkpoints/weapon/
```

---

## 🔐 Security Considerations

### 🔒 Data Privacy

* Encrypt stored video data
* Anonymize faces where required
* Comply with local privacy laws

---

### 🛡️ Model Security

* Sign model files
* Verify integrity (checksums)
* Protect against adversarial inputs

---

### 🚧 Physical Security

* Use tamper-resistant camera hardware
* Secure storage for models and logs
* Encrypt communication channels

---

## 🛠️ Troubleshooting

| Issue               | Solution                       |
| ------------------- | ------------------------------ |
| Camera not detected | Run `ls /dev/video*`           |
| Low FPS             | Reduce resolution / model size |
| False positives     | Tune confidence thresholds     |
| Crashes             | Check GPU memory usage         |

---

## 🆘 Support

If you encounter issues:

* Check logs in `logs/`
* Review documentation in `docs/`
* Open a GitHub issue

---

## ✅ Deployment Summary

| Mode      | Use Case         | Command              |
| --------- | ---------------- | -------------------- |
| Webcam    | Real-time demo   | `--source 0`         |
| Video     | Testing          | `--source video.mp4` |
| Multi-Cam | Advanced systems | `--source 0 2 4`     |

