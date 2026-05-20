# System Architecture

## Overview

The In-Car Violence Detection System is a real-time multi-model pipeline that combines weapon detection and violence detection to evaluate threat levels in vehicle cabin environments.

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│              IN-CAR VIOLENCE DETECTION SYSTEM              │
│                         Version 1.0                         │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    INPUT: Video Stream                        │
│              (Webcam / Video File / CCTV Feed)              │
└─────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┼───────────────┐
              │               │               │
              ▼               ▼               ▼
    ┌─────────────────┐ ┌─────────────────┐ ┌─────────────┐
    │  FRAME BUFFER   │ │  FRAME BUFFER   │ │  THREAT     │
    │  (8 frames)     │ │  (1 frame)      │ │  EVALUATOR  │
    │  172x172 @ 8fps │ │  640x640        │ │             │
    └────────┬────────┘ └────────┬────────┘ └──────┬──────┘
             │                   │                  │
             ▼                   ▼                  │
    ┌─────────────────┐ ┌─────────────────┐        │
    │  VIOLENCE       │ │  WEAPON         │        │
    │  DETECTION      │ │  DETECTION      │        │
    │                 │ │                 │        │
    │  MoViNet A0     │ │  YOLOv8n        │        │
    │  Backbone       │ │  (Ultralytics)  │        │
    │                 │ │                 │        │
    │  MIL Classifier │ │  Classes:       │        │
    │  (Net object)   │ │  • Knife        │        │
    │                 │ │  • Gun          │        │
    │  Output:        │ │  • Sharp Object │        │
    │  Probability    │ │                 │        │
    │  [0.0 - 1.0]    │ │  Output:        │        │
    │                 │ │  Bounding Boxes │        │
    │  8-frame clips  │ │  + Labels       │        │
    │  Causal mode    │ │                 │        │
    │  (streaming)    │ │  Every frame    │        │
    └────────┬────────┘ └────────┬────────┘        │
             │                   │                  │
             │   Violence Score  │   Detections     │
             └─────────┬─────────┴────────┬─────────┘
                       │                  │
                       ▼                  ▼
              ┌─────────────────────────────────┐
              │      THREAT LEVEL LOGIC         │
              │                                 │
              │  ┌─────────────────────────┐    │
              │  │ Weapon + Violence > 0.6 │ → CRITICAL (RED)   │
              │  └─────────────────────────┘    │
              │  ┌─────────────────────────┐    │
              │  │ Violence > 0.6 only     │ → WARNING (ORANGE) │
              │  └─────────────────────────┘    │
              │  ┌─────────────────────────┐    │
              │  │ Weapon detected only    │ → WATCH (YELLOW)   │
              │  └─────────────────────────┘    │
              │  ┌─────────────────────────┐    │
              │  │ Violence > 0.3          │ → WATCH (YELLOW)   │
              │  └─────────────────────────┘    │
              │  ┌─────────────────────────┐    │
              │  │ Nothing detected        │ → NORMAL (GREEN)   │
              │  └─────────────────────────┘    │
              └─────────────────────────────────┘
                              │
                              ▼
              ┌─────────────────────────────────┐
              │     ALERT SYSTEM + OUTPUT       │
              │                                 │
              │  • Bounding boxes on weapons    │
              │  • Violence probability meter   │
              │  • Threat level banner          │
              │  • Color-coded alerts           │
              │  • FPS counter                  │
              │  • Console logging              │
              │                                 │
              │  Output: Annotated Video + Logs │
              └─────────────────────────────────┘
```

## Data Flow

1. **Input**: Video stream (webcam index 0, or video file path)
2. **Parallel Processing**:
   - **Weapon Branch**: Every frame → YOLOv8n → bounding boxes + class labels
   - **Violence Branch**: Every 4th frame (stride) → buffer 8 frames → VD-MIL → probability
3. **Threat Evaluation**: Combines both outputs using logic table
4. **Temporal Smoothing**: 30-frame buffer for violence scores (3 seconds at 10fps)
5. **Alert Debouncing**: 3-second cooldown between same-level alerts
6. **Visualization**: Draws all info on frame and outputs annotated video

## Model Specifications

### Weapon Detection (YOLOv8n)
| Attribute | Value |
|-----------|-------|
| Framework | Ultralytics YOLOv8n |
| Input | 640×640 RGB images |
| Output | Bounding boxes (x1,y1,x2,y2) + class + confidence |
| Classes | Knife, Gun, Sharp Object |
| Inference | Every frame |
| Speed | ~30fps on GTX 1650 |

### Violence Detection (VD-MIL)
| Attribute | Value |
|-----------|-------|
| Framework | PyTorch + MoViNet-pytorch |
| Backbone | MoViNet A0 (causal=True) |
| Classifier | MIL (Multi-Instance Learning) Net |
| Input | 8-frame clips, 172×172, 8fps |
| Preprocessing | BGR→RGB, resize 172×172, /255.0, NO ImageNet norm |
| Tensor Format | (Batch, Channels, Time, Height, Width) = (1, 3, 8, 172, 172) |
| Output | Violence probability [0.0 - 1.0] |
| Inference | Every stride frames (default 4) |
| Critical | Must call `clean_activation_buffers()` before each forward |

## Threat Level Definitions

| Level | Color | Condition | Action |
|-------|-------|-----------|--------|
| **NORMAL** | 🟢 Green | No threats | Continue monitoring |
| **WATCH** | 🟡 Yellow | Weapon present OR suspicious activity | Increase vigilance |
| **WARNING** | 🟠 Orange | Violence detected | Alert operator |
| **CRITICAL** | 🔴 Red | Weapon + Violence together | Immediate emergency response |

## System Requirements

### Minimum
- GPU: NVIDIA GTX 1650 (4GB VRAM)
- RAM: 8GB
- Storage: 10GB (models + data)
- OS: Ubuntu 20.04+ / Windows 10+

### Recommended
- GPU: NVIDIA RTX 3060+ (8GB+ VRAM)
- RAM: 16GB
- Storage: 50GB
- OS: Ubuntu 22.04 LTS

## Performance Benchmarks (v1.0)

| Scenario | Resolution | FPS | GPU Usage |
|----------|-----------|-----|-----------|
| Video file (1280×720) | 720p | ~30fps | ~85% |
| Webcam (640×480) | 480p | ~35fps | ~70% |
| Video file (1920×1080) | 1080p | ~20fps | ~95% |

---
*Architecture Version 1.0 | Last Updated: May 20, 2026*
