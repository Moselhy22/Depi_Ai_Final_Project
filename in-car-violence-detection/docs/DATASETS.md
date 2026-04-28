# 📊 Datasets Documentation

This document provides a comprehensive overview of all datasets used in the **In-Car Violence Detection System**. It includes dataset sources, structure, usage, and preprocessing details.

---

## 📁 Overview

The project utilizes **three datasets**:

1. **Violence in Car** – Primary dataset (in-car scenarios)
2. **SCVD (Smart-City CCTV Violence Detection)** – Secondary dataset (diverse environments)
3. **Guns & Knives CCTV Dataset** – Weapon detection dataset (YOLO training)

---

## 🚗 Dataset 1: Violence in Car

| Attribute | Details |
|----------|--------|
| **Source** | https://www.kaggle.com/datasets/xuantin/violence-in-car |
| **Author** | xuantin |
| **Type** | Video clips |
| **Format** | `.mp4`, `.avi` |
| **Storage Path** | `data/raw/violence-in-car/` |

### 🎯 Classes

| Class | Label | Description |
|------|------|------------|
| NonViolence | 0 | Normal in-car behavior |
| Violence | 1 | Physical violence inside vehicle |

### 🧠 Usage

- Primary dataset for training the violence detection model
- Closely matches real-world deployment conditions
- Used for both training and validation

### ⚙️ Preprocessing

- Frame extraction: **10 FPS**
- Resolution: **224 × 224**
- Clip generation:
  - 16-frame sequences
  - 8-frame stride

---

## 🏙️ Dataset 2: SCVD (Smart-City CCTV Violence Detection)

| Attribute | Details |
|----------|--------|
| **Source** | https://www.kaggle.com/datasets/toluwaniaremu/smartcity-cctv-violence-detection-dataset-scvd |
| **Reference Paper** | SSIVD-Net (Aremu et al., 2024) |
| **Type** | CCTV video footage |
| **Format** | `.mp4`, `.avi`, `.mov`, `.mkv` |
| **Storage Path** | `data/raw/scvd/` |

### 🎯 Classes

| Class | Label | Description |
|------|------|------------|
| NonViolence | 0 | Normal CCTV activity |
| Violence | 1 | Violent actions |
| Weapons | 2 | Presence of weapons |

### 🧠 Usage

- Secondary dataset for improving generalization
- Adds environmental diversity (CCTV vs in-car)
- Enables optional multi-task learning (violence + weapon detection)

### 📚 Citation

```bibtex
@inproceedings{scvd2024,
  author={Aremu, Toluwani and Zhiyuan, Li and Alameeri, Reem and Khan, Mustaqeem and Saddik, Abdulmotaleb El},
  title={SSIVD-Net: A Novel Salient Super Image Classification and Detection Technique for Weaponized Violence},
  booktitle={Intelligent Computing},
  year={2024},
  publisher={Springer Nature Switzerland}
}
````

---

## 🔫 Dataset 3: Guns & Knives Detection (CCTV)

| Attribute        | Details                                                                                                                                                                      |
| ---------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Source**       | [https://www.kaggle.com/datasets/kruthisb999/guns-and-knifes-detection-in-cctv-videos](https://www.kaggle.com/datasets/kruthisb999/guns-and-knifes-detection-in-cctv-videos) |
| **Author**       | kruthisb999                                                                                                                                                                  |
| **Type**         | CCTV video clips                                                                                                                                                             |
| **Format**       | `.mp4`, `.avi`, `.mov`                                                                                                                                                       |
| **Storage Path** | `data/raw/guns-knives-cctv/`                                                                                                                                                 |

### 🎯 Classes (YOLO Format)

| Class      | ID | Description       |
| ---------- | -- | ----------------- |
| guns       | 0  | Firearms          |
| knives     | 1  | Bladed weapons    |
| background | 2  | No weapon present |

### 📂 Structure

```
combined_gunsnknifes/
├── guns/
├── knives/
└── background/
```

### 🧠 Usage

* Training YOLOv8 for **weapon detection**
* Converted into YOLO annotation format
* Frame extraction: every 5 frames

### ⚠️ Important Note

This dataset **requires bounding box annotations**.

Current preprocessing uses placeholder labels. For production-level performance, use annotation tools such as:

* LabelImg
* CVAT
* Roboflow

---

## 📊 Data Splits

| Dataset         | Train | Validation | Test |
| --------------- | ----- | ---------- | ---- |
| Violence in Car | 70%   | 15%        | 15%  |
| SCVD            | 70%   | 15%        | 15%  |
| Guns & Knives   | 80%   | 10%        | 10%  |

---

## ⬇️ Download Instructions

### 1. Setup

* Create an account at: [https://www.kaggle.com](https://www.kaggle.com)
* Install Kaggle API:

```bash
pip install kaggle
```

### 2. Download Datasets

```bash
kaggle datasets download -d xuantin/violence-in-car
kaggle datasets download -d toluwaniaremu/smartcity-cctv-violence-detection-dataset-scvd
kaggle datasets download -d kruthisb999/guns-and-knifes-detection-in-cctv-videos
```

### 3. Directory Placement

Place datasets under:

```
data/raw/
```

---

## ⚙️ Preprocessed Data Structure

After preprocessing:

```
data/processed/
├── violence_in_car/
│   ├── train/clips/
│   ├── val/clips/
│   └── test/clips/
├── scvd/
│   ├── train/clips/
│   ├── val/clips/
│   └── test/clips/
└── guns_knives/
    └── yolo_format/
        ├── images/
        │   ├── train/
        │   ├── val/
        │   └── test/
        └── labels/
            ├── train/
            ├── val/
            └── test/
```

---

## ⚖️ License & Ethical Considerations

* Datasets are used strictly for **research and safety applications**
* Ensure compliance with **privacy laws and regulations**
* Avoid deployment in surveillance systems without **explicit consent**
* Use responsibly to enhance **public safety**, not intrusion

---

## ✅ Summary

| Dataset         | Purpose                 | Type         |
| --------------- | ----------------------- | ------------ |
| Violence in Car | Core model training     | In-car video |
| SCVD            | Generalization          | CCTV video   |
| Guns & Knives   | Object detection (YOLO) | CCTV video   |


