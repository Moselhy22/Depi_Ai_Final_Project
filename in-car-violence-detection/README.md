# 🚗 In-Car Violence Detection System

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2.1-ee4c2c.svg)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-11.8-green.svg)](https://developer.nvidia.com/cuda)

&gt; **Real-time violence and weapon detection system for vehicle cabin monitoring using computer vision and deep learning.**

---

## 📋 Table of Contents

- [Features](#-features)
- [Datasets](#-datasets)
- [Installation](#-installation)
- [Project Structure](#-project-structure)
- [Training](#-training)
- [License](#-license)

---

## ✨ Features

| Feature | Description | Status |
|---------|-------------|--------|
| **Violence Detection** | Detects physical assault (punching, slapping, choking, grappling) | 🔄 In Progress |
| **Weapon Detection** | Detects knives, guns, and sharp objects in real-time | 🔄 In Progress |
| **Alert System** | Tiered alerts (Watch → Warning → Critical → Emergency) | 🔄 In Progress |

---

## 📊 Datasets

| Dataset | Source | Type | Usage |
|---------|--------|------|-------|
| **Violence in Car** | [Kaggle](https://www.kaggle.com/datasets/xuantin/violence-in-car) | Video | 🎯 **Primary** |
| **SCVD** | [Kaggle](https://www.kaggle.com/datasets/toluwaniaremu/smartcity-cctv-violence-detection-dataset-scvd) | Video | 🎯 **Secondary** |
| **Guns & Knives CCTV** | [Kaggle](https://www.kaggle.com/datasets/kruthisb999/guns-and-knifes-detection-in-cctv-videos) | Video | 🔫 **Weapon Detection** |

&gt; **Note:** Datasets are NOT included in this repository. Download them from Kaggle and place in `data/raw/`.

---

## 🚀 Installation

### Prerequisites

- **NVIDIA GPU** with CUDA 11.8+ support
- **Conda** (Miniconda or Anaconda)
- **Git**

### Step 1: Clone Repository

```bash
git clone https://github.com/YOUR_USERNAME/in-car-violence-detection.git
cd in-car-violence-detection