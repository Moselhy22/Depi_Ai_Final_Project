#!/bin/bash

# Training script for violence and weapon detection models

echo "=========================================="
echo "  In-Car Violence Detection - Training"
echo "=========================================="
echo ""

# Activate conda environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate in-car-violence-detection

if [ $? -ne 0 ]; then
    echo "❌ Failed to activate conda environment"
    exit 1
fi

echo "✅ Environment activated: in-car-violence-detection"
echo ""

# Train violence detection model
echo "------------------------------------------"
echo "Training Violence Detection Model"
echo "------------------------------------------"

# Check if processed data exists
if [ ! -d "data/processed/violence_in_car/train/clips" ]; then
    echo "⚠️  Processed violence data not found!"
    echo "Please run preprocessing first: ./scripts/run_preprocessing.sh"
    exit 1
fi

python -m src.training.train_violence \
    --train-dir data/processed/violence_in_car/train/clips \
    --val-dir data/processed/violence_in_car/val/clips

echo ""
echo "------------------------------------------"
echo "Training Weapon Detection Model"
echo "------------------------------------------"

# Check if YOLO dataset exists
if [ ! -f "data/processed/guns_knives/yolo_format/dataset.yaml" ]; then
    echo "⚠️  YOLO dataset not found!"
    echo "Please run preprocessing first: ./scripts/run_preprocessing.sh"
    exit 1
fi

python -m src.training.train_weapon

echo ""
echo "=========================================="
echo "  Training Complete!"
echo "=========================================="
echo ""
echo "Checkpoints saved to:"
echo "  - models/checkpoints/violence/"
echo "  - models/checkpoints/weapon/"
