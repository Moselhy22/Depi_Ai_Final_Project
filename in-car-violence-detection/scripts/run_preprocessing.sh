#!/bin/bash

# Preprocessing script for all datasets

echo "=========================================="
echo "  In-Car Violence Detection - Preprocessing"
echo "=========================================="
echo ""

# Activate conda environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate in-car-violence-detection

# Check if environment activated
if [ $? -ne 0 ]; then
    echo "❌ Failed to activate conda environment"
    exit 1
fi

echo "✅ Environment activated: in-car-violence-detection"
echo ""

# Process Violence in Car dataset
echo "------------------------------------------"
echo "Processing Dataset 1: Violence in Car"
echo "------------------------------------------"
python -m src.preprocessing.violence_preprocessor

echo ""
echo "------------------------------------------"
echo "Processing Dataset 3: Guns and Knives"
echo "------------------------------------------"
python -m src.preprocessing.weapon_preprocessor

echo ""
echo "=========================================="
echo "  Preprocessing Complete!"
echo "=========================================="
