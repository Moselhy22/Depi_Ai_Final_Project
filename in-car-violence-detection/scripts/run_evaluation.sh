#!/bin/bash

# Evaluation script for trained models

echo "=========================================="
echo "  In-Car Violence Detection - Evaluation"
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

# Evaluate violence detection model
echo "------------------------------------------"
echo "Evaluating Violence Detection Model"
echo "------------------------------------------"

VIOLENCE_MODEL="models/checkpoints/violence/best_model.pth"

if [ ! -f "$VIOLENCE_MODEL" ]; then
    echo "⚠️  Violence model not found at: $VIOLENCE_MODEL"
    echo "Please train the model first: ./scripts/run_training.sh"
else
    python -m src.evaluation.evaluate_violence \
        --model "$VIOLENCE_MODEL" \
        --test-dir data/processed/violence_in_car/test/clips
fi

echo ""
echo "------------------------------------------"
echo "Evaluating Weapon Detection Model"
echo "------------------------------------------"

WEAPON_MODEL="models/checkpoints/weapon/best.pt"

if [ ! -f "$WEAPON_MODEL" ]; then
    echo "⚠️  Weapon model not found at: $WEAPON_MODEL"
    echo "Please train the model first: ./scripts/run_training.sh"
else
    python -m src.evaluation.evaluate_weapon \
        --model "$WEAPON_MODEL" \
        --split test
fi

echo ""
echo "=========================================="
echo "  Evaluation Complete!"
echo "=========================================="
