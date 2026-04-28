"""
Tests for model architectures.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import torch
from src.models.violence_detector import InCarViolenceDetector, SpatialAttention, TemporalAttention


def test_spatial_attention():
    """Test spatial attention module."""
    print("Testing SpatialAttention...")
    
    att = SpatialAttention(in_channels=2048)
    x = torch.randn(2, 2048, 7, 7)
    
    out, weights = att(x)
    
    assert out.shape == (2, 2048, 7, 7)
    assert weights.shape == (2, 1, 7, 7)
    assert torch.all((weights >= 0) & (weights <= 1))
    
    print("✅ SpatialAttention works")


def test_temporal_attention():
    """Test temporal attention module."""
    print("Testing TemporalAttention...")
    
    att = TemporalAttention(hidden_dim=512)
    x = torch.randn(2, 16, 1024)  # batch=2, time=16, hidden*2=1024
    
    context, weights = att(x)
    
    assert context.shape == (2, 1024)
    assert weights.shape == (2, 16, 1)
    assert torch.allclose(weights.sum(dim=1), torch.ones(2, 1), atol=1e-6)
    
    print("✅ TemporalAttention works")


def test_violence_detector():
    """Test full violence detector model."""
    print("Testing InCarViolenceDetector...")
    
    model = InCarViolenceDetector(num_classes=2)
    
    # Test forward pass
    x = torch.randn(2, 16, 3, 224, 224)
    output, temp_w, spatial_w = model(x)
    
    assert output.shape == (2, 2)
    assert temp_w.shape == (2, 16, 1)
    assert len(spatial_w) == 16
    
    # Test predict
    model.eval()
    probs, preds, scores = model.predict(x)
    
    assert probs.shape == (2, 2)
    assert preds.shape == (2,)
    assert scores.shape == (2,)
    assert torch.all((scores >= 0) & (scores <= 1))
    
    print("✅ InCarViolenceDetector works")


def test_model_parameters():
    """Test model parameter count."""
    print("Testing model parameters...")
    
    model = InCarViolenceDetector()
    total = sum(p.numel() for p in model.parameters())
    
    assert total > 0
    print(f"✅ Total parameters: {total:,}")


if __name__ == "__main__":
    print("=" * 50)
    print("Running Model Tests")
    print("=" * 50)
    
    test_spatial_attention()
    test_temporal_attention()
    test_violence_detector()
    test_model_parameters()
    
    print("\n" + "=" * 50)
    print("All model tests passed! ✅")
    print("=" * 50)
