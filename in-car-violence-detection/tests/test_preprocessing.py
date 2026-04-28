"""
Tests for preprocessing modules.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.preprocessing.violence_preprocessor import ViolencePreprocessor
from src.preprocessing.weapon_preprocessor import WeaponPreprocessor
from src.preprocessing.utils import resize_frame, normalize_frame, get_video_info


def test_imports():
    """Test that all modules import correctly."""
    print("Testing imports...")
    
    # Test violence preprocessor
    vp = ViolencePreprocessor()
    assert vp is not None
    print("✅ ViolencePreprocessor imported")
    
    # Test weapon preprocessor
    wp = WeaponPreprocessor()
    assert wp is not None
    print("✅ WeaponPreprocessor imported")
    
    print("✅ All imports successful!")


def test_utils():
    """Test utility functions."""
    print("\nTesting utility functions...")
    
    import numpy as np
    
    # Test resize_frame
    frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    resized = resize_frame(frame, (224, 224))
    assert resized.shape == (224, 224, 3)
    print("✅ resize_frame works")
    
    # Test normalize_frame
    normalized = normalize_frame(resized)
    assert normalized.max() <= 1.0
    assert normalized.min() >= 0.0
    print("✅ normalize_frame works")
    
    print("✅ All utility tests passed!")


def test_config_loading():
    """Test configuration loading."""
    print("\nTesting config loading...")
    
    from src.utils.config_parser import load_config
    
    cfg = load_config("configs/dataset_paths.yaml")
    assert cfg is not None
    
    d = cfg.to_dict()
    assert 'datasets' in d
    assert 'violence_in_car' in d['datasets']
    print("✅ Config loaded successfully")
    
    print("✅ Config test passed!")


if __name__ == "__main__":
    print("=" * 50)
    print("Running Preprocessing Tests")
    print("=" * 50)
    
    test_imports()
    test_utils()
    test_config_loading()
    
    print("\n" + "=" * 50)
    print("All tests passed! ✅")
    print("=" * 50)
