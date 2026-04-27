"""
In-Car Violence Detection System
================================

A real-time violence and weapon detection system for vehicle cabin monitoring.
"""

__version__ = "0.1.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

import torch


def check_gpu():
    """Check GPU availability and print device info."""
    if torch.cuda.is_available():
        device = torch.cuda.get_device_name(0)
        print(f"✅ CUDA is available!")
        print(f"   Device: {device}")
        print(f"   CUDA Version: {torch.version.cuda}")
        print(f"   PyTorch Version: {torch.__version__}")
        return True
    else:
        print("⚠️  CUDA not available. Using CPU.")
        return False


if __name__ == "__main__":
    check_gpu()