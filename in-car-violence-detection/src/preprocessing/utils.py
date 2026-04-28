"""
Shared preprocessing utilities.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional


def resize_frame(frame: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    """
    Resize frame to target size.
    
    Args:
        frame: Input frame (H, W, C)
        target_size: (width, height)
        
    Returns:
        Resized frame
    """
    return cv2.resize(frame, target_size)


def normalize_frame(frame: np.ndarray, 
                   mean: Optional[np.ndarray] = None,
                   std: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Normalize frame to [0, 1] or standardize.
    
    Args:
        frame: Input frame
        mean: Mean for standardization (default: ImageNet)
        std: Std for standardization (default: ImageNet)
        
    Returns:
        Normalized frame
    """
    frame = frame.astype(np.float32) / 255.0
    
    if mean is not None and std is not None:
        frame = (frame - mean) / std
    
    return frame


def apply_augmentation(frame: np.ndarray, 
                      rotation: float = 0,
                      flip_horizontal: bool = False,
                      brightness: float = 0) -> np.ndarray:
    """
    Apply basic augmentations to a frame.
    
    Args:
        frame: Input frame
        rotation: Rotation angle in degrees
        flip_horizontal: Whether to flip horizontally
        brightness: Brightness adjustment (-1 to 1)
        
    Returns:
        Augmented frame
    """
    h, w = frame.shape[:2]
    
    # Rotation
    if rotation != 0:
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, rotation, 1.0)
        frame = cv2.warpAffine(frame, M, (w, h))
    
    # Horizontal flip
    if flip_horizontal:
        frame = cv2.flip(frame, 1)
    
    # Brightness
    if brightness != 0:
        frame = np.clip(frame + brightness * 255, 0, 255).astype(np.uint8)
    
    return frame


def get_video_info(video_path: str) -> dict:
    """
    Get video file information.
    
    Args:
        video_path: Path to video file
        
    Returns:
        Dictionary with video info
    """
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        return {}
    
    info = {
        'fps': cap.get(cv2.CAP_PROP_FPS),
        'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        'duration': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / cap.get(cv2.CAP_PROP_FPS) if cap.get(cv2.CAP_PROP_FPS) > 0 else 0
    }
    
    cap.release()
    return info


def validate_dataset_path(path: str, required_subdirs: List[str] = None) -> bool:
    """
    Validate that dataset path exists and has required structure.
    
    Args:
        path: Dataset root path
        required_subdirs: List of required subdirectories
        
    Returns:
        True if valid, False otherwise
    """
    path = Path(path)
    
    if not path.exists():
        return False
    
    if required_subdirs:
        for subdir in required_subdirs:
            if not (path / subdir).exists():
                return False
    
    return True


def count_videos(directory: str, extensions: List[str] = None) -> int:
    """
    Count video files in directory.
    
    Args:
        directory: Directory to search
        extensions: List of video extensions
        
    Returns:
        Number of video files
    """
    if extensions is None:
        extensions = ['.mp4', '.avi', '.mov', '.mkv']
    
    path = Path(directory)
    count = 0
    
    for ext in extensions:
        count += len(list(path.rglob(f"*{ext}")))
    
    return count
