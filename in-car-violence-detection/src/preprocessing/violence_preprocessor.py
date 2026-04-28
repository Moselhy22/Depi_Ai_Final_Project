"""
Violence detection data preprocessor.
Handles Dataset 1 (Violence in Car) and Dataset 2 (SCVD).
"""

import cv2
import os
import json
import shutil
import random
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import List, Tuple, Dict
import sys

# Add parent to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils.config_parser import load_config
from src.utils.logger import setup_logger


class ViolencePreprocessor:
    """
    Preprocessor for violence detection datasets.
    Converts videos to numpy clip arrays for model training.
    """
    
    def __init__(self, config_path: str = "configs/dataset_paths.yaml"):
        self.cfg = load_config(config_path)
        self.logger = setup_logger("ViolencePreprocessor")
        
        # Extract preprocessing params
        prep = self.cfg.to_dict()['preprocessing']
        self.frame_size = tuple(prep['frame_size'])
        self.clip_length = prep['clip_length']
        self.stride = prep['stride']
        self.target_fps = prep['target_fps']
        
        self.logger.info(f"Preprocessor initialized: frame_size={self.frame_size}, "
                        f"clip_length={self.clip_length}, stride={self.stride}")
    
    def extract_frames(self, video_path: str) -> List[np.ndarray]:
        """
        Extract frames from video at target FPS.
        
        Args:
            video_path: Path to video file
            
        Returns:
            List of frames as numpy arrays (RGB, resized)
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            self.logger.error(f"Cannot open video: {video_path}")
            return []
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Calculate frame sampling rate
        if fps > self.target_fps:
            sample_every = int(fps / self.target_fps)
        else:
            sample_every = 1
        
        frames = []
        frame_idx = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_idx % sample_every == 0:
                # Resize and convert to RGB
                frame_resized = cv2.resize(frame, self.frame_size)
                frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)
            
            frame_idx += 1
        
        cap.release()
        
        self.logger.debug(f"Extracted {len(frames)} frames from {video_path}")
        return frames
    
    def create_clips(self, frames: List[np.ndarray], video_id: str, 
                     label: int, output_dir: str, dataset_name: str) -> int:
        """
        Create overlapping clips from frames.
        
        Args:
            frames: List of frames
            video_id: Unique video identifier
            label: Class label (0=normal, 1=violence)
            output_dir: Where to save clips
            dataset_name: Name of source dataset
            
        Returns:
            Number of clips created
        """
        if len(frames) < self.clip_length:
            self.logger.warning(f"Video {video_id} too short ({len(frames)} frames), padding")
            # Pad with last frame
            while len(frames) < self.clip_length:
                frames.append(frames[-1] if frames else 
                             np.zeros((*self.frame_size, 3), dtype=np.uint8))
        
        clips_created = 0
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        for i in range(0, len(frames) - self.clip_length + 1, self.stride):
            clip = np.array(frames[i:i + self.clip_length])
            
            # Save as .npy
            clip_filename = f"{dataset_name}_{video_id}_clip{clips_created:04d}.npy"
            clip_path = output_path / clip_filename
            np.save(clip_path, clip)
            
            # Save metadata
            meta = {
                'dataset': dataset_name,
                'video_id': video_id,
                'clip_id': clips_created,
                'label': label,
                'start_frame': i,
                'num_frames': len(clip),
                'shape': list(clip.shape),
                'frame_size': self.frame_size
            }
            
            meta_path = clip_path.with_suffix('.json')
            with open(meta_path, 'w') as f:
                json.dump(meta, f, indent=2)
            
            clips_created += 1
        
        return clips_created
    
    def process_dataset(self, dataset_name: str, split_ratios: Dict[str, float] = None):
        """
        Process an entire dataset (Violence in Car or SCVD).
        
        Args:
            dataset_name: 'violence_in_car' or 'scvd'
            split_ratios: Dict with 'train', 'val', 'test' ratios
        """
        cfg_dict = self.cfg.to_dict()
        dataset_cfg = cfg_dict['datasets'][dataset_name]
        prep_cfg = cfg_dict['preprocessing']
        
        if split_ratios is None:
            split_ratios = prep_cfg['violence']
        
        raw_path = Path(dataset_cfg['raw_path'])
        processed_base = Path(dataset_cfg['processed_path'])
        
        self.logger.info(f"Processing dataset: {dataset_cfg['name']}")
        self.logger.info(f"Raw path: {raw_path}")
        self.logger.info(f"Output path: {processed_base}")
        
        if not raw_path.exists():
            self.logger.error(f"Raw data not found: {raw_path}")
            self.logger.info("Please download the dataset and place it in the correct location")
            return
        
        # Process each class
        for class_info in dataset_cfg['classes']:
            class_name = class_info['name']
            label = class_info['label']
            class_dir = raw_path / class_info['path']
            
            if not class_dir.exists():
                self.logger.warning(f"Class directory not found: {class_dir}")
                continue
            
            # Get all videos
            videos = []
            for ext in dataset_cfg['video_extensions']:
                videos.extend(list(class_dir.glob(f"*{ext}")))
            
            if not videos:
                self.logger.warning(f"No videos found in {class_dir}")
                continue
            
            self.logger.info(f"Found {len(videos)} videos for class '{class_name}'")
            
            # Shuffle and split
            random.seed(42)
            random.shuffle(videos)
            
            n_total = len(videos)
            n_train = int(n_total * split_ratios['train_ratio'])
            n_val = int(n_total * split_ratios['val_ratio'])
            
            splits = {
                'train': videos[:n_train],
                'val': videos[n_train:n_train + n_val],
                'test': videos[n_train + n_val:]
            }
            
            # Process each split
            for split_name, split_videos in splits.items():
                self.logger.info(f"Processing {split_name}: {len(split_videos)} videos")
                
                split_output = processed_base / split_name / 'clips'
                split_output.mkdir(parents=True, exist_ok=True)
                
                total_clips = 0
                
                for video_path in tqdm(split_videos, desc=f"{class_name}/{split_name}"):
                    # Extract frames
                    frames = self.extract_frames(str(video_path))
                    
                    if not frames:
                        continue
                    
                    # Create clips
                    video_id = f"{class_name}_{video_path.stem}"
                    clips = self.create_clips(
                        frames, video_id, label, 
                        str(split_output), dataset_name
                    )
                    total_clips += clips
                
                self.logger.info(f"Created {total_clips} clips for {split_name}")
        
        self.logger.info(f"Dataset {dataset_name} processing complete!")
        self._print_statistics(processed_base)
    
    def _print_statistics(self, processed_path: Path):
        """Print dataset statistics."""
        self.logger.info("=" * 50)
        self.logger.info("DATASET STATISTICS")
        self.logger.info("=" * 50)
        
        for split in ['train', 'val', 'test']:
            split_dir = processed_path / split / 'clips'
            if not split_dir.exists():
                continue
            
            clips = list(split_dir.glob("*.npy"))
            labels = []
            
            for clip_path in clips:
                meta_path = clip_path.with_suffix('.json')
                if meta_path.exists():
                    with open(meta_path, 'r') as f:
                        meta = json.load(f)
                    labels.append(meta['label'])
            
            if labels:
                n_normal = labels.count(0)
                n_violence = labels.count(1)
                n_weapon = labels.count(2)
                
                self.logger.info(f"{split.upper()}: {len(clips)} clips")
                self.logger.info(f"  Normal: {n_normal}, Violence: {n_violence}, Weapons: {n_weapon}")
        
        self.logger.info("=" * 50)


def main():
    """Main entry point."""
    preprocessor = ViolencePreprocessor()
    
    # Process Violence in Car dataset
    preprocessor.process_dataset('violence_in_car')
    
    # Process SCVD dataset
    preprocessor.process_dataset('scvd')


if __name__ == "__main__":
    main()
