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
        
        prep = self.cfg.to_dict()['preprocessing']
        self.frame_size = tuple(prep['frame_size'])
        self.clip_length = prep['clip_length']
        self.stride = prep['stride']
        self.target_fps = prep['target_fps']
        
        self.logger.info(f"Preprocessor initialized: frame_size={self.frame_size}, "
                        f"clip_length={self.clip_length}, stride={self.stride}")
    
    def extract_frames(self, video_path: str) -> List[np.ndarray]:
        """Extract frames from video at target FPS."""
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            self.logger.error(f"Cannot open video: {video_path}")
            return []
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        
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
                frame_resized = cv2.resize(frame, self.frame_size)
                frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)
            
            frame_idx += 1
        
        cap.release()
        return frames
    
    def create_clips(self, frames: List[np.ndarray], video_id: str, 
                     label: int, output_dir: str, dataset_name: str) -> int:
        """Create overlapping clips from frames."""
        if len(frames) < self.clip_length:
            self.logger.warning(f"Video {video_id} too short ({len(frames)} frames), padding")
            while len(frames) < self.clip_length:
                frames.append(frames[-1] if frames else 
                             np.zeros((*self.frame_size, 3), dtype=np.uint8))
        
        clips_created = 0
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        for i in range(0, len(frames) - self.clip_length + 1, self.stride):
            clip = np.array(frames[i:i + self.clip_length])
            
            clip_filename = f"{dataset_name}_{video_id}_clip{clips_created:04d}.npy"
            clip_path = output_path / clip_filename
            np.save(clip_path, clip)
            
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
    
    def process_violence_in_car(self):
        """Process Dataset 1: Violence in Car (already split into train/val)."""
        cfg_dict = self.cfg.to_dict()
        dataset_cfg = cfg_dict['datasets']['violence_in_car']
        raw_path = Path(dataset_cfg['raw_path'])
        processed_base = Path(dataset_cfg['processed_path'])
        
        self.logger.info(f"Processing: {dataset_cfg['name']}")
        
        if not raw_path.exists():
            self.logger.error(f"Raw data not found: {raw_path}")
            return
        
        # Process each split (train, val)
        for split_name in dataset_cfg['splits']:
            split_dir = raw_path / split_name
            if not split_dir.exists():
                continue
            
            self.logger.info(f"\nProcessing split: {split_name}")
            
            # Create output directory
            split_output = processed_base / split_name / 'clips'
            split_output.mkdir(parents=True, exist_ok=True)
            
            # Process each class
            for class_info in dataset_cfg['classes']:
                class_name = class_info['name']
                label = class_info['label']
                class_dir = split_dir / class_name
                
                if not class_dir.exists():
                    self.logger.warning(f"Class dir not found: {class_dir}")
                    continue
                
                # Get videos
                videos = []
                for ext in dataset_cfg['video_extensions']:
                    videos.extend(list(class_dir.glob(f"*{ext}")))
                
                self.logger.info(f"  {class_name}: {len(videos)} videos")
                
                total_clips = 0
                for video_path in tqdm(videos, desc=f"{split_name}/{class_name}"):
                    frames = self.extract_frames(str(video_path))
                    if not frames:
                        continue
                    
                    video_id = f"{split_name}_{class_name}_{video_path.stem}"
                    clips = self.create_clips(
                        frames, video_id, label, 
                        str(split_output), 'violence_in_car'
                    )
                    total_clips += clips
                
                self.logger.info(f"  Created {total_clips} clips")
        
        self.logger.info("Dataset 1 processing complete!")
        self._print_statistics(processed_base)
    
    def process_scvd(self):
        """Process Dataset 2: SCVD (has Train/Test, we use Train for train/val, Test for test)."""
        cfg_dict = self.cfg.to_dict()
        dataset_cfg = cfg_dict['datasets']['scvd']
        raw_path = Path(dataset_cfg['raw_path'])
        processed_base = Path(dataset_cfg['processed_path'])
        
        self.logger.info(f"Processing: {dataset_cfg['name']}")
        
        if not raw_path.exists():
            self.logger.error(f"Raw data not found: {raw_path}")
            return
        
        # Process Train split -> create train/val from it
        train_dir = raw_path / 'Train'
        if train_dir.exists():
            self.logger.info("\nProcessing Train split (will create train/val)")
            
            # Collect all videos with labels
            all_videos = []
            for class_info in dataset_cfg['classes']:
                class_name = class_info['name']
                label = class_info['label']
                class_dir = train_dir / class_name
                
                if not class_dir.exists():
                    continue
                
                videos = []
                for ext in dataset_cfg['video_extensions']:
                    videos.extend(list(class_dir.glob(f"*{ext}")))
                
                for v in videos:
                    all_videos.append((v, label, class_name))
            
            self.logger.info(f"Total videos in Train: {len(all_videos)}")
            
            # Shuffle and split into train/val (80/20)
            random.seed(42)
            random.shuffle(all_videos)
            
            n_train = int(len(all_videos) * 0.8)
            train_videos = all_videos[:n_train]
            val_videos = all_videos[n_train:]
            
            # Process train videos
            train_output = processed_base / 'train' / 'clips'
            train_output.mkdir(parents=True, exist_ok=True)
            
            self.logger.info(f"Train videos: {len(train_videos)}")
            for video_path, label, class_name in tqdm(train_videos, desc="Train"):
                frames = self.extract_frames(str(video_path))
                if frames:
                    video_id = f"train_{class_name}_{video_path.stem}"
                    self.create_clips(frames, video_id, label, str(train_output), 'scvd')
            
            # Process val videos
            val_output = processed_base / 'val' / 'clips'
            val_output.mkdir(parents=True, exist_ok=True)
            
            self.logger.info(f"Val videos: {len(val_videos)}")
            for video_path, label, class_name in tqdm(val_videos, desc="Val"):
                frames = self.extract_frames(str(video_path))
                if frames:
                    video_id = f"val_{class_name}_{video_path.stem}"
                    self.create_clips(frames, video_id, label, str(val_output), 'scvd')
        
        # Process Test split -> use as test set
        test_dir = raw_path / 'Test'
        if test_dir.exists():
            self.logger.info("\nProcessing Test split")
            
            test_output = processed_base / 'test' / 'clips'
            test_output.mkdir(parents=True, exist_ok=True)
            
            for class_info in dataset_cfg['classes']:
                class_name = class_info['name']
                label = class_info['label']
                class_dir = test_dir / class_name
                
                if not class_dir.exists():
                    continue
                
                videos = []
                for ext in dataset_cfg['video_extensions']:
                    videos.extend(list(class_dir.glob(f"*{ext}")))
                
                self.logger.info(f"  {class_name}: {len(videos)} videos")
                
                for video_path in tqdm(videos, desc=f"Test/{class_name}"):
                    frames = self.extract_frames(str(video_path))
                    if frames:
                        video_id = f"test_{class_name}_{video_path.stem}"
                        self.create_clips(frames, video_id, label, str(test_output), 'scvd')
        
        self.logger.info("Dataset 2 processing complete!")
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
    
    # Process Violence in Car
    preprocessor.process_violence_in_car()
    
    # Process SCVD
    preprocessor.process_scvd()


if __name__ == "__main__":
    main()
