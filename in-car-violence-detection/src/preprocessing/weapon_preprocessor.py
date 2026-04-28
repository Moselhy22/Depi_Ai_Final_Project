"""
Weapon detection data preprocessor.
Handles Dataset 3 (Guns and Knives CCTV) and converts to YOLO format.
"""

import cv2
import os
import random
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import List, Tuple, Dict
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils.config_parser import load_config
from src.utils.logger import setup_logger


class WeaponPreprocessor:
    """
    Preprocessor for weapon detection dataset.
    Extracts frames from videos and creates YOLO format dataset.
    """
    
    def __init__(self, config_path: str = "configs/dataset_paths.yaml"):
        self.cfg = load_config(config_path)
        self.logger = setup_logger("WeaponPreprocessor")
        
        prep = self.cfg.to_dict()['preprocessing']
        self.img_size = 640  # YOLO default
        
        self.logger.info(f"Weapon preprocessor initialized: img_size={self.img_size}")
    
    def extract_frames(self, video_path: str, extract_every: int = 5) -> List[np.ndarray]:
        """
        Extract frames from video.
        
        Args:
            video_path: Path to video file
            extract_every: Extract 1 frame every N frames
            
        Returns:
            List of frames as numpy arrays (BGR format for YOLO)
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            self.logger.error(f"Cannot open video: {video_path}")
            return []
        
        frames = []
        frame_idx = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_idx % extract_every == 0:
                # Keep BGR format (YOLO expects BGR)
                frames.append(frame)
            
            frame_idx += 1
        
        cap.release()
        return frames
    
    def create_yolo_structure(self, output_dir: str):
        """
        Create YOLO directory structure.
        
        Args:
            output_dir: Base output directory
        """
        yolo_dir = Path(output_dir) / 'yolo_format'
        
        for split in ['train', 'val', 'test']:
            (yolo_dir / 'images' / split).mkdir(parents=True, exist_ok=True)
            (yolo_dir / 'labels' / split).mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"YOLO structure created at: {yolo_dir}")
        return yolo_dir
    
    def save_yolo_label(self, label_path: Path, class_id: int, 
                        bbox: Tuple[float, float, float, float]):
        """
        Save YOLO format label file.
        
        Args:
            label_path: Path to .txt file
            class_id: Class index
            bbox: (x_center, y_center, width, height) all normalized 0-1
        """
        with open(label_path, 'w') as f:
            f.write(f"{class_id} {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}\n")
    
    def process_dataset(self):
        """
        Process Guns and Knives dataset into YOLO format.
        """
        cfg_dict = self.cfg.to_dict()
        dataset_cfg = cfg_dict['datasets']['guns_knives']
        prep_cfg = cfg_dict['preprocessing']
        
        raw_path = Path(dataset_cfg['raw_path'])
        processed_base = Path(dataset_cfg['processed_path'])
        
        self.logger.info(f"Processing dataset: {dataset_cfg['name']}")
        self.logger.info(f"Raw path: {raw_path}")
        
        # Base directory inside dataset
        base_dir = raw_path / dataset_cfg.get('base_dir', '')
        
        if not base_dir.exists():
            self.logger.error(f"Base directory not found: {base_dir}")
            return
        
        # Create YOLO structure
        yolo_dir = self.create_yolo_structure(processed_base)
        
        # Collect all samples
        all_samples = []
        
        for class_info in dataset_cfg['classes']:
            class_name = class_info['name']
            class_id = class_info['yolo_id']
            class_dir = base_dir / class_name
            
            if not class_dir.exists():
                self.logger.warning(f"Class directory not found: {class_dir}")
                continue
            
            # Find all videos recursively
            videos = []
            for ext in dataset_cfg['video_extensions']:
                videos.extend(list(class_dir.rglob(f"*{ext}")))
            
            self.logger.info(f"Found {len(videos)} videos for class '{class_name}'")
            
            for video_path in videos:
                all_samples.append({
                    'path': str(video_path),
                    'class_name': class_name,
                    'class_id': class_id
                })
        
        if not all_samples:
            self.logger.error("No samples found!")
            return
        
        # Shuffle and split
        random.seed(42)
        random.shuffle(all_samples)
        
        split_cfg = prep_cfg['weapon']
        n_total = len(all_samples)
        n_train = int(n_total * split_cfg['train_ratio'])
        n_val = int(n_total * split_cfg['val_ratio'])
        
        splits = {
            'train': all_samples[:n_train],
            'val': all_samples[n_train:n_train + n_val],
            'test': all_samples[n_train + n_val:]
        }
        
        # Process each split
        for split_name, samples in splits.items():
            self.logger.info(f"Processing {split_name}: {len(samples)} videos")
            
            img_output = yolo_dir / 'images' / split_name
            label_output = yolo_dir / 'labels' / split_name
            
            img_count = 0
            
            for sample in tqdm(samples, desc=f"Processing {split_name}"):
                video_path = Path(sample['path'])
                class_id = sample['class_id']
                class_name = sample['class_name']
                
                # Extract frames
                frames = self.extract_frames(
                    str(video_path), 
                    extract_every=split_cfg['extract_every_n_frames']
                )
                
                for frame in frames:
                    # Save image
                    img_name = f"{video_path.stem}_frame{img_count:05d}.jpg"
                    img_path = img_output / img_name
                    cv2.imwrite(str(img_path), frame)
                    
                    # Save label
                    # For now, we use full-image placeholder
                    # In real usage, you should annotate with actual bounding boxes
                    label_name = img_name.replace('.jpg', '.txt')
                    label_path = label_output / label_name
                    
                    # Placeholder: whole image as bbox (to be replaced with real annotations)
                    # class_id x_center y_center width height
                    h, w = frame.shape[:2]
                    if class_name == 'background':
                        # Empty file for background
                        with open(label_path, 'w') as f:
                            pass
                    else:
                        # Full image placeholder (replace with actual bbox!)
                        self.save_yolo_label(label_path, class_id, (0.5, 0.5, 1.0, 1.0))
                    
                    img_count += 1
            
            self.logger.info(f"Saved {img_count} images for {split_name}")
        
        # Create dataset.yaml
        self._create_dataset_yaml(yolo_dir, dataset_cfg['classes'])
        
        self.logger.info("Weapon dataset processing complete!")
        self._print_statistics(yolo_dir)
    
    def _create_dataset_yaml(self, yolo_dir: Path, classes: List[Dict]):
        """Create YOLO dataset configuration file."""
        names = [c['name'] for c in classes]
        
        yaml_content = f"""path: {yolo_dir.absolute()}
train: images/train
val: images/val
test: images/test

nc: {len(names)}
names: {names}
"""
        
        yaml_path = yolo_dir / 'dataset.yaml'
        with open(yaml_path, 'w') as f:
            f.write(yaml_content)
        
        self.logger.info(f"Created dataset.yaml: {yaml_path}")
    
    def _print_statistics(self, yolo_dir: Path):
        """Print dataset statistics."""
        self.logger.info("=" * 50)
        self.logger.info("WEAPON DATASET STATISTICS")
        self.logger.info("=" * 50)
        
        for split in ['train', 'val', 'test']:
            img_dir = yolo_dir / 'images' / split
            if not img_dir.exists():
                continue
            
            images = list(img_dir.glob("*.jpg"))
            labels = list((yolo_dir / 'labels' / split).glob("*.txt"))
            
            self.logger.info(f"{split.upper()}: {len(images)} images, {len(labels)} labels")
        
        self.logger.info("=" * 50)
        self.logger.info("NOTE: Labels contain placeholder bounding boxes!")
        self.logger.info("Please annotate with actual bounding boxes for training.")
        self.logger.info("=" * 50)


def main():
    """Main entry point."""
    preprocessor = WeaponPreprocessor()
    preprocessor.process_dataset()


if __name__ == "__main__":
    main()
