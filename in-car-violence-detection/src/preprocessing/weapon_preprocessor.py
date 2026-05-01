"""
Weapon detection data preprocessor.
Dataset 3 is already in YOLO format - just copy to processed location.
"""

import os
import shutil
from pathlib import Path
from typing import List, Dict
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils.config_parser import load_config
from src.utils.logger import setup_logger


class WeaponPreprocessor:
    """
    Preprocessor for weapon detection dataset.
    Since data is already in YOLO format, we just copy it.
    """
    
    def __init__(self, config_path: str = "configs/dataset_paths.yaml"):
        self.cfg = load_config(config_path)
        self.logger = setup_logger("WeaponPreprocessor")
    
    def count_files(self, directory: Path, extensions: List[str]) -> int:
        """Count files with given extensions."""
        count = 0
        for ext in extensions:
            count += len(list(directory.glob(f"*{ext}")))
            count += len(list(directory.glob(f"*{ext.upper()}")))
        return count
    
    def process_dataset(self):
        """Copy YOLO-formatted dataset to processed location."""
        cfg_dict = self.cfg.to_dict()
        dataset_cfg = cfg_dict['datasets']['guns_knives']
        
        raw_path = Path(dataset_cfg['raw_path']).resolve()
        processed_path = Path(dataset_cfg['processed_path']).resolve()
        
        self.logger.info(f"Processing: {dataset_cfg['name']}")
        self.logger.info(f"Raw path: {raw_path}")
        self.logger.info(f"Processed path: {processed_path}")
        
        if not raw_path.exists():
            self.logger.error(f"Raw data not found: {raw_path}")
            return
        
        # Create processed directory structure
        yolo_output = processed_path / 'yolo_format'
        yolo_output.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        for split in ['train', 'val', 'test']:
            (yolo_output / 'images' / split).mkdir(parents=True, exist_ok=True)
            (yolo_output / 'labels' / split).mkdir(parents=True, exist_ok=True)
        
        # Copy images and labels
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.webp']
        
        for split in ['train', 'val', 'test']:
            src_images = raw_path / split / 'images'
            src_labels = raw_path / split / 'labels'
            
            dst_images = yolo_output / 'images' / split
            dst_labels = yolo_output / 'labels' / split
            
            # Copy images
            if src_images.exists():
                self.logger.info(f"Copying {split} images from {src_images}...")
                copied = 0
                for ext in image_extensions:
                    for img in src_images.glob(f"*{ext}"):
                        shutil.copy2(img, dst_images / img.name)
                        copied += 1
                    for img in src_images.glob(f"*{ext.upper()}"):
                        shutil.copy2(img, dst_images / img.name)
                        copied += 1
                self.logger.info(f"  Copied {copied} images")
            
            # Copy labels
            if src_labels.exists():
                self.logger.info(f"Copying {split} labels from {src_labels}...")
                copied = 0
                for lbl in src_labels.glob("*.txt"):
                    shutil.copy2(lbl, dst_labels / lbl.name)
                    copied += 1
                self.logger.info(f"  Copied {copied} labels")
        
        # Copy and update data.yaml
        src_yaml = raw_path / 'data.yaml'
        dst_yaml = yolo_output / 'dataset.yaml'
        
        if src_yaml.exists():
            with open(src_yaml, 'r') as f:
                content = f.read()
            
            with open(dst_yaml, 'w') as f:
                f.write(content)
            
            self.logger.info(f"Created dataset.yaml: {dst_yaml}")
        
        # Count files
        for split in ['train', 'val', 'test']:
            img_dir = yolo_output / 'images' / split
            lbl_dir = yolo_output / 'labels' / split
            
            if img_dir.exists():
                n_images = self.count_files(img_dir, image_extensions)
                n_labels = len(list(lbl_dir.glob("*.txt")))
                self.logger.info(f"{split}: {n_images} images, {n_labels} labels")
        
        self.logger.info("Weapon dataset processing complete!")


def main():
    """Main entry point."""
    preprocessor = WeaponPreprocessor()
    preprocessor.process_dataset()


if __name__ == "__main__":
    main()
