"""
Weapon detection training pipeline using YOLOv8.
"""

import os
import sys
from pathlib import Path

from ultralytics import YOLO

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils.config_parser import load_config
from src.utils.logger import setup_logger


class WeaponTrainer:
    """Trainer for weapon detection using YOLOv8."""
    
    def __init__(self, config_path: str = "configs/train_weapon.yaml"):
        self.cfg = load_config(config_path)
        self.logger = setup_logger("WeaponTrainer")
        
        self.model = None
    
    def setup_model(self):
        """Initialize YOLOv8 model."""
        model_cfg = self.cfg.to_dict()['model']
        variant = model_cfg['variant']
        
        # Load pre-trained YOLOv8
        self.model = YOLO(f"yolov8{variant}.pt")
        
        self.logger.info(f"YOLOv8{variant} model loaded")
    
    def train(self):
        """Train weapon detection model."""
        data_cfg = self.cfg.to_dict()['data']
        train_cfg = self.cfg.to_dict()['training']
        aug_cfg = self.cfg.to_dict()['augmentation']
        log_cfg = self.cfg.to_dict()['logging']
        
        yaml_path = data_cfg['yaml_path']
        
        self.logger.info(f"Starting training with data: {yaml_path}")
        
        # Train
        results = self.model.train(
            data=yaml_path,
            epochs=train_cfg['epochs'],
            batch=train_cfg['batch_size'],
            imgsz=train_cfg['imgsz'],
            patience=train_cfg['patience'],
            
            # Optimizer
            optimizer=train_cfg['optimizer']['name'],
            lr0=train_cfg['optimizer']['lr0'],
            lrf=train_cfg['optimizer']['lrf'],
            momentum=train_cfg['optimizer']['momentum'],
            weight_decay=train_cfg['optimizer']['weight_decay'],
            
            # Warmup
            warmup_epochs=train_cfg['warmup_epochs'],
            
            # Loss
            box=train_cfg['box'],
            cls=train_cfg['cls'],
            dfl=train_cfg['dfl'],
            
            # Augmentation
            augment=True,
            mosaic=aug_cfg['mosaic'],
            mixup=aug_cfg['mixup'],
            copy_paste=aug_cfg['copy_paste'],
            degrees=aug_cfg['degrees'],
            translate=aug_cfg['translate'],
            scale=aug_cfg['scale'],
            shear=aug_cfg['shear'],
            perspective=aug_cfg['perspective'],
            flipud=aug_cfg['flipud'],
            fliplr=aug_cfg['fliplr'],
            hsv_h=aug_cfg['hsv_h'],
            hsv_s=aug_cfg['hsv_s'],
            hsv_v=aug_cfg['hsv_v'],
            
            # Hardware
            device=0,  # GPU
            workers=train_cfg.get('workers', 8),
            
            # Logging
            project=log_cfg['save_dir'],
            name=log_cfg['name'],
            exist_ok=True,
            pretrained=True,
            verbose=True
        )
        
        self.logger.info(f"Training complete!")
        self.logger.info(f"Best mAP50: {results.results_dict.get('metrics/mAP50(B)', 'N/A')}")
        
        return results
    
    def validate(self):
        """Validate on test set."""
        self.logger.info("Running validation on test set...")
        results = self.model.val(split='test')
        
        self.logger.info(f"Test mAP50: {results.box.map50:.4f}")
        self.logger.info(f"Test mAP50-95: {results.box.map:.4f}")
        
        return results
    
    def export(self, format: str = "onnx"):
        """Export model to deployment format."""
        self.logger.info(f"Exporting model to {format}...")
        path = self.model.export(format=format)
        self.logger.info(f"Exported to: {path}")
        return path


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train weapon detection model')
    parser.add_argument('--config', default='configs/train_weapon.yaml', help='Config file')
    args = parser.parse_args()
    
    trainer = WeaponTrainer(args.config)
    trainer.setup_model()
    
    # Train
    results = trainer.train()
    
    # Validate
    trainer.validate()
    
    # Export
    # trainer.export("onnx")
    
    print("\nTraining complete!")


if __name__ == "__main__":
    main()
