"""
Weapon detection evaluation script using YOLOv8.
"""

import os
import sys
from pathlib import Path

from ultralytics import YOLO

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils.config_parser import load_config
from src.utils.logger import setup_logger


class WeaponEvaluator:
    """Evaluator for weapon detection model (YOLOv8)."""
    
    def __init__(self, model_path: str, config_path: str = "configs/train_weapon.yaml"):
        self.model_path = model_path
        self.cfg = load_config(config_path)
        self.logger = setup_logger("WeaponEvaluator")
        
        self.model = None
    
    def load_model(self):
        """Load trained YOLOv8 model."""
        self.logger.info(f"Loading model from: {self.model_path}")
        
        self.model = YOLO(self.model_path)
        
        self.logger.info("YOLOv8 model loaded")
    
    def evaluate(self, data_yaml: str = None, split: str = 'test'):
        """
        Evaluate model on test set.
        
        Args:
            data_yaml: Path to dataset.yaml (uses config default if None)
            split: Dataset split to evaluate ('train', 'val', 'test')
        """
        if self.model is None:
            self.load_model()
        
        if data_yaml is None:
            data_yaml = self.cfg.to_dict()['data']['yaml_path']
        
        self.logger.info(f"Evaluating on {split} split...")
        self.logger.info(f"Data config: {data_yaml}")
        
        # Run validation
        results = self.model.val(data=data_yaml, split=split)
        
        # Extract metrics
        metrics = {
            'mAP50': results.box.map50,
            'mAP50-95': results.box.map,
            'precision': results.box.mp,
            'recall': results.box.mr,
            'fitness': results.fitness
        }
        
        self.logger.info("=" * 50)
        self.logger.info("WEAPON DETECTION EVALUATION RESULTS")
        self.logger.info("=" * 50)
        self.logger.info(f"mAP@50: {metrics['mAP50']:.4f}")
        self.logger.info(f"mAP@50-95: {metrics['mAP50-95']:.4f}")
        self.logger.info(f"Precision: {metrics['precision']:.4f}")
        self.logger.info(f"Recall: {metrics['recall']:.4f}")
        self.logger.info(f"Fitness: {metrics['fitness']:.4f}")
        self.logger.info("=" * 50)
        
        return metrics, results
    
    def predict_video(self, video_path: str, save: bool = True):
        """
        Run inference on a video file.
        
        Args:
            video_path: Path to video
            save: Whether to save annotated video
        """
        if self.model is None:
            self.load_model()
        
        self.logger.info(f"Running inference on: {video_path}")
        
        results = self.model(video_path, save=save, conf=0.5)
        
        # Print detections
        for r in results:
            boxes = r.boxes
            if boxes is not None:
                for box in boxes:
                    cls = int(box.cls)
                    conf = float(box.conf)
                    self.logger.info(f"Detected: {r.names[cls]} ({conf:.2f})")
        
        return results
    
    def export_model(self, format: str = "onnx"):
        """Export model to deployment format."""
        if self.model is None:
            self.load_model()
        
        self.logger.info(f"Exporting model to {format}...")
        path = self.model.export(format=format)
        self.logger.info(f"Exported to: {path}")
        return path


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate weapon detection model')
    parser.add_argument('--model', required=True, help='Path to YOLO model (.pt)')
    parser.add_argument('--data', default=None, help='Path to dataset.yaml')
    parser.add_argument('--split', default='test', choices=['train', 'val', 'test'])
    parser.add_argument('--config', default='configs/train_weapon.yaml', help='Config file')
    args = parser.parse_args()
    
    evaluator = WeaponEvaluator(args.model, args.config)
    
    # Evaluate
    metrics, results = evaluator.evaluate(args.data, args.split)
    
    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()
