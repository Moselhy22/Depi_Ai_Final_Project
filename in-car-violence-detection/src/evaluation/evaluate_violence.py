"""
Violence detection evaluation script.
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                            roc_auc_score, confusion_matrix, classification_report)
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.models.violence_detector import InCarViolenceDetector
from src.training.train_violence import ViolenceDataset
from src.utils.config_parser import load_config
from src.utils.logger import setup_logger


class ViolenceEvaluator:
    """Evaluator for violence detection model."""
    
    def __init__(self, model_path: str, config_path: str = "configs/train_violence.yaml"):
        self.model_path = model_path
        self.cfg = load_config(config_path)
        self.logger = setup_logger("ViolenceEvaluator")
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        
        self.results = {}
    
    def load_model(self):
        """Load trained model from checkpoint."""
        self.logger.info(f"Loading model from: {self.model_path}")
        
        checkpoint = torch.load(self.model_path, map_location=self.device)
        
        # Initialize model
        model_cfg = self.cfg.to_dict()['model']
        self.model = InCarViolenceDetector(
            num_classes=2,
            hidden_dim=model_cfg['hidden_dim'],
            num_layers=model_cfg['num_layers'],
            dropout=model_cfg['dropout'],
            backbone=model_cfg['backbone']
        ).to(self.device)
        
        # Load weights
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        self.logger.info(f"Model loaded from epoch {checkpoint.get('epoch', 'unknown')}")
        self.logger.info(f"Validation metrics: {checkpoint.get('metrics', {})}")
        
        return checkpoint
    
    def evaluate(self, test_loader: DataLoader) -> Dict:
        """Evaluate model on test set."""
        if self.model is None:
            self.load_model()
        
        self.logger.info("Starting evaluation...")
        
        all_preds, all_targets, all_probs = [], [], []
        total_loss = 0
        
        criterion = nn.CrossEntropyLoss()
        
        with torch.no_grad():
            for clips, labels in test_loader:
                clips, labels = clips.to(self.device), labels.to(self.device)
                
                outputs, _, _ = self.model(clips)
                loss = criterion(outputs, labels)
                total_loss += loss.item()
                
                probs = torch.softmax(outputs, dim=1)
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                
                all_preds.extend(preds)
                all_targets.extend(labels.cpu().numpy())
                all_probs.extend(probs[:, 1].cpu().numpy())
        
        # Metrics
        acc = accuracy_score(all_targets, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_targets, all_preds, average='binary', zero_division=0
        )
        
        try:
            auc = roc_auc_score(all_targets, all_probs)
        except ValueError:
            auc = 0.5
        
        cm = confusion_matrix(all_targets, all_preds)
        
        self.results = {
            'loss': total_loss / len(test_loader),
            'accuracy': acc,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc,
            'confusion_matrix': cm,
            'predictions': all_preds,
            'targets': all_targets,
            'probabilities': all_probs
        }
        
        self.logger.info("=" * 50)
        self.logger.info("EVALUATION RESULTS")
        self.logger.info("=" * 50)
        self.logger.info(f"Loss: {self.results['loss']:.4f}")
        self.logger.info(f"Accuracy: {self.results['accuracy']:.4f}")
        self.logger.info(f"Precision: {self.results['precision']:.4f}")
        self.logger.info(f"Recall: {self.results['recall']:.4f}")
        self.logger.info(f"F1-Score: {self.results['f1']:.4f}")
        self.logger.info(f"AUC-ROC: {self.results['auc']:.4f}")
        self.logger.info("=" * 50)
        
        return self.results
    
    def plot_confusion_matrix(self, save_path: str = "confusion_matrix.png"):
        """Plot and save confusion matrix."""
        if not self.results:
            self.logger.error("No results to plot. Run evaluate() first.")
            return
        
        cm = self.results['confusion_matrix']
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Non-Violence', 'Violence'],
                   yticklabels=['Non-Violence', 'Violence'])
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
        
        self.logger.info(f"Confusion matrix saved: {save_path}")
    
    def plot_roc_curve(self, save_path: str = "roc_curve.png"):
        """Plot and save ROC curve."""
        if not self.results:
            self.logger.error("No results to plot. Run evaluate() first.")
            return
        
        from sklearn.metrics import roc_curve
        
        fpr, tpr, thresholds = roc_curve(
            self.results['targets'], 
            self.results['probabilities']
        )
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {self.results["auc"]:.3f})', linewidth=2)
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
        
        self.logger.info(f"ROC curve saved: {save_path}")
    
    def print_classification_report(self):
        """Print detailed classification report."""
        if not self.results:
            self.logger.error("No results. Run evaluate() first.")
            return
        
        report = classification_report(
            self.results['targets'],
            self.results['predictions'],
            target_names=['Non-Violence', 'Violence']
        )
        
        self.logger.info("\nClassification Report:\n" + report)


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate violence detection model')
    parser.add_argument('--model', required=True, help='Path to model checkpoint')
    parser.add_argument('--test-dir', required=True, help='Test clips directory')
    parser.add_argument('--batch-size', type=int, default=4, help='Batch size')
    parser.add_argument('--config', default='configs/train_violence.yaml', help='Config file')
    args = parser.parse_args()
    
    # Create evaluator
    evaluator = ViolenceEvaluator(args.model, args.config)
    
    # Load data
    test_dataset = ViolenceDataset(args.test_dir)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Evaluate
    results = evaluator.evaluate(test_loader)
    
    # Plot
    evaluator.plot_confusion_matrix()
    evaluator.plot_roc_curve()
    evaluator.print_classification_report()
    
    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()
