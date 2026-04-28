"""
Violence detection training pipeline.
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, Any, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support, 
                            roc_auc_score, confusion_matrix)
import numpy as np

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.models.violence_detector import InCarViolenceDetector
from src.utils.config_parser import load_config
from src.utils.logger import setup_logger


class ViolenceDataset(Dataset):
    """Dataset for violence detection clips."""
    
    def __init__(self, clips_dir: str):
        self.clips_dir = Path(clips_dir)
        self.clip_files = sorted(list(self.clips_dir.glob("*.npy")))
        
        if not self.clip_files:
            raise ValueError(f"No clips found in {clips_dir}")
        
        self.labels = []
        for clip_file in self.clip_files:
            meta_file = clip_file.with_suffix('.json')
            with open(meta_file, 'r') as f:
                meta = json.load(f)
            self.labels.append(meta['label'])
    
    def __len__(self):
        return len(self.clip_files)
    
    def __getitem__(self, idx):
        clip = np.load(self.clip_files[idx])
        
        # Convert to tensor: (T, H, W, C) -> (T, C, H, W)
        clip = torch.from_numpy(clip).float().permute(0, 3, 1, 2) / 255.0
        
        # Normalize (ImageNet stats)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        clip = (clip - mean) / std
        
        label = self.labels[idx]
        
        return clip, torch.tensor(label, dtype=torch.long)


class ViolenceTrainer:
    """Trainer for violence detection model."""
    
    def __init__(self, config_path: str = "configs/train_violence.yaml"):
        self.cfg = load_config(config_path)
        self.logger = setup_logger("ViolenceTrainer")
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger.info(f"Using device: {self.device}")
        
        # Training config
        train_cfg = self.cfg.to_dict()['training']
        self.epochs = train_cfg['epochs']
        self.patience = train_cfg['early_stopping']['patience']
        self.best_val_f1 = 0
        self.patience_counter = 0
        
        # History
        self.history = {
            'train_loss': [], 'train_acc': [], 'train_f1': [],
            'val_loss': [], 'val_acc': [], 'val_f1': [], 'val_auc': []
        }
        
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
    
    def setup_model(self, num_classes: int = 2):
        """Initialize model, optimizer, loss."""
        model_cfg = self.cfg.to_dict()['model']
        
        self.model = InCarViolenceDetector(
            num_classes=num_classes,
            hidden_dim=model_cfg['hidden_dim'],
            num_layers=model_cfg['num_layers'],
            dropout=model_cfg['dropout'],
            backbone=model_cfg['backbone']
        ).to(self.device)
        
        # Optimizer
        opt_cfg = self.cfg.to_dict()['training']['optimizer']
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=opt_cfg['lr'],
            weight_decay=opt_cfg['weight_decay'],
            betas=tuple(opt_cfg['betas'])
        )
        
        # Scheduler
        sched_cfg = self.cfg.to_dict()['training']['scheduler']
        # Will be initialized after knowing steps per epoch
        
        # Loss
        loss_cfg = self.cfg.to_dict()['training']['loss']
        weights = torch.tensor(loss_cfg['class_weights']).to(self.device)
        self.criterion = nn.CrossEntropyLoss(weight=weights)
        
        self.logger.info(f"Model initialized: {sum(p.numel() for p in self.model.parameters()):,} parameters")
    
    def setup_scheduler(self, steps_per_epoch: int):
        """Setup learning rate scheduler."""
        sched_cfg = self.cfg.to_dict()['training']['scheduler']
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=sched_cfg['max_lr'],
            epochs=self.epochs,
            steps_per_epoch=steps_per_epoch,
            pct_start=sched_cfg['pct_start']
        )
    
    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        all_preds, all_targets = [], []
        
        for batch_idx, (clips, labels) in enumerate(train_loader):
            clips, labels = clips.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs, _, _ = self.model(clips)
            loss = self.criterion(outputs, labels)
            
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), 
                self.cfg.to_dict()['training']['gradient_clipping']['max_norm']
            )
            
            self.optimizer.step()
            if self.scheduler:
                self.scheduler.step()
            
            total_loss += loss.item()
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            
            all_preds.extend(preds)
            all_targets.extend(labels.cpu().numpy())
            
            if batch_idx % 10 == 0:
                self.logger.info(f"  Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")
        
        acc = accuracy_score(all_targets, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_targets, all_preds, average='binary', zero_division=0
        )
        
        return total_loss / len(train_loader), acc, f1
    
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate model."""
        self.model.eval()
        total_loss = 0
        all_preds, all_targets, all_probs = [], [], []
        
        with torch.no_grad():
            for clips, labels in val_loader:
                clips, labels = clips.to(self.device), labels.to(self.device)
                outputs, _, _ = self.model(clips)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                probs = torch.softmax(outputs, dim=1)
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                
                all_preds.extend(preds)
                all_targets.extend(labels.cpu().numpy())
                all_probs.extend(probs[:, 1].cpu().numpy())
        
        acc = accuracy_score(all_targets, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_targets, all_preds, average='binary', zero_division=0
        )
        
        # AUC
        try:
            auc = roc_auc_score(all_targets, all_probs)
        except ValueError:
            auc = 0.5
        
        return {
            'loss': total_loss / len(val_loader),
            'accuracy': acc,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc
        }
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader):
        """Full training loop."""
        self.setup_scheduler(len(train_loader))
        
        self.logger.info("=" * 60)
        self.logger.info("STARTING TRAINING")
        self.logger.info("=" * 60)
        
        for epoch in range(self.epochs):
            self.logger.info(f"\nEpoch {epoch + 1}/{self.epochs}")
            
            # Train
            train_loss, train_acc, train_f1 = self.train_epoch(train_loader)
            
            # Validate
            val_metrics = self.validate(val_loader)
            
            # Log
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['train_f1'].append(train_f1)
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_acc'].append(val_metrics['accuracy'])
            self.history['val_f1'].append(val_metrics['f1'])
            self.history['val_auc'].append(val_metrics['auc'])
            
            self.logger.info(f"Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, F1: {train_f1:.4f}")
            self.logger.info(f"Val   - Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.4f}")
            self.logger.info(f"      - Precision: {val_metrics['precision']:.4f}, Recall: {val_metrics['recall']:.4f}")
            self.logger.info(f"      - F1: {val_metrics['f1']:.4f}, AUC: {val_metrics['auc']:.4f}")
            
            # Early stopping
            if val_metrics['f1'] > self.best_val_f1:
                self.best_val_f1 = val_metrics['f1']
                self.patience_counter = 0
                self.save_checkpoint('best_model.pth', val_metrics, epoch)
                self.logger.info("✅ New best model saved!")
            else:
                self.patience_counter += 1
                self.logger.info(f"⏳ Patience: {self.patience_counter}/{self.patience}")
                
                if self.patience_counter >= self.patience:
                    self.logger.info(f"🛑 Early stopping at epoch {epoch + 1}")
                    break
        
        self.logger.info("=" * 60)
        self.logger.info("TRAINING COMPLETE")
        self.logger.info("=" * 60)
        
        return self.history
    
    def save_checkpoint(self, filename: str, metrics: Dict, epoch: int):
        """Save model checkpoint."""
        ckpt_dir = Path(self.cfg.to_dict()['logging']['checkpoint_dir'])
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
            'history': self.history
        }
        
        path = ckpt_dir / filename
        torch.save(checkpoint, path)
        self.logger.info(f"Checkpoint saved: {path}")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train violence detection model')
    parser.add_argument('--config', default='configs/train_violence.yaml', help='Config file')
    parser.add_argument('--train-dir', required=True, help='Training clips directory')
    parser.add_argument('--val-dir', required=True, help='Validation clips directory')
    parser.add_argument('--batch-size', type=int, default=None, help='Override batch size')
    args = parser.parse_args()
    
    # Load config
    trainer = ViolenceTrainer(args.config)
    
    # Override batch size if provided
    batch_size = args.batch_size or trainer.cfg.to_dict()['training']['batch_size']
    
    # Create datasets
    train_dataset = ViolenceDataset(args.train_dir)
    val_dataset = ViolenceDataset(args.val_dir)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Setup and train
    trainer.setup_model(num_classes=2)
    history = trainer.train(train_loader, val_loader)
    
    print("\nTraining complete!")


if __name__ == "__main__":
    main()
