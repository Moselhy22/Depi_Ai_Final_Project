#!/usr/bin/env python3
import os
import sys
import json
import argparse
from pathlib import Path

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                            roc_auc_score, confusion_matrix)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, "/home/moselhy/Desktop/Depi_Ai_Final_Project/violence-detection-mil")

from vdmil_wrapper import VDMILViolenceDetector


class VDMILEvalDataset(Dataset):
    def __init__(self, clips_dir, clip_length=8):
        self.clip_length = clip_length
        self.samples = []
        
        pos_dir = os.path.join(clips_dir, "positive")
        neg_dir = os.path.join(clips_dir, "negative")
        
        if os.path.exists(pos_dir):
            for f in sorted(os.listdir(pos_dir)):
                if f.endswith(('.mp4', '.avi', '.mov')):
                    self.samples.append((os.path.join(pos_dir, f), 1))
        
        if os.path.exists(neg_dir):
            for f in sorted(os.listdir(neg_dir)):
                if f.endswith(('.mp4', '.avi', '.mov')):
                    self.samples.append((os.path.join(neg_dir, f), 0))
        
        print(f"Loaded {len(self.samples)} test clips")
        print(f"  Positive: {sum(1 for _, l in self.samples if l == 1)}")
        print(f"  Negative: {sum(1 for _, l in self.samples if l == 0)}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        video_path, label = self.samples[idx]
        import cv2
        frames = []
        cap = cv2.VideoCapture(video_path)
        while len(frames) < self.clip_length:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()
        while len(frames) < self.clip_length:
            frames.append(frames[-1] if frames else np.zeros((224, 224, 3), dtype=np.uint8))
        return frames[:self.clip_length], label


class ViolenceEvaluator:
    def __init__(self, classifier_path, backbone_weights_dir=None, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.detector = VDMILViolenceDetector(
            classifier_path=classifier_path,
            backbone_weights_dir=backbone_weights_dir,
            device=device,
            clip_length=8,
            frame_size=(224, 224)
        )
        self.results = {}
        print(f"[Evaluator] Loaded model on {self.device}")
    
    def evaluate(self, test_loader):
        print("\n[Evaluator] Starting evaluation...")
        all_preds, all_targets, all_probs = [], [], []
        
        for batch_idx, (clips, labels) in enumerate(test_loader):
            for clip, label in zip(clips, labels):
                prob = self.detector.predict_clip(clip)
                pred = 1 if prob > 0.5 else 0
                all_probs.append(prob)
                all_preds.append(pred)
                all_targets.append(label.item() if isinstance(label, torch.Tensor) else label)
            
            if (batch_idx + 1) % 10 == 0:
                print(f"  Processed {batch_idx + 1}/{len(test_loader)} batches")
        
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
            'accuracy': acc, 'precision': precision, 'recall': recall,
            'f1': f1, 'auc': auc, 'confusion_matrix': cm,
            'predictions': all_preds, 'targets': all_targets,
            'probabilities': all_probs
        }
        
        print("\n" + "=" * 50)
        print("EVALUATION RESULTS")
        print("=" * 50)
        print(f"Accuracy:  {acc:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1-Score:  {f1:.4f}")
        print(f"AUC-ROC:   {auc:.4f}")
        print("=" * 50)
        return self.results
    
    def plot_confusion_matrix(self, save_path="confusion_matrix.png"):
        if not self.results:
            return
        cm = self.results['confusion_matrix']
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Non-Violence', 'Violence'],
                   yticklabels=['Non-Violence', 'Violence'])
        plt.title('Confusion Matrix - VD-MIL')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"✅ Confusion matrix saved: {save_path}")
    
    def plot_roc_curve(self, save_path="roc_curve.png"):
        if not self.results:
            return
        from sklearn.metrics import roc_curve
        fpr, tpr, _ = roc_curve(self.results['targets'], self.results['probabilities'])
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC (AUC = {self.results["auc"]:.3f})', linewidth=2)
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve - VD-MIL')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"✅ ROC curve saved: {save_path}")
    
    def save_results(self, save_path="evaluation_results.json"):
        if not self.results:
            return
        save_dict = {
            'accuracy': float(self.results['accuracy']),
            'precision': float(self.results['precision']),
            'recall': float(self.results['recall']),
            'f1': float(self.results['f1']),
            'auc': float(self.results['auc']),
            'confusion_matrix': self.results['confusion_matrix'].tolist()
        }
        with open(save_path, 'w') as f:
            json.dump(save_dict, f, indent=2)
        print(f"✅ Results saved: {save_path}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate VD-MIL violence detection')
    parser.add_argument('--model', required=True, help='Path to model')
    parser.add_argument('--backbone-weights', help='Path to MoViNet backbone weights dir')
    parser.add_argument('--test-clips', required=True, help='Test clips directory')
    parser.add_argument('--batch-size', type=int, default=4, help='Batch size')
    parser.add_argument('--output-dir', default='evaluation_output', help='Output directory')
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    evaluator = ViolenceEvaluator(
        classifier_path=args.model,
        backbone_weights_dir=args.backbone_weights
    )
    
    test_dataset = VDMILEvalDataset(args.test_clips, clip_length=8)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )
    
    results = evaluator.evaluate(test_loader)
    evaluator.plot_confusion_matrix(os.path.join(args.output_dir, "confusion_matrix.png"))
    evaluator.plot_roc_curve(os.path.join(args.output_dir, "roc_curve.png"))
    evaluator.save_results(os.path.join(args.output_dir, "results.json"))
    
    print("\n✅ Evaluation complete!")
    print(f"Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
