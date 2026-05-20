#!/usr/bin/env python3
import os, sys, json, argparse
from pathlib import Path
import torch, numpy as np, cv2
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, "/home/moselhy/Desktop/Depi_Ai_Final_Project/violence-detection-mil")
from vdmil_wrapper import VDMILViolenceDetector

def evaluate_on_videos(test_dir, model_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    detector = VDMILViolenceDetector(
        classifier_path=model_path,
        device=device,
        clip_length=8,
        frame_size=(224, 224)
    )
    
    results = []
    video_exts = ('.mp4', '.avi', '.mov', '.mkv')
    
    for class_name, label in [("Violence", 1), ("Normal", 0)]:
        class_dir = os.path.join(test_dir, class_name)
        if not os.path.exists(class_dir):
            print(f"Warning: {class_dir} not found, skipping")
            continue
            
        videos = [f for f in os.listdir(class_dir) if f.lower().endswith(video_exts)]
        print(f"\nProcessing {len(videos)} {class_name} videos from {class_dir}...")
        
        for video_file in videos:
            video_path = os.path.join(class_dir, video_file)
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                print(f"  ⚠️ Cannot open {video_file}")
                continue
            
            frames = []
            predictions = []
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(frame)
                
                if len(frames) >= 8:
                    prob = detector.predict(frames[-8:])
                    predictions.append(prob)
                    
            cap.release()
            
            avg_prob = np.mean(predictions) if predictions else 0
            pred = 1 if avg_prob > 0.5 else 0
            
            results.append({
                'video': video_file,
                'true_label': label,
                'predicted': pred,
                'probability': float(avg_prob)
            })
            
            print(f"  {video_file}: prob={avg_prob:.3f}, pred={pred}")
    
    if not results:
        print("ERROR: No videos were processed!")
        return
    
    y_true = [r['true_label'] for r in results]
    y_pred = [r['predicted'] for r in results]
    y_prob = [r['probability'] for r in results]
    
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
    
    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc = 0.5
    
    cm = confusion_matrix(y_true, y_pred)
    
    print("\n" + "="*50)
    print("EVALUATION RESULTS (Raw Videos)")
    print("="*50)
    print(f"Total videos: {len(results)}")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"AUC-ROC:   {auc:.4f}")
    print("="*50)
    
    with open(os.path.join(output_dir, "results.json"), 'w') as f:
        json.dump({
            'accuracy': acc, 'precision': precision, 'recall': recall,
            'f1': f1, 'auc': auc, 'confusion_matrix': cm.tolist(),
            'details': results
        }, f, indent=2)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
               xticklabels=['Non-Violence', 'Violence'],
               yticklabels=['Non-Violence', 'Violence'])
    plt.title('Confusion Matrix - VD-MIL')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"), dpi=150)
    plt.close()
    
    print(f"\n✅ Results saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--test-dir', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--output-dir', default='evaluation_output_raw')
    args = parser.parse_args()
    evaluate_on_videos(args.test_dir, args.model, args.output_dir)
