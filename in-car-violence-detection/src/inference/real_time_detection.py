"""
Integrated real-time violence and weapon detection system for in-car monitoring.
"""

import os
import sys
import time
import argparse
from collections import deque
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import cv2
import torch
import numpy as np
from ultralytics import YOLO

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.models.violence_detector import InCarViolenceDetector
from src.utils.config_parser import load_config
from src.utils.logger import setup_logger


class IntegratedDetectionSystem:
    """
    Real-time integrated system combining:
    1. Violence detection (CNN-LSTM)
    2. Weapon detection (YOLOv8)
    3. Threat level evaluation
    4. Alert system
    """
    
    # Threat levels
    THREAT_NORMAL = 0
    THREAT_WATCH = 1
    THREAT_WARNING = 2
    THREAT_CRITICAL = 3
    
    THREAT_NAMES = {
        THREAT_NORMAL: "NORMAL",
        THREAT_WATCH: "WATCH",
        THREAT_WARNING: "WARNING",
        THREAT_CRITICAL: "CRITICAL"
    }
    
    THREAT_COLORS = {
        THREAT_NORMAL: (0, 255, 0),      # Green
        THREAT_WATCH: (0, 255, 255),       # Yellow
        THREAT_WARNING: (0, 165, 255),     # Orange
        THREAT_CRITICAL: (0, 0, 255)       # Red
    }
    
    def __init__(self,
                 violence_model_path: Optional[str] = None,
                 weapon_model_path: Optional[str] = None,
                 config_path: str = "configs/train_violence.yaml"):
        self.logger = setup_logger("IntegratedDetection")
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger.info(f"Using device: {self.device}")
        
        # Config
        self.cfg = load_config(config_path)
        prep_cfg = self.cfg.to_dict()['preprocessing']
        self.frame_size = tuple(prep_cfg['frame_size'])
        self.clip_length = prep_cfg['clip_length']
        self.target_fps = prep_cfg['target_fps']
        
        # Models
        self.violence_model = None
        self.weapon_model = None
        
        if violence_model_path:
            self.load_violence_model(violence_model_path)
        
        if weapon_model_path:
            self.load_weapon_model(weapon_model_path)
        
        # Buffers for temporal analysis
        self.frame_buffer = deque(maxlen=self.clip_length)
        self.violence_score_buffer = deque(maxlen=30)  # 3 seconds at 10 fps
        self.weapon_detection_buffer = deque(maxlen=10)
        
        # Alert state
        self.current_threat_level = self.THREAT_NORMAL
        self.last_alert_time = 0
        self.alert_cooldown = 5  # seconds between same-level alerts
        
        self.logger.info("Integrated Detection System initialized")
    
    def load_violence_model(self, model_path: str):
        """Load violence detection model."""
        self.logger.info(f"Loading violence model: {model_path}")
        
        checkpoint = torch.load(model_path, map_location=self.device)
        
        model_cfg = self.cfg.to_dict()['model']
        self.violence_model = InCarViolenceDetector(
            num_classes=2,
            hidden_dim=model_cfg['hidden_dim'],
            num_layers=model_cfg['num_layers'],
            dropout=model_cfg['dropout'],
            backbone=model_cfg['backbone']
        ).to(self.device)
        
        self.violence_model.load_state_dict(checkpoint['model_state_dict'])
        self.violence_model.eval()
        
        self.logger.info("Violence model loaded")
    
    def load_weapon_model(self, model_path: str):
        """Load weapon detection model."""
        self.logger.info(f"Loading weapon model: {model_path}")
        
        self.weapon_model = YOLO(model_path)
        
        self.logger.info("Weapon model loaded")
    
    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """Preprocess frame for violence detection."""
        resized = cv2.resize(frame, self.frame_size)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        return rgb
    
    def detect_weapons(self, frame: np.ndarray) -> List[Dict]:
        """Run weapon detection on frame."""
        if self.weapon_model is None:
            return []
        
        results = self.weapon_model(frame, conf=0.5, verbose=False)
        detections = []
        
        for result in results:
            for box in result.boxes:
                detection = {
                    'class': result.names[int(box.cls)],
                    'confidence': float(box.conf),
                    'bbox': box.xyxy[0].cpu().numpy().astype(int)
                }
                detections.append(detection)
        
        return detections
    
    def detect_violence(self, clip: List[np.ndarray]) -> float:
        """Run violence detection on clip."""
        if self.violence_model is None or len(clip) < self.clip_length:
            return 0.0
        
        # Convert to tensor
        clip_array = np.array(clip) / 255.0
        clip_tensor = torch.from_numpy(clip_array).float().permute(0, 3, 1, 2).unsqueeze(0)
        clip_tensor = clip_tensor.to(self.device)
        
        # Normalize
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1).to(self.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1).to(self.device)
        clip_tensor = (clip_tensor - mean) / std
        
        with torch.no_grad():
            output, _, _ = self.violence_model(clip_tensor)
            prob = torch.softmax(output, dim=1)
            violence_prob = prob[0][1].item()
        
        return violence_prob
    
    def evaluate_threat(self, weapon_dets: List[Dict], violence_score: float) -> Tuple[int, str]:
        """
        Evaluate threat level based on weapon and violence detections.
        
        Returns:
            threat_level: 0-3
            reason: Description of threat
        """
        has_weapon = len(weapon_dets) > 0
        weapon_drawn = any(
            d['class'] in ['knife', 'gun'] and d['confidence'] > 0.7 
            for d in weapon_dets
        )
        
        # Temporal smoothing for violence
        self.violence_score_buffer.append(violence_score)
        avg_violence = np.mean(self.violence_score_buffer) if self.violence_score_buffer else 0
        violence_detected = avg_violence > 0.6
        
        # Threat logic
        if weapon_drawn and violence_detected:
            return self.THREAT_CRITICAL, "Weapon brandished + Violence detected"
        elif violence_detected and has_weapon:
            return self.THREAT_WARNING, "Violence + Weapon present"
        elif violence_detected:
            return self.THREAT_WARNING, "Violence detected"
        elif weapon_drawn:
            return self.THREAT_WARNING, "Weapon drawn"
        elif has_weapon:
            return self.THREAT_WATCH, "Weapon detected (concealed)"
        elif avg_violence > 0.3:
            return self.THREAT_WATCH, "Suspicious activity"
        else:
            return self.THREAT_NORMAL, "Normal"
    
    def trigger_alert(self, level: int, reason: str, frame: np.ndarray):
        """Trigger alert based on threat level."""
        current_time = time.time()
        
        # Debounce
        if level == self.current_threat_level and (current_time - self.last_alert_time) < self.alert_cooldown:
            return
        
        self.current_threat_level = level
        self.last_alert_time = current_time
        
        if level == self.THREAT_WATCH:
            self.logger.info(f"[WATCH] {reason}")
        elif level == self.THREAT_WARNING:
            self.logger.warning(f"[WARNING] {reason}")
        elif level == self.THREAT_CRITICAL:
            self.logger.critical(f"[CRITICAL] {reason}")
    
    def draw_detections(self, frame: np.ndarray, weapon_dets: List[Dict], 
                       violence_score: float, threat_level: int) -> np.ndarray:
        """Draw all detections and info on frame."""
        h, w = frame.shape[:2]
        
        # Draw weapon detections
        for det in weapon_dets:
            x1, y1, x2, y2 = det['bbox']
            color = (0, 0, 255) if det['class'] in ['knife', 'gun'] else (0, 255, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"{det['class']}: {det['confidence']:.2f}"
            cv2.putText(frame, label, (x1, max(y1 - 10, 20)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Draw violence meter
        bar_x, bar_y = 10, h - 60
        bar_w, bar_h = 200, 20
        filled_w = int(bar_w * violence_score)
        
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (50, 50, 50), -1)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + filled_w, bar_y + bar_h),
                     (0, int(255 * (1 - violence_score)), int(255 * violence_score)), -1)
        cv2.putText(frame, f"Violence: {violence_score:.2f}", (bar_x, bar_y - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Draw threat level
        threat_name = self.THREAT_NAMES[threat_level]
        threat_color = self.THREAT_COLORS[threat_level]
        
        cv2.putText(frame, f"THREAT: {threat_name}", (w - 200, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, threat_color, 2)
        
        # Draw FPS
        cv2.putText(frame, "Press 'q' to quit", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return frame
    
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, int, str]:
        """
        Process a single frame through the entire pipeline.
        
        Returns:
            annotated_frame: Frame with drawings
            threat_level: Current threat level
            reason: Threat reason
        """
        # Preprocess for violence detection
        preprocessed = self.preprocess_frame(frame)
        self.frame_buffer.append(preprocessed)
        
        # Weapon detection (every frame)
        weapon_dets = self.detect_weapons(frame)
        
        # Violence detection (when buffer is full)
        violence_score = 0.0
        if len(self.frame_buffer) == self.clip_length:
            violence_score = self.detect_violence(list(self.frame_buffer))
        
        # Evaluate threat
        threat_level, reason = self.evaluate_threat(weapon_dets, violence_score)
        
        # Trigger alert
        self.trigger_alert(threat_level, reason, frame)
        
        # Draw
        annotated = self.draw_detections(frame, weapon_dets, violence_score, threat_level)
        
        return annotated, threat_level, reason
    
    def run(self, source=0):
        """
        Run real-time detection.
        
        Args:
            source: Camera index (0) or video file path
        """
        cap = cv2.VideoCapture(source)
        
        if not cap.isOpened():
            self.logger.error(f"Cannot open source: {source}")
            return
        
        self.logger.info("Starting real-time detection...")
        self.logger.info("Press 'q' to quit")
        
        fps_counter = 0
        fps_time = time.time()
        current_fps = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame
            annotated, threat_level, reason = self.process_frame(frame)
            
            # Calculate FPS
            fps_counter += 1
            if time.time() - fps_time >= 1.0:
                current_fps = fps_counter
                fps_counter = 0
                fps_time = time.time()
            
            # Draw FPS
            cv2.putText(annotated, f"FPS: {current_fps}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # Show
            cv2.imshow('In-Car Violence Detection', annotated)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        self.logger.info("Detection stopped")
    
    def process_video(self, video_path: str, output_path: Optional[str] = None):
        """
        Process a video file and save results.
        
        Args:
            video_path: Input video path
            output_path: Output video path (optional)
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            self.logger.error(f"Cannot open video: {video_path}")
            return
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Setup writer if output path provided
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        self.logger.info(f"Processing video: {video_path}")
        
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            annotated, threat_level, reason = self.process_frame(frame)
            
            if writer:
                writer.write(annotated)
            
            frame_count += 1
            if frame_count % 30 == 0:
                self.logger.info(f"Processed {frame_count} frames...")
        
        cap.release()
        if writer:
            writer.release()
        
        self.logger.info(f"Video processing complete: {frame_count} frames")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Real-time violence detection')
    parser.add_argument('--source', default=0, help='Camera index or video path')
    parser.add_argument('--violence-model', help='Path to violence model')
    parser.add_argument('--weapon-model', help='Path to weapon model')
    parser.add_argument('--output', help='Output video path (for file input)')
    args = parser.parse_args()
    
    # Create system
    system = IntegratedDetectionSystem(
        violence_model_path=args.violence_model,
        weapon_model_path=args.weapon_model
    )
    
    # Determine source type
    try:
        source = int(args.source)
        is_camera = True
    except ValueError:
        source = args.source
        is_camera = False
    
    if is_camera:
        system.run(source)
    else:
        system.process_video(source, args.output)


if __name__ == "__main__":
    main()
