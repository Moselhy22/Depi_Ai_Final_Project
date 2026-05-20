#!/usr/bin/env python3
"""
Integrated Real-Time Violence & Weapon Detection System - VD-MIL Version
======================================================================
Replaces old CNN-LSTM violence detector with VD-MIL (MoViNet A0 + MIL).
Runs both YOLOv8 weapon detection and VD-MIL violence detection in real-time.

Usage:
    # Webcam:
    PYTHONPATH="${PYTHONPATH}:$(pwd)/src/models:$(pwd)/../violence-detection-mil"         python3 -B src/inference/real_time_detection_vdmil.py --source 0

    # Video file:
    PYTHONPATH="${PYTHONPATH}:$(pwd)/src/models:$(pwd)/../violence-detection-mil"         python3 -B src/inference/real_time_detection_vdmil.py         --source /path/to/video.mp4 --output /path/to/output.mp4
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

# ------------------------------------------------------------------
# PATH SETUP (Critical for imports)
# ------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
VDMIL_REPO = PROJECT_ROOT.parent / "violence-detection-mil"

sys.path.insert(0, str(PROJECT_ROOT / "src" / "models"))
sys.path.insert(0, str(VDMIL_REPO))

from vdmil_wrapper import create_vdmil_detector


class IntegratedDetectionSystemVDMIL:
    """
    Real-time integrated system combining:
      1. Violence detection (VD-MIL: MoViNet A0 + MIL classifier)
      2. Weapon detection (YOLOv8n)
      3. Threat level evaluation
      4. Alert system + visualization
    """

    # Threat levels
    THREAT_NORMAL   = 0
    THREAT_WATCH    = 1
    THREAT_WARNING  = 2
    THREAT_CRITICAL = 3

    THREAT_NAMES = {
        THREAT_NORMAL:   "NORMAL",
        THREAT_WATCH:    "WATCH",
        THREAT_WARNING:  "WARNING",
        THREAT_CRITICAL: "CRITICAL"
    }

    THREAT_COLORS_BGR = {
        THREAT_NORMAL:   (0, 255, 0),    # Green
        THREAT_WATCH:    (0, 255, 255),  # Yellow
        THREAT_WARNING:  (0, 165, 255),  # Orange
        THREAT_CRITICAL: (0, 0, 255)     # Red
    }

    def __init__(self,
                 violence_model_path: Optional[str] = None,
                 weapon_model_path: Optional[str] = None,
                 backbone_weights_dir: Optional[str] = None,
                 device: str = 'cuda',
                 clip_length: int = 8,
                 frame_size: Tuple[int, int] = (172, 172),
                 stride: int = 4):
        """
        Args:
            violence_model_path: Path to VD-MIL classifier .pt file
            weapon_model_path:   Path to YOLOv8 best.pt
            backbone_weights_dir: Path to MoViNet weights directory
            device: 'cuda' or 'cpu'
            clip_length: Number of frames per clip (default 8)
            frame_size: (H, W) for violence model preprocessing (default 172x172)
            stride: Process violence prediction every N frames
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"[System] Using device: {self.device}")

        # VD-MIL parameters
        self.clip_length = clip_length
        self.frame_size = frame_size
        self.stride = stride

        # Models
        self.violence_processor = None
        self.weapon_model = None

        if violence_model_path:
            self.load_violence_model(violence_model_path, backbone_weights_dir)

        if weapon_model_path:
            self.load_weapon_model(weapon_model_path)

        # Temporal smoothing buffers
        self.violence_score_buffer = deque(maxlen=30)   # ~3 sec at 10 fps
        self.weapon_detection_buffer = deque(maxlen=10)

        # Alert state
        self.current_threat_level = self.THREAT_NORMAL
        self.last_alert_time = 0
        self.alert_cooldown = 3  # seconds between same-level alerts

        # FPS tracking
        self.fps_counter = 0
        self.fps_time = time.time()
        self.current_fps = 0

        print("[System] Integrated Detection System initialized")

    # ------------------------------------------------------------------
    # Model Loading
    # ------------------------------------------------------------------
    def load_violence_model(self, model_path: str, backbone_weights_dir: Optional[str] = None):
        """Load VD-MIL violence detector via wrapper."""
        print(f"[VD-MIL] Loading violence model: {model_path}")

        self.violence_processor = create_vdmil_detector(
            classifier_path=model_path,
            backbone_weights_dir=backbone_weights_dir,
            device=str(self.device)
        )
        print("[VD-MIL] Violence model loaded successfully")

    def load_weapon_model(self, model_path: str):
        """Load YOLOv8 weapon detector."""
        print(f"[YOLO] Loading weapon model: {model_path}")
        self.weapon_model = YOLO(model_path)
        print("[YOLO] Weapon model loaded successfully")

    # ------------------------------------------------------------------
    # Detection Methods
    # ------------------------------------------------------------------
    def detect_weapons(self, frame: np.ndarray) -> List[Dict]:
        """Run YOLOv8 weapon detection on a single frame."""
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

    def detect_violence(self, frame: np.ndarray) -> float:
        """
        Feed frame into VD-MIL processor.
        Returns smoothed violence probability [0, 1].
        """
        if self.violence_processor is None:
            return 0.0

        prob = self.violence_processor.process_frame(frame)
        self.violence_score_buffer.append(prob)

        # Temporal smoothing: use mean of recent predictions
        avg_prob = np.mean(self.violence_score_buffer) if self.violence_score_buffer else prob
        return float(avg_prob)

    # ------------------------------------------------------------------
    # Threat Evaluation
    # ------------------------------------------------------------------
    def evaluate_threat(self, weapon_dets: List[Dict], violence_score: float) -> Tuple[int, str]:
        """
        Evaluate threat level based on weapon + violence detections.

        Logic:
            Weapon + Violence  -> CRITICAL (EMERGENCY)
            Violence only      -> WARNING
            Weapon only        -> WATCH
            Nothing            -> NORMAL
        """
        has_weapon = len(weapon_dets) > 0

        # High-confidence weapon drawn?
        weapon_drawn = any(
            d['class'].lower() in ['knife', 'gun', 'pistol'] and d['confidence'] > 0.7
            for d in weapon_dets
        )

        # Violence threshold (smoothed)
        violence_detected = violence_score > 0.6

        # Threat logic
        if weapon_drawn and violence_detected:
            return self.THREAT_CRITICAL, "EMERGENCY: Weapon brandished + Violence detected!"
        elif violence_detected and has_weapon:
            return self.THREAT_WARNING, "WARNING: Violence + Weapon present"
        elif violence_detected:
            return self.THREAT_WARNING, "WARNING: Violence detected"
        elif weapon_drawn:
            return self.THREAT_WARNING, "WARNING: Weapon drawn"
        elif has_weapon:
            return self.THREAT_WATCH, "WATCH: Weapon detected (concealed)"
        elif violence_score > 0.3:
            return self.THREAT_WATCH, "WATCH: Suspicious activity"
        else:
            return self.THREAT_NORMAL, "NORMAL"

    def trigger_alert(self, level: int, reason: str):
        """Log alert with debounce (cooldown between same-level alerts)."""
        current_time = time.time()

        if level == self.current_threat_level and (current_time - self.last_alert_time) < self.alert_cooldown:
            return

        self.current_threat_level = level
        self.last_alert_time = current_time

        prefix = {
            self.THREAT_WATCH:    "[WATCH]",
            self.THREAT_WARNING:  "[WARNING]",
            self.THREAT_CRITICAL: "[CRITICAL]"
        }.get(level, "[INFO]")

        print(f"{prefix} {reason}")

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------
    def draw_detections(self, frame: np.ndarray,
                        weapon_dets: List[Dict],
                        violence_score: float,
                        threat_level: int) -> np.ndarray:
        """Draw bounding boxes, violence meter, threat level, and FPS."""
        h, w = frame.shape[:2]

        # --- Weapon bounding boxes ---
        for det in weapon_dets:
            x1, y1, x2, y2 = det['bbox']
            is_dangerous = det['class'].lower() in ['knife', 'gun', 'pistol']
            color = (0, 0, 255) if is_dangerous else (0, 255, 255)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"{det['class']}: {det['confidence']:.2f}"
            cv2.putText(frame, label, (x1, max(y1 - 10, 20)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # --- Violence probability meter (bottom-left) ---
        bar_x, bar_y = 10, h - 80
        bar_w, bar_h = 250, 25
        filled_w = int(bar_w * violence_score)

        # Background
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (50, 50, 50), -1)
        # Fill (green -> red)
        fill_color = (0, int(255 * (1 - violence_score)), int(255 * violence_score))
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + filled_w, bar_y + bar_h), fill_color, -1)
        # Border
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (255, 255, 255), 1)
        # Label
        cv2.putText(frame, f"Violence: {violence_score:.2f}", (bar_x, bar_y - 8),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

        # --- Threat level banner (top-right) ---
        threat_name = self.THREAT_NAMES[threat_level]
        threat_color = self.THREAT_COLORS_BGR[threat_level]

        # Draw filled rectangle for visibility
        text_size = cv2.getTextSize(f"THREAT: {threat_name}", cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
        banner_x2 = w - 10
        banner_x1 = banner_x2 - text_size[0] - 20
        banner_y1 = 10
        banner_y2 = banner_y1 + text_size[1] + 20

        cv2.rectangle(frame, (banner_x1, banner_y1), (banner_x2, banner_y2), threat_color, -1)
        cv2.putText(frame, f"THREAT: {threat_name}", (banner_x1 + 10, banner_y2 - 8),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # --- FPS (top-left) ---
        fps_color = (0, 255, 0) if self.current_fps > 15 else (0, 165, 255) if self.current_fps > 8 else (0, 0, 255)
        cv2.putText(frame, f"FPS: {self.current_fps}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, fps_color, 2)

        # --- Help text ---
        cv2.putText(frame, "Press 'q' to quit | 'r' to reset buffers", (10, h - 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

        return frame

    # ------------------------------------------------------------------
    # Frame Processing Pipeline
    # ------------------------------------------------------------------
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, int, str, float]:
        """
        Process a single frame through the entire pipeline.

        Returns:
            annotated_frame, threat_level, reason, violence_score
        """
        # Weapon detection (every frame, fast)
        weapon_dets = self.detect_weapons(frame)

        # Violence detection (buffered, every stride frames)
        violence_score = self.detect_violence(frame)

        # Evaluate threat
        threat_level, reason = self.evaluate_threat(weapon_dets, violence_score)

        # Trigger alert (with debounce)
        self.trigger_alert(threat_level, reason)

        # Update FPS
        self.fps_counter += 1
        if time.time() - self.fps_time >= 1.0:
            self.current_fps = self.fps_counter
            self.fps_counter = 0
            self.fps_time = time.time()

        # Draw everything
        annotated = self.draw_detections(frame, weapon_dets, violence_score, threat_level)

        return annotated, threat_level, reason, violence_score

    # ------------------------------------------------------------------
    # Run Modes
    # ------------------------------------------------------------------
    def run_webcam(self, source: int = 0):
        """Run real-time detection from webcam."""
        cap = cv2.VideoCapture(source)

        if not cap.isOpened():
            print(f"[ERROR] Cannot open webcam: {source}")
            return

        # Set resolution for performance
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        print("[System] Starting real-time detection from webcam...")
        print("[System] Press 'q' to quit | 'r' to reset buffers")

        while True:
            ret, frame = cap.read()
            if not ret:
                print("[WARN] Frame capture failed, retrying...")
                time.sleep(0.1)
                continue

            annotated, threat_level, reason, _ = self.process_frame(frame)

            cv2.imshow('In-Car Violence Detection (VD-MIL)', annotated)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                if self.violence_processor:
                    self.violence_processor.reset()
                self.violence_score_buffer.clear()
                self.weapon_detection_buffer.clear()
                print("[System] Buffers reset")

        cap.release()
        cv2.destroyAllWindows()
        print("[System] Detection stopped")

    def run_video(self, video_path: str, output_path: Optional[str] = None):
        """Process a video file and optionally save annotated output."""
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            print(f"[ERROR] Cannot open video: {video_path}")
            return

        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"[System] Processing video: {video_path}")
        print(f"[System] {width}x{height} @ {fps}fps | {total_frames} frames")

        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            print(f"[System] Saving output to: {output_path}")

        frame_count = 0
        start_time = time.time()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            annotated, threat_level, reason, violence_score = self.process_frame(frame)

            if writer:
                writer.write(annotated)

            frame_count += 1

            # Progress + preview
            if frame_count % 30 == 0:
                elapsed = time.time() - start_time
                progress = (frame_count / total_frames * 100) if total_frames > 0 else 0
                print(f"  Frame {frame_count}/{total_frames} ({progress:.1f}%) | "
                      f"FPS: {self.current_fps} | Threat: {self.THREAT_NAMES[threat_level]}")

            # Show preview window (optional, can be disabled for speed)
            cv2.imshow('Processing...', annotated)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("[System] Interrupted by user")
                break

        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()

        elapsed = time.time() - start_time
        print(f"[System] Complete: {frame_count} frames in {elapsed:.1f}s "
              f"({frame_count/elapsed:.1f} fps avg)")

    def reset(self):
        """Reset all buffers and state."""
        if self.violence_processor:
            self.violence_processor.reset()
        self.violence_score_buffer.clear()
        self.weapon_detection_buffer.clear()
        self.current_threat_level = self.THREAT_NORMAL
        self.last_alert_time = 0
        print("[System] All buffers and state reset")


# ===================================================================
# Main Entry Point
# ===================================================================
def main():
    parser = argparse.ArgumentParser(
        description='Real-time In-Car Violence & Weapon Detection (VD-MIL)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Webcam (default):
  python real_time_detection_vdmil.py --source 0

  # Video file:
  python real_time_detection_vdmil.py --source video.mp4 --output out.mp4

  # Custom model paths:
  python real_time_detection_vdmil.py \
      --violence-model /path/to/model_best.pt \
      --weapon-model /path/to/best.pt \
      --source 0
        """
    )

    parser.add_argument('--source', default='0',
                       help='Camera index (0) or video file path')
    parser.add_argument('--violence-model',
                       default=str(PROJECT_ROOT.parent / "violence-detection-mil" /
                                   "models" / "checkpoints" / "violence" / "model_best.pt"),
                       help='Path to VD-MIL classifier checkpoint')
    parser.add_argument('--weapon-model',
                       default=str(PROJECT_ROOT / "models" / "checkpoints" / "weapon" /
                                   "yolov8n_weapons" / "best.pt"),
                       help='Path to YOLOv8 weapon model')
    parser.add_argument('--backbone-weights',
                       default=str(PROJECT_ROOT.parent / "violence-detection-mil" / "movinet_weights"),
                       help='Path to MoViNet backbone weights directory')
    parser.add_argument('--output', help='Output video path (for file input)')
    parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'],
                       help='Device to run models on')
    parser.add_argument('--clip-length', type=int, default=8,
                       help='Number of frames per violence clip')
    parser.add_argument('--stride', type=int, default=4,
                       help='Process violence prediction every N frames')

    args = parser.parse_args()

    # Determine source type
    try:
        source = int(args.source)
        is_camera = True
    except ValueError:
        source = args.source
        is_camera = False

    # Validate model paths
    if not os.path.exists(args.violence_model):
        print(f"[ERROR] Violence model not found: {args.violence_model}")
        print("[HINT] Use --violence-model to specify correct path")
        sys.exit(1)

    if not os.path.exists(args.weapon_model):
        print(f"[ERROR] Weapon model not found: {args.weapon_model}")
        print("[HINT] Use --weapon-model to specify correct path")
        sys.exit(1)

    # Create system
    system = IntegratedDetectionSystemVDMIL(
        violence_model_path=args.violence_model,
        weapon_model_path=args.weapon_model,
        backbone_weights_dir=args.backbone_weights,
        device=args.device,
        clip_length=args.clip_length,
        stride=args.stride
    )

    # Run
    if is_camera:
        system.run_webcam(source)
    else:
        system.run_video(source, args.output)


if __name__ == "__main__":
    main()
