#!/usr/bin/env python3
"""
In-Car Violence Detection Dashboard - Version 1.5
Real-time web dashboard with dark theme UI/UX
"""

import os
import sys
import time
import threading
from pathlib import Path
from datetime import datetime
from collections import deque

from flask import Flask, render_template, Response, jsonify
import cv2
import torch
import numpy as np
from ultralytics import YOLO

# PATH SETUP
PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
VDMIL_REPO = PROJECT_ROOT.parent / "violence-detection-mil"

sys.path.insert(0, str(PROJECT_ROOT / "src" / "models"))
sys.path.insert(0, str(VDMIL_REPO))

from vdmil_wrapper import create_vdmil_detector

app = Flask(__name__)

# GLOBAL STATE
class DetectionState:
    def __init__(self):
        self.frame = None
        self.threat_level = 0
        self.threat_name = "NORMAL"
        self.violence_score = 0.0
        self.weapon_count = 0
        self.weapon_detections = []
        self.fps = 0
        self.is_running = False
        self.alert_history = deque(maxlen=100)
        self.stats = {
            'total_frames': 0,
            'violence_frames': 0,
            'weapon_frames': 0,
            'critical_alerts': 0,
            'warning_alerts': 0,
            'watch_alerts': 0,
            'start_time': None
        }
        self.violence_history = deque(maxlen=300)
        self.threat_history = deque(maxlen=300)

    def add_alert(self, level, name, reason, violence_score):
        self.alert_history.appendleft({
            'timestamp': datetime.now().strftime('%H:%M:%S'),
            'level': level,
            'name': name,
            'reason': reason,
            'violence_score': round(violence_score, 3)
        })
        if level == 3:
            self.stats['critical_alerts'] += 1
        elif level == 2:
            self.stats['warning_alerts'] += 1
        elif level == 1:
            self.stats['watch_alerts'] += 1

state = DetectionState()

# DETECTION SYSTEM
class DashboardDetectionSystem:
    THREAT_NAMES = {0: "NORMAL", 1: "WATCH", 2: "WARNING", 3: "CRITICAL"}
    THREAT_COLORS = {0: "#00ff88", 1: "#ffd93d", 2: "#ff9500", 3: "#ff3333"}

    def __init__(self, violence_model_path, weapon_model_path, backbone_weights_dir, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"[Dashboard] Loading models on {self.device}...")

        self.violence_processor = create_vdmil_detector(
            classifier_path=violence_model_path,
            backbone_weights_dir=backbone_weights_dir,
            device=str(self.device)
        )

        self.weapon_model = YOLO(weapon_model_path)

        self.violence_score_buffer = deque(maxlen=30)
        self.current_threat = 0
        self.last_alert_time = 0
        self.alert_cooldown = 3

        print("[Dashboard] Models loaded!")

    def detect_weapons(self, frame):
        results = self.weapon_model(frame, conf=0.3, verbose=False)
        detections = []
        for result in results:
            for box in result.boxes:
                detections.append({
                    'class': result.names[int(box.cls)],
                    'confidence': float(box.conf),
                    'bbox': box.xyxy[0].cpu().numpy().astype(int).tolist()
                })
        return detections

    def detect_violence(self, frame):
        prob = self.violence_processor.process_frame(frame)
        self.violence_score_buffer.append(prob)
        return float(np.mean(self.violence_score_buffer)) if self.violence_score_buffer else prob

    def evaluate_threat(self, weapon_dets, violence_score):
        has_weapon = len(weapon_dets) > 0
        weapon_drawn = any(d['class'].lower() in ['knife', 'gun', 'pistol'] and d['confidence'] > 0.7 for d in weapon_dets)
        violence_detected = violence_score > 0.6

        if weapon_drawn and violence_detected:
            return 3, "EMERGENCY: Weapon + Violence!"
        elif violence_detected and has_weapon:
            return 2, "WARNING: Violence + Weapon"
        elif violence_detected:
            return 2, "WARNING: Violence detected"
        elif weapon_drawn:
            return 2, "WARNING: Weapon drawn"
        elif has_weapon:
            return 1, "WATCH: Weapon present"
        elif violence_score > 0.3:
            return 1, "WATCH: Suspicious activity"
        else:
            return 0, "NORMAL"

    def process_frame(self, frame):
        weapon_dets = self.detect_weapons(frame)
        violence_score = self.detect_violence(frame)
        threat_level, reason = self.evaluate_threat(weapon_dets, violence_score)

        current_time = time.time()
        if threat_level != self.current_threat or (current_time - self.last_alert_time) > self.alert_cooldown:
            if threat_level >= self.current_threat or (current_time - self.last_alert_time) > self.alert_cooldown:
                state.add_alert(threat_level, self.THREAT_NAMES[threat_level], reason, violence_score)
                self.current_threat = threat_level
                self.last_alert_time = current_time

        state.threat_level = threat_level
        state.threat_name = self.THREAT_NAMES[threat_level]
        state.violence_score = violence_score
        state.weapon_count = len(weapon_dets)
        state.weapon_detections = weapon_dets
        state.violence_history.append(violence_score)
        state.threat_history.append(threat_level)
        state.stats['total_frames'] += 1
        if violence_score > 0.3:
            state.stats['violence_frames'] += 1
        if len(weapon_dets) > 0:
            state.stats['weapon_frames'] += 1

        annotated = self.draw_dashboard(frame, weapon_dets, violence_score, threat_level)
        return annotated

    def draw_dashboard(self, frame, weapon_dets, violence_score, threat_level):
        h, w = frame.shape[:2]

        for det in weapon_dets:
            x1, y1, x2, y2 = det['bbox']
            color = (0, 0, 255) if det['class'].lower() in ['knife', 'gun', 'pistol'] else (0, 255, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"{det['class']}: {det['confidence']:.2f}"
            cv2.putText(frame, label, (x1, max(y1-10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        bar_x, bar_y = 10, h - 40
        bar_w, bar_h = 200, 15
        filled = int(bar_w * violence_score)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x+bar_w, bar_y+bar_h), (50,50,50), -1)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x+filled, bar_y+bar_h), 
                     (0, int(255*(1-violence_score)), int(255*violence_score)), -1)
        cv2.putText(frame, f"Violence: {violence_score:.2f}", (bar_x, bar_y-5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)

        colors = {0: (0,255,0), 1: (0,255,255), 2: (0,165,255), 3: (0,0,255)}
        names = {0: "NORMAL", 1: "WATCH", 2: "WARNING", 3: "CRITICAL"}
        color = colors[threat_level]
        cv2.putText(frame, f"THREAT: {names[threat_level]}", (w-200, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        return frame

# VIDEO STREAMING THREAD
def detection_thread(source, violence_model, weapon_model, backbone_weights):
    detector = DashboardDetectionSystem(violence_model, weapon_model, backbone_weights)
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print(f"[ERROR] Cannot open source: {source}")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    state.is_running = True
    state.stats['start_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    fps_counter = 0
    fps_time = time.time()

    while state.is_running:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.01)
            continue

        annotated = detector.process_frame(frame)
        state.frame = annotated

        fps_counter += 1
        if time.time() - fps_time >= 1.0:
            state.fps = fps_counter
            fps_counter = 0
            fps_time = time.time()

    cap.release()

# FLASK ROUTES
@app.route('/')
def index():
    return render_template('dashboard.html')

@app.route('/video_feed')
def video_feed():
    def generate():
        while True:
            if state.frame is not None:
                ret, buffer = cv2.imencode('.jpg', state.frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            time.sleep(0.033)
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/status')
def api_status():
    return jsonify({
        'threat_level': state.threat_level,
        'threat_name': state.threat_name,
        'threat_color': DashboardDetectionSystem.THREAT_COLORS[state.threat_level],
        'violence_score': round(state.violence_score, 3),
        'weapon_count': state.weapon_count,
        'fps': state.fps,
        'is_running': state.is_running,
        'alerts': list(state.alert_history)[:20],
        'stats': dict(state.stats),
        'violence_history': list(state.violence_history),
        'threat_history': list(state.threat_history)
    })

@app.route('/api/stats')
def api_stats():
    return jsonify(dict(state.stats))

# MAIN
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', default='0')
    parser.add_argument('--violence-model', 
                       default=str(PROJECT_ROOT.parent / "violence-detection-mil" / "models" / "checkpoints" / "violence" / "model_best.pt"))
    parser.add_argument('--weapon-model',
                       default=str(PROJECT_ROOT / "models" / "checkpoints" / "weapon" / "yolov8n_weapons" / "weights" / "best.pt"))
    parser.add_argument('--backbone-weights',
                       default=str(PROJECT_ROOT.parent / "violence-detection-mil" / "movinet_weights"))
    parser.add_argument('--port', type=int, default=5000)
    args = parser.parse_args()

    try:
        source = int(args.source)
    except ValueError:
        source = args.source

    thread = threading.Thread(target=detection_thread, 
                             args=(source, args.violence_model, args.weapon_model, args.backbone_weights))
    thread.daemon = True
    thread.start()

    print(f"\n[Dashboard] Starting server at http://localhost:{args.port}")
    print("[Dashboard] Open your browser and go to the URL above\n")

    app.run(host='0.0.0.0', port=args.port, debug=False, threaded=True)
