"""
violence_weapon.py  —  Violence & Weapon Detection Module
Standalone module — can be imported by app.py or run independently.

Detection pipeline:
  - Violence : MoViNet A0 backbone + MIL classifier (VD-MIL)
  - Weapon   : YOLOv8n fine-tuned on weapons dataset

Threat levels:
  NORMAL   (0) : No threats
  WATCH    (1) : Suspicious activity / Weapon detected (concealed)
  WARNING  (2) : Violence detected / Weapon drawn
  CRITICAL (3) : Weapon brandished + Violence detected
"""

import os
import sys
import time
import threading
import collections
from typing import Dict, List, Tuple, Optional

import cv2
import numpy as np
import torch

# ── Paths ──────────────────────────────────────────────────────────────
_PROJECT_ROOT = r"D:\DEPI GP"

VIOLENCE_MODEL_PATH = os.path.join(
    _PROJECT_ROOT, "models", "violance and gun", "violance", "model_best.pt"
)
WEAPON_MODEL_PATH = os.path.join(
    _PROJECT_ROOT, "models", "violance and gun", "gun", "best.pt"
)
MOVINET_WEIGHTS_DIR = os.path.join(
    _PROJECT_ROOT, "models", "violance and gun", "movinets"
)

VDMIL_REPO = os.path.join(_PROJECT_ROOT, "violence-detection-mil")


# ═══════════════════════════════════════════════════════════════════════
#  Auto-download movinets on first run
# ═══════════════════════════════════════════════════════════════════════
def _ensure_movinets():
    """Download movinets package if not present."""
    if os.path.isdir(os.path.join(MOVINET_WEIGHTS_DIR, "movinets")):
        return True

    # If the full violence-detection-mil repo exists, copy movinets from it
    if os.path.isdir(os.path.join(VDMIL_REPO, "movinets")):
        import shutil
        print("[VW] Copying movinets from violence-detection-mil repo...")
        try:
            dest = os.path.join(MOVINET_WEIGHTS_DIR, "movinets")
            if not os.path.exists(dest):
                shutil.copytree(os.path.join(VDMIL_REPO, "movinets"), dest)
            print("[VW] movinets copied successfully.")
            return True
        except Exception as e:
            print(f"[VW] Failed to copy movinets: {e}")

    # Try downloading from the original repo
    print("[VW] movinets not found. Attempting to clone from GitHub...")
    try:
        import subprocess
        os.makedirs(MOVINET_WEIGHTS_DIR, exist_ok=True)
        subprocess.check_call([
            sys.executable, "-m", "pip", "install",
            "git+https://github.com/rogerfigueroa/violence-detection-mil.git",
            "--target", MOVINET_WEIGHTS_DIR, "--quiet"
        ])
        print("[VW] violence-detection-mil package installed.")
        return True
    except Exception as e:
        print(f"[VW] Failed to install violence-detection-mil: {e}")
        print("[VW] Please manually place the 'movinets' folder in:")
        print(f"    {MOVINET_WEIGHTS_DIR}")
        return False


# ═══════════════════════════════════════════════════════════════════════
#  MIL Classifier (inline fallback)
# ═══════════════════════════════════════════════════════════════════════
import torch.nn as nn

class Net(nn.Module):
    """MIL classifier: 2048 -> 1024 -> 512 -> 1 (sigmoid)."""
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x


# ═══════════════════════════════════════════════════════════════════════
#  VD-MIL Violence Detector
# ═══════════════════════════════════════════════════════════════════════
class VDMILViolenceDetector:
    """MoViNet A0 + MIL classifier for violence detection."""

    def __init__(self, classifier_path: str, backbone_weights_dir: str,
                 device: str = 'cuda', clip_length: int = 8,
                 frame_size: Tuple[int, int] = (172, 172), stride: int = 4):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.clip_length = clip_length
        self.frame_size = frame_size
        self.stride = stride
        self.frame_buffer: List[np.ndarray] = []

        # Add movinets to path
        movinets_path = os.path.join(backbone_weights_dir, "movinets")
        if os.path.isdir(movinets_path):
            sys.path.insert(0, backbone_weights_dir)
            sys.path.insert(0, movinets_path)

        print(f"[VW] Loading violence model on {self.device}...")

        # Import movinets
        from movinets import MoViNet
        from movinets.config import _C

        # Load MoViNet backbone
        self.model_movinet = MoViNet(
            _C.MODEL.MoViNetA0, causal=True, pretrained=True,
            model_dir=backbone_weights_dir
        )
        self.model_movinet.classifier[3] = nn.Identity(
            54, unused_argument1=0.1, unused_argument2=False
        )
        self.model_movinet.to(self.device)
        self.model_movinet.eval()

        # Load MIL classifier
        self.classifiers = []
        checkpoint = torch.load(
            classifier_path, map_location=self.device, weights_only=False
        )

        if hasattr(checkpoint, 'forward'):
            checkpoint.eval()
            self.classifiers.append(checkpoint.to(self.device))
            print("[VW] Loaded Net classifier (full model)")
        else:
            # Try loading as state dict
            try:
                movinets_cls_path = os.path.join(backbone_weights_dir, "movinets")
                if os.path.exists(os.path.join(movinets_cls_path, "movinet_classifier.py")):
                    sys.path.insert(0, movinets_cls_path)
                from movinet_classifier import Net
                net = Net()
                net.load_state_dict(checkpoint)
                net.eval()
                self.classifiers.append(net.to(self.device))
                print("[VW] Loaded classifier from state_dict")
            except Exception:
                # Fallback: use checkpoint directly
                checkpoint.eval()
                self.classifiers.append(checkpoint.to(self.device))
                print("[VW] Loaded classifier (fallback)")

        print("[VW] Violence model loaded successfully!")

    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """Resize frame to 172x172 RGB."""
        resized = cv2.resize(frame, (172, 172), cv2.INTER_AREA)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        return rgb

    def update_buffer(self, frame: np.ndarray):
        """Add frame to rolling buffer."""
        self.frame_buffer.append(self.preprocess_frame(frame))
        if len(self.frame_buffer) > self.clip_length:
            self.frame_buffer = self.frame_buffer[-self.clip_length:]

    def predict(self) -> float:
        """Run inference on current buffer. Returns violence probability [0,1]."""
        if len(self.frame_buffer) < self.clip_length:
            return 0.0

        clip_frames = self.frame_buffer[-self.clip_length:]
        list_tensor = [torch.from_numpy(img)[None] for img in clip_frames]
        tensor_clip = torch.cat(list_tensor, axis=0)
        tensor_clip = tensor_clip.to(self.device)
        tensor_clip = tensor_clip.permute(3, 0, 1, 2)
        tensor_clip = tensor_clip[None]
        tensor_clip = tensor_clip.to(torch.float32)
        tensor_clip = tensor_clip / 255.0

        with torch.no_grad():
            self.model_movinet.clean_activation_buffers()
            xf = self.model_movinet(tensor_clip)
            norm_xf = torch.nn.functional.normalize(xf, dim=1, p=2)
            scores = []
            for net in self.classifiers:
                y = net(norm_xf)
                scores.append(y.to('cpu').numpy())
            scores = np.concatenate(scores, axis=1)
            prob = float(scores[0][0])

        return prob

    def reset_buffer(self):
        self.frame_buffer = []


class ViolenceWeaponDetector:
    """
    Integrated Violence + Weapon detection module.
    Designed for per-frame processing from a streaming route.
    """

    # Threat levels
    THREAT_NORMAL   = 0
    THREAT_WATCH    = 1
    THREAT_WARNING  = 2
    THREAT_CRITICAL = 3

    THREAT_NAMES = {
        0: "NORMAL", 1: "WATCH", 2: "WARNING", 3: "CRITICAL"
    }
    THREAT_COLORS = {
        0: "#00e5a0", 1: "#ffa502", 2: "#ff9500", 3: "#ff4757"
    }

    def __init__(self, violence_model_path: str = None,
                 weapon_model_path: str = None,
                 backbone_weights_dir: str = None,
                 device: str = 'cuda'):
        self.ready = False
        self.violence_detector = None
        self.weapon_model = None
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        # Rolling state
        self.violence_score_buffer = collections.deque(maxlen=30)
        self.current_threat = self.THREAT_NORMAL
        self.last_alert_time = 0
        self.alert_cooldown = 3  # seconds between alerts

        # FPS tracking
        self.frame_count = 0
        self.fps_timer = time.time()
        self.display_fps = 0.0

        # Thread-safe state
        self._state_lock = threading.Lock()
        self._latest_state = {
            "threat_level": 0,
            "threat_name": "NORMAL",
            "threat_color": "#00e5a0",
            "violence_score": 0.0,
            "weapon_count": 0,
            "weapon_detections": [],
            "fps": 0.0,
            "alert": None,
            "is_running": False,
        }

        try:
            # ── Ensure movinets available ──
            if backbone_weights_dir is None:
                backbone_weights_dir = MOVINET_WEIGHTS_DIR

            _ensure_movinets()

            # ── Load Violence model ──
            if violence_model_path is None:
                violence_model_path = VIOLENCE_MODEL_PATH

            if os.path.exists(violence_model_path):
                try:
                    self.violence_detector = VDMILViolenceDetector(
                        classifier_path=violence_model_path,
                        backbone_weights_dir=backbone_weights_dir,
                        device=str(self.device)
                    )
                except Exception as ve:
                    print(f"[VW] Violence detector failed to load: {ve}")
                    print("[VW] Continuing with weapon-only detection.")
                    self.violence_detector = None
            else:
                print(f"[VW] Violence model not found: {violence_model_path}")
                print("[VW] Violence detection disabled.")

            # ── Load Weapon model (YOLO) ──
            if weapon_model_path is None:
                weapon_model_path = WEAPON_MODEL_PATH

            if os.path.exists(weapon_model_path):
                from ultralytics import YOLO
                self.weapon_model = YOLO(weapon_model_path)
                print("[VW] Weapon model loaded.")
            else:
                print(f"[VW] Weapon model not found: {weapon_model_path}")
                print("[VW] Weapon detection disabled.")

            self.ready = True
            print("[VW] Violence+Weapon detector initialized.")

        except Exception as e:
            print(f"[VW] Init failed: {e}")
            import traceback
            traceback.print_exc()
            self.ready = False

    # ── Weapon confidence threshold
    WEAPON_CONF_THRESHOLD = 0.20

    def detect_weapons(self, frame: np.ndarray) -> List[Dict]:
        """Run YOLO weapon detection on frame."""
        if self.weapon_model is None:
            return []
        results = self.weapon_model(frame, conf=self.WEAPON_CONF_THRESHOLD, verbose=False)
        detections = []
        for result in results:
            for box in result.boxes:
                cls_name = result.names[int(box.cls)]
                conf = float(box.conf)
                detections.append({
                    'class': cls_name,
                    'confidence': conf,
                    'bbox': box.xyxy[0].cpu().numpy().astype(int).tolist()
                })
        return detections

    def detect_violence(self, frame: np.ndarray) -> float:
        """Run violence detection on frame. Returns smoothed probability."""
        if self.violence_detector is None:
            return 0.0
        self.violence_detector.update_buffer(frame)
        self.frame_count += 1
        if self.frame_count % 4 == 0:
            prob = self.violence_detector.predict()
            self.violence_score_buffer.append(prob)
        return float(np.mean(self.violence_score_buffer)) if self.violence_score_buffer else 0.0

    SHARP_WEAPON_CLASSES = {
        'knife', 'dagger', 'blade', 'sword', 'machete',
        'gun', 'pistol', 'rifle', 'handgun', 'firearm', 'weapon'
    }

    def evaluate_threat(self, weapon_dets: List[Dict], violence_score: float) -> Tuple[int, str]:
        """Evaluate threat level based on weapon and violence detections."""
        has_weapon = len(weapon_dets) > 0
        weapon_drawn = any(
            d['class'].lower() in self.SHARP_WEAPON_CLASSES and d['confidence'] > 0.50
            for d in weapon_dets
        )
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

    def process_frame(self, frame: np.ndarray):
        """
        Process a single frame through the entire pipeline.

        Returns:
            annotated_frame (np.ndarray): Frame with detections drawn
            state (dict): Current detection state
        """
        h, w = frame.shape[:2]
        now = time.time()

        # FPS tracking
        self.frame_count += 1
        if now - self.fps_timer >= 1.0:
            self.display_fps = self.frame_count / (now - self.fps_timer + 1e-6)
            self.fps_timer = now
            self.frame_count = 0

        # Detection
        weapon_dets = self.detect_weapons(frame)
        violence_score = self.detect_violence(frame)
        threat_level, reason = self.evaluate_threat(weapon_dets, violence_score)

        # Alert debounce
        alert = None
        if threat_level != self.current_threat or (now - self.last_alert_time) > self.alert_cooldown:
            if threat_level >= self.current_threat:
                alert = reason
                self.current_threat = threat_level
                self.last_alert_time = now

        # Draw weapon bounding boxes
        annotated = frame.copy()
        for det in weapon_dets:
            x1, y1, x2, y2 = det['bbox']
            color = (0, 0, 255) if det['class'].lower() in ['knife', 'gun', 'pistol'] else (0, 255, 255)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            label = f"{det['class']}: {det['confidence']:.0%}"
            cv2.putText(annotated, label, (x1, max(y1 - 10, 20)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)

        # Draw violence meter bar
        bar_x, bar_y = 10, h - 45
        bar_w, bar_h = 200, 18
        filled = int(bar_w * min(violence_score, 1.0))
        cv2.rectangle(annotated, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (50, 50, 50), -1)
        bar_color = (0, int(255 * (1 - violence_score)), int(255 * violence_score))
        cv2.rectangle(annotated, (bar_x, bar_y), (bar_x + filled, bar_y + bar_h), bar_color, -1)
        cv2.putText(annotated, f"Violence: {violence_score:.2f}",
                   (bar_x, bar_y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

        # Draw threat level indicator
        threat_colors_bgr = {0: (0, 229, 160), 1: (255, 165, 2), 2: (255, 149, 0), 3: (255, 71, 87)}
        threat_name = self.THREAT_NAMES[threat_level]
        threat_color = threat_colors_bgr[threat_level]
        cv2.putText(annotated, f"THREAT: {threat_name}", (w - 220, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, threat_color, 2, cv2.LINE_AA)

        # Alert banner at bottom
        if alert and threat_level >= 2:
            overlay = annotated.copy()
            cv2.rectangle(overlay, (0, h - 80), (w, h), (0, 0, 160), -1)
            cv2.addWeighted(overlay, 0.55, annotated, 0.45, 0, annotated)
            cv2.putText(annotated, f"  {alert}", (w // 2 - 200, h - 28),
                       cv2.FONT_HERSHEY_DUPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)

        # Build state
        state = {
            "threat_level": threat_level,
            "threat_name": threat_name,
            "threat_color": self.THREAT_COLORS[threat_level],
            "violence_score": round(violence_score, 3),
            "weapon_count": len(weapon_dets),
            "weapon_detections": weapon_dets,
            "fps": round(self.display_fps, 1),
            "alert": alert,
            "is_running": True,
        }

        with self._state_lock:
            self._latest_state = dict(state)

        return annotated, state

    def get_state(self) -> dict:
        """Return latest cached state (thread-safe)."""
        with self._state_lock:
            return dict(self._latest_state)


# ═══════════════════════════════════════════════════════════════════════
#  Module-level API (used by app.py — lazy loading)
# ═══════════════════════════════════════════════════════════════════════
_detector: Optional[ViolenceWeaponDetector] = None
_detector_lock = threading.Lock()
_load_error: Optional[str] = None


def load_detector(device: str = 'cuda') -> bool:
    """Initialize the violence+weapon detector. Call once."""
    global _detector, _load_error
    with _detector_lock:
        if _detector is not None and _detector.ready:
            return True
        if _load_error is not None:
            return False
        try:
            _detector = ViolenceWeaponDetector(device=device)
            if not _detector.ready:
                _load_error = "Detector init returned not ready"
                _detector = None
                return False
            return True
        except Exception as e:
            _load_error = str(e)
            _detector = None
            print(f"[VW] Load failed: {e}")
            return False


def process_frame(frame: np.ndarray):
    """Process a single frame. Returns (annotated, state)."""
    if _detector is None:
        if not load_detector():
            return frame, {
                "threat_level": 0, "threat_name": "OFFLINE",
                "threat_color": "#6b7a90", "violence_score": 0,
                "weapon_count": 0, "weapon_detections": [],
                "fps": 0, "alert": "Model not loaded",
                "is_running": False
            }
    return _detector.process_frame(frame)


def get_state() -> dict:
    """Get latest detection state (no inference)."""
    if _detector is None:
        return {
            "threat_level": 0, "threat_name": "OFFLINE",
            "threat_color": "#6b7a90", "violence_score": 0,
            "weapon_count": 0, "weapon_detections": [],
            "fps": 0, "alert": None, "is_running": False
        }
    return _detector.get_state()


def is_loaded() -> bool:
    return _detector is not None and _detector.ready


# ═══════════════════════════════════════════════════════════════════════
#  Standalone mode (python violence_weapon.py)
# ═══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Violence & Weapon Detection')
    parser.add_argument('--source', default='0', help='Camera index or video path')
    parser.add_argument('--device', default='cuda', help='Device: cuda or cpu')
    args = parser.parse_args()

    print("[VW] Starting standalone mode...")
    if not load_detector(device=args.device):
        print("[VW] Failed to initialize. Exiting.")
        sys.exit(1)

    src = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print(f"[VW] Cannot open source: {args.source}")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("[VW] Running... Press Q to quit")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        annotated, state = _detector.process_frame(frame)
        cv2.imshow("Violence & Weapon Detection", annotated)
        if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[VW] Stopped.")
