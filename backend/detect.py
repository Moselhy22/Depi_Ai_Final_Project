"""
detect.py  —  V3  (module mode + standalone)
Real-time in-cabin drowsiness detection using YOLO11 + MediaPipe Ensemble.

Alert tiers:
  MICROSLEEP : EAR < 0.24 for > 1.5 s
  DROWSY     : PERCLOS > 70% over 3-second rolling window
  FATIGUED   : >= 3 yawns in 60 s

Usage as module (imported by app.py):
  from detect import init_detector, detect_drowsiness, get_drowsiness_state

Usage standalone:
  python detect.py               # default webcam
  python detect.py --source 1    # secondary camera
  python detect.py --no-sound    # visual alerts only
"""

import argparse
import collections
import sqlite3
import sys
import time
import threading
import warnings
from pathlib import Path

import cv2
import numpy as np

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────────
# detect.py lives in  D:\DEPI GP\backend\
# so parent = backend, parent.parent = D:\DEPI GP
_PROJECT_ROOT  = Path(__file__).resolve().parent.parent

DEFAULT_WEIGHTS = Path(r"D:\DEPI GP\models\yolov11n\best.pt")
LOG_DB_PATH     = _PROJECT_ROOT / "logs" / "drowsiness_log.db"
ALERT_SOUND     = _PROJECT_ROOT / "sounds" / "alert.wav"

# ── Thresholds ────────────────────────────────────────────────────────────────
EAR_THRESHOLD     = 0.24
PERCLOS_WINDOW    = 90      # 3-second rolling window at 30 fps
PERCLOS_THRESHOLD = 0.70
YAWN_WINDOW_SECS  = 60
YAWN_COUNT_ALERT  = 3
MAR_THRESHOLD     = 0.55

# ── Class IDs ─────────────────────────────────────────────────────────────────
CLASS_AWAKE  = 0
CLASS_DROWSY = 1

# ── Colours (BGR) ─────────────────────────────────────────────────────────────
GREEN  = (0, 220, 100)
ORANGE = (0, 165, 255)
RED    = (0, 50, 255)
WHITE  = (255, 255, 255)

# ── MediaPipe landmark indices ────────────────────────────────────────────────
_LEFT_EYE  = [362, 385, 387, 263, 373, 380]
_RIGHT_EYE = [33,  160, 158, 133, 153, 144]
_MOUTH     = [61,  40,  37,  0,  267, 270, 291, 321]


# ════════════════════════════════════════════
#  Geometry helpers
# ════════════════════════════════════════════

def _dist(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))


def compute_ear(pts):
    A = _dist(pts[1], pts[5])
    B = _dist(pts[2], pts[4])
    C = _dist(pts[0], pts[3])
    return (A + B) / (2.0 * C + 1e-6)


def compute_mar(pts):
    A = _dist(pts[1], pts[7])
    B = _dist(pts[2], pts[6])
    C = _dist(pts[3], pts[5])
    D = _dist(pts[0], pts[4])
    return (A + B + C) / (2.0 * D + 1e-6)


def get_ear_mar(face_lm, w, h):
    lm = face_lm.landmark
    pt = lambda i: (lm[i].x * w, lm[i].y * h)
    ear = (compute_ear([pt(i) for i in _LEFT_EYE]) +
           compute_ear([pt(i) for i in _RIGHT_EYE])) / 2.0
    mar = compute_mar([pt(i) for i in _MOUTH])
    return ear, mar


# ════════════════════════════════════════════
#  DB helpers (standalone only)
# ════════════════════════════════════════════

def init_db(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.execute("""CREATE TABLE IF NOT EXISTS events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT, alert_type TEXT, duration_s REAL, notes TEXT)""")
    conn.commit()
    return conn


def log_event(conn, alert_type, duration_s=0.0, notes=""):
    conn.execute("INSERT INTO events VALUES (NULL,?,?,?,?)",
                 (time.strftime("%Y-%m-%d %H:%M:%S"), alert_type,
                  round(duration_s, 2), notes))
    conn.commit()


# ════════════════════════════════════════════
#  Audio helpers (standalone only)
# ════════════════════════════════════════════

def init_audio(sound_path, use_sound):
    if not use_sound:
        return None
    try:
        import pygame
        pygame.mixer.init()
        if sound_path.exists():
            return pygame.mixer.Sound(str(sound_path))
        sample_rate = 44100
        t    = np.linspace(0, 0.4, int(sample_rate * 0.4), endpoint=False)
        wave = (np.sin(2 * np.pi * 880 * t) * 32767 * 0.7).astype(np.int16)
        return pygame.sndarray.make_sound(np.column_stack([wave, wave]))
    except Exception as e:
        print(f"Audio init failed: {e}. Visual alerts only.")
        return None


def play_alert(sound, last_played, cooldown=2.0):
    now = time.time()
    if sound and (now - last_played) >= cooldown:
        try: sound.play()
        except: pass
        return now
    return last_played


# ════════════════════════════════════════════
#  HUD drawing
# ════════════════════════════════════════════

def draw_hud(frame, state):
    h, w = frame.shape[:2]
    panel = frame.copy()
    cv2.rectangle(panel, (0, 0), (290, h), (15, 15, 15), -1)
    cv2.addWeighted(panel, 0.55, frame, 0.45, 0, frame)

    def put(text, y, color=WHITE, scale=0.55, thick=1):
        cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                    scale, color, thick, cv2.LINE_AA)

    put("DROWSINESS MONITOR", 30, WHITE, 0.65, 2)
    cv2.line(frame, (10, 38), (280, 38), (80, 80, 80), 1)

    ear_col = RED if state["ear"] < EAR_THRESHOLD else GREEN
    put(f"EAR     : {state['ear']:.3f}", 65,  ear_col)
    put(f"MAR     : {state['mar']:.3f}", 90,  WHITE)

    perclos = state["perclos"]
    col     = RED if perclos >= PERCLOS_THRESHOLD else (ORANGE if perclos > 0.4 else GREEN)
    put(f"PERCLOS : {perclos*100:.0f}%", 120, col)
    cv2.rectangle(frame, (10, 128), (10 + int(perclos * 250), 142), col, -1)
    cv2.rectangle(frame, (10, 128), (260, 142), WHITE, 1)

    put(f"Yawns (60s) : {state['yawns']}", 165,
        RED if state["yawns"] >= YAWN_COUNT_ALERT else WHITE)
    put(f"Model  : YOLO11n + MP", 195, (120, 120, 120), 0.42)
    put(f"FPS    : {state['fps']:.1f}", 215, (150, 150, 150))

    alert = state.get("alert")
    if alert:
        ov = frame.copy()
        cv2.rectangle(ov, (0, h - 80), (w, h), (0, 0, 180), -1)
        cv2.addWeighted(ov, 0.6, frame, 0.4, 0, frame)
        cv2.putText(frame, f"  {alert}", (w // 2 - 220, h - 28),
                    cv2.FONT_HERSHEY_DUPLEX, 1.0, WHITE, 2, cv2.LINE_AA)


# ════════════════════════════════════════════
#  MODULE-LEVEL API  (used by app.py)
# ════════════════════════════════════════════

_detector       = None   # DrowsinessDetector instance (or None)
_detector_lock = threading.Lock()


class DrowsinessDetector:
    """Stateful per-frame drowsiness detector."""

    def __init__(self):
        from ultralytics import YOLO
        import torch

        # ── YOLO11 model ────────────────────────────────────
        weights = Path(DEFAULT_WEIGHTS)
        if not weights.exists():
            candidates = sorted(_PROJECT_ROOT.glob("runs/detect/*/weights/best.pt"),
                                key=lambda p: p.stat().st_mtime, reverse=True)
            if candidates:
                weights = candidates[0]
                print(f"[DROWSY] Using latest weights: {weights}")
            else:
                raise FileNotFoundError(
                    f"No drowsiness model found at {weights}. "
                    "Train first or place best.pt in models/yolov11n/")

        self.device = 0 if torch.cuda.is_available() else "cpu"
        print(f"[DROWSY] Loading YOLO11: {weights}  |  device: {'GPU' if self.device == 0 else 'CPU'}")
        self.model = YOLO(str(weights))

        # ── MediaPipe ───────────────────────────────────────
        self.mp_ok = False
        self.face_mesh = None
        try:
            import mediapipe as mp
            self.face_mesh = mp.solutions.face_mesh.FaceMesh(
                max_num_faces=1, refine_landmarks=True,
                min_detection_confidence=0.5, min_tracking_confidence=0.5)
            self.mp_ok = True
            print("[DROWSY] MediaPipe loaded.")
        except Exception:
            print("[DROWSY] MediaPipe unavailable — geometric EAR/MAR disabled.")

        # ── Rolling state ───────────────────────────────────
        self.perclos_buf      = collections.deque(maxlen=PERCLOS_WINDOW)
        self.yawn_timestamps  = collections.deque()
        self.microsleep_start = None
        self.prev_alert       = None

        # ── FPS tracking ────────────────────────────────────
        self.frame_count = 0
        self.fps_timer   = time.time()
        self.display_fps = 0.0

        # ── Thread-safe latest state ────────────────────────
        self._state_lock = threading.Lock()
        self._latest_state = {
            "ear": 0.0, "mar": 0.0, "perclos": 0.0,
            "yawns": 0, "fps": 0.0, "alert": None,
            "alert_type": None,
        }

        print("[DROWSY] Detector initialized.")

    def detect(self, frame):
        """
        Process a single frame.

        Returns:
            annotated_frame  (np.ndarray) – frame with HUD drawn
            state            (dict)        – current metrics snapshot
        """
        h, w = frame.shape[:2]

        # ── FPS ─────────────────────────────────────────────
        self.frame_count += 1
        now = time.time()
        if now - self.fps_timer >= 1.0:
            self.display_fps = self.frame_count / (now - self.fps_timer + 1e-6)
            self.fps_timer   = now
            self.frame_count = 0

        # ── YOLO11 inference ────────────────────────────────
        results = self.model.predict(
            source=frame, conf=0.35, iou=0.45,
            device=self.device, verbose=False)[0]

        yolo_drowsy = 0
        yolo_awake  = 0
        annotated = frame.copy()

        for box in results.boxes:
            cls  = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            if cls == CLASS_DROWSY:
                yolo_drowsy += 1
                cv2.rectangle(annotated, (x1, y1), (x2, y2), RED, 2)
                cv2.putText(annotated, f"Drowsy {conf:.2f}", (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, RED, 1)
            elif cls == CLASS_AWAKE:
                yolo_awake += 1
                cv2.rectangle(annotated, (x1, y1), (x2, y2), GREEN, 2)
                cv2.putText(annotated, f"Awake {conf:.2f}", (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, GREEN, 1)

        eye_closed_yolo = yolo_drowsy > 0 and yolo_awake == 0

        # ── MediaPipe ───────────────────────────────────────
        ear, mar          = 0.0, 0.0
        eye_closed_mp     = False
        if self.mp_ok and self.face_mesh is not None:
            rgb       = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            mp_result = self.face_mesh.process(rgb)
            if mp_result.multi_face_landmarks:
                lm           = mp_result.multi_face_landmarks[0]
                ear, mar     = get_ear_mar(lm, w, h)
                eye_closed_mp = ear < EAR_THRESHOLD
                if eye_closed_mp:
                    if self.microsleep_start is None:
                        self.microsleep_start = time.time()
                else:
                    self.microsleep_start = None

        # ── Ensemble PERCLOS ────────────────────────────────
        is_closed = eye_closed_yolo or eye_closed_mp
        self.perclos_buf.append(1 if is_closed else 0)

        # ── Yawn detection ──────────────────────────────────
        if mar > MAR_THRESHOLD:
            self.yawn_timestamps.append(now)
        while self.yawn_timestamps and (now - self.yawn_timestamps[0]) > YAWN_WINDOW_SECS:
            self.yawn_timestamps.popleft()

        # ── PERCLOS score ───────────────────────────────────
        perclos = float(np.mean(self.perclos_buf)) if self.perclos_buf else 0.0

        # ── Alert logic ─────────────────────────────────────
        alert      = None
        alert_type = None
        if self.microsleep_start and (now - self.microsleep_start) >= 1.5:
            alert      = f"!! MICROSLEEP ({now - self.microsleep_start:.1f}s) !!"
            alert_type = "MICROSLEEP"
        elif perclos >= PERCLOS_THRESHOLD and len(self.perclos_buf) == PERCLOS_WINDOW:
            alert      = "DROWSY - Eyes closing!"
            alert_type = "DROWSY"
        elif len(self.yawn_timestamps) >= YAWN_COUNT_ALERT:
            alert      = f"FATIGUED - {len(self.yawn_timestamps)} yawns in 60s"
            alert_type = "FATIGUED"
        self.prev_alert = alert_type

        # ── Build state ─────────────────────────────────────
        state = {
            "ear":         ear,
            "mar":         mar,
            "perclos":     perclos,
            "yawns":       len(self.yawn_timestamps),
            "fps":         self.display_fps,
            "alert":       alert,
            "alert_type":  alert_type,
        }

        # ── Draw HUD ────────────────────────────────────────
        draw_hud(annotated, state)

        # ── Cache state for polling ─────────────────────────
        with self._state_lock:
            self._latest_state = dict(state)

        return annotated, state

    def get_state(self):
        with self._state_lock:
            return dict(self._latest_state)

    def close(self):
        if self.face_mesh:
            try: self.face_mesh.close()
            except: pass


# ── Module-level convenience functions ────────────────────────────────────────

def init_detector():
    """Initialize the drowsiness detector. Call once before detect_drowsiness()."""
    global _detector
    with _detector_lock:
        if _detector is not None:
            return True
        try:
            _detector = DrowsinessDetector()
            return True
        except Exception as e:
            print(f"[ERROR] Drowsiness init failed: {e}")
            _detector = None
            return False


def detect_drowsiness(frame):
    """
    Process a single frame through the drowsiness pipeline.

    Args:
        frame (np.ndarray): BGR image from camera

    Returns:
        annotated (np.ndarray): frame with HUD overlay
        state (dict): current detection metrics
    """
    if _detector is None:
        if not init_detector():
            return frame, {"ear":0,"mar":0,"perclos":0,"yawns":0,"fps":0,"alert":None,"alert_type":None}
    return _detector.detect(frame)


def get_drowsiness_state():
    """Return the latest cached state dict (thread-safe, no inference)."""
    if _detector is None:
        return {"ear":0,"mar":0,"perclos":0,"yawns":0,"fps":0,"alert":None,"alert_type":None}
    return _detector.get_state()


# ════════════════════════════════════════════
#  Standalone mode  (python detect.py)
# ════════════════════════════════════════════

def run(args):
    """Standalone mode: opens camera, runs detection loop, shows window."""
    det = DrowsinessDetector()

    src = int(args.source) if str(args.source).isdigit() else args.source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print(f"Cannot open camera: {args.source}")
        sys.exit(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    fps_cam = cap.get(cv2.CAP_PROP_FPS) or 30

    conn  = init_db(LOG_DB_PATH)
    sound = init_audio(ALERT_SOUND, not args.no_sound)
    last_alert_sound = 0.0

    print("\nRunning - Press [Q] to quit\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        annotated, state = det.detect(frame)

        # Play sound on alert
        if state.get("alert"):
            last_alert_sound = play_alert(sound, last_alert_sound)

        cv2.imshow("In-Cabin DMS  v2  |  Q = Quit", annotated)
        if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
            break

    cap.release()
    cv2.destroyAllWindows()
    conn.close()
    det.close()
    print("Session ended. Log:", LOG_DB_PATH)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source",   default="0")
    parser.add_argument("--weights",  default=str(DEFAULT_WEIGHTS))
    parser.add_argument("--conf",     type=float, default=0.35)
    parser.add_argument("--no-sound", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
