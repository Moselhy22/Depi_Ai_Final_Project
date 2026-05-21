"""
detect.py  —  V2
Real-time in-cabin drowsiness detection using YOLO11 + MediaPipe Ensemble.

Alert tiers:
  MICROSLEEP : EAR < 0.24 for > 1.5 s
  DROWSY     : PERCLOS > 70% over 3-second rolling window
  FATIGUED   : >= 3 yawns in 60 s

Usage:
  python detect.py               # default webcam
  python detect.py --source 1    # secondary camera
  python detect.py --no-sound    # visual alerts only
"""

import argparse
import collections
import sqlite3
import sys
import time
import warnings
from pathlib import Path

import cv2
import numpy as np

warnings.filterwarnings("ignore")

PROJECT_ROOT    = Path(__file__).resolve().parent
DEFAULT_WEIGHTS = PROJECT_ROOT / "models" / "best.pt"
LOG_DB_PATH     = PROJECT_ROOT / "logs" / "drowsiness_log.db"
ALERT_SOUND     = PROJECT_ROOT / "sounds" / "alert.wav"

# ── Thresholds ────────────────────────────────────────────────────────────────
EAR_THRESHOLD     = 0.24   # Tuned for no-glasses faces
PERCLOS_WINDOW    = 90     # 3-second rolling window at 30 fps
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


def draw_hud(frame, state):
    h, w = frame.shape[:2]
    panel = frame.copy()
    cv2.rectangle(panel, (0, 0), (290, h), (15, 15, 15), -1)
    cv2.addWeighted(panel, 0.55, frame, 0.45, 0, frame)

    def put(text, y, color=WHITE, scale=0.55, thick=1):
        cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                    scale, color, thick, cv2.LINE_AA)

    put("IN-CABIN DMS  v2", 30, WHITE, 0.65, 2)
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
    put(f"Model  : YOLO11n + MediaPipe", 195, (120, 120, 120), 0.42)
    put(f"FPS    : {state['fps']:.1f}", 215, (150, 150, 150))

    alert = state.get("alert")
    if alert:
        ov = frame.copy()
        cv2.rectangle(ov, (0, h - 80), (w, h), (0, 0, 180), -1)
        cv2.addWeighted(ov, 0.6, frame, 0.4, 0, frame)
        cv2.putText(frame, f"  {alert}", (w // 2 - 220, h - 28),
                    cv2.FONT_HERSHEY_DUPLEX, 1.0, WHITE, 2, cv2.LINE_AA)


def run(args):
    from ultralytics import YOLO
    import torch

    # ── MediaPipe ─────────────────────────────────────────────────
    try:
        import mediapipe as mp
        face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1, refine_landmarks=True,
            min_detection_confidence=0.5, min_tracking_confidence=0.5)
        mp_ok = True
    except Exception:
        face_mesh = None
        mp_ok     = False
        print("MediaPipe unavailable — geometric EAR/MAR disabled.")

    # ── Load YOLO11 model ──────────────────────────────────────────
    weights = Path(args.weights)
    if not weights.exists():
        candidates = sorted(PROJECT_ROOT.glob("runs/detect/*/weights/best.pt"),
                            key=lambda p: p.stat().st_mtime, reverse=True)
        if candidates:
            weights = candidates[0]
            print(f"Using latest weights: {weights}")
        else:
            print("No model found. Train first: train.bat")
            sys.exit(1)

    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"Loading YOLO11: {weights}  |  device: {'GPU' if device == 0 else 'CPU'}")
    model = YOLO(str(weights))

    # ── Camera ────────────────────────────────────────────────────
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

    perclos_buf      = collections.deque(maxlen=PERCLOS_WINDOW)
    yawn_timestamps  = collections.deque()
    microsleep_start = None
    last_alert_sound = 0.0
    prev_alert       = None
    frame_count      = 0
    fps_timer        = time.time()
    display_fps      = 0.0

    state = dict(ear=0.0, mar=0.0, perclos=0.0, yawns=0, fps=0.0, alert=None)
    print("\nRunning — Press [Q] to quit\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        h, w = frame.shape[:2]
        if time.time() - fps_timer >= 1.0:
            display_fps = frame_count / (time.time() - fps_timer + 1e-6)
            fps_timer   = time.time()
            frame_count = 0
        state["fps"] = display_fps

        # ── YOLO11 inference ───────────────────────────────────────
        results = model.predict(source=frame, conf=args.conf,
                                iou=0.45, device=device, verbose=False)[0]

        yolo_drowsy = 0
        yolo_awake  = 0
        for box in results.boxes:
            cls  = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            if cls == CLASS_DROWSY:
                yolo_drowsy += 1
                cv2.rectangle(frame, (x1, y1), (x2, y2), RED, 2)
                cv2.putText(frame, f"Drowsy {conf:.2f}", (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, RED, 1)
            elif cls == CLASS_AWAKE:
                yolo_awake += 1
                cv2.rectangle(frame, (x1, y1), (x2, y2), GREEN, 2)
                cv2.putText(frame, f"Awake {conf:.2f}", (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, GREEN, 1)

        yolo_eye_detected = (yolo_drowsy + yolo_awake) > 0
        eye_closed_yolo   = yolo_drowsy > 0 and yolo_awake == 0

        # ── MediaPipe (always run for EAR/MAR accuracy) ───────────
        ear, mar          = 0.0, 0.0
        eye_closed_mp     = False
        if mp_ok:
            rgb       = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_result = face_mesh.process(rgb)
            if mp_result.multi_face_landmarks:
                lm           = mp_result.multi_face_landmarks[0]
                ear, mar     = get_ear_mar(lm, w, h)
                eye_closed_mp = ear < EAR_THRESHOLD
                if eye_closed_mp:
                    if microsleep_start is None:
                        microsleep_start = time.time()
                else:
                    microsleep_start = None

        # ── Ensemble PERCLOS (YOLO OR MediaPipe) ──────────────────
        is_closed = eye_closed_yolo or eye_closed_mp
        perclos_buf.append(1 if is_closed else 0)

        state["ear"] = ear
        state["mar"] = mar

        # ── Yawn detection ────────────────────────────────────────
        now = time.time()
        if mar > MAR_THRESHOLD:
            yawn_timestamps.append(now)
        while yawn_timestamps and (now - yawn_timestamps[0]) > YAWN_WINDOW_SECS:
            yawn_timestamps.popleft()
        state["yawns"] = len(yawn_timestamps)

        # ── PERCLOS score ─────────────────────────────────────────
        perclos          = np.mean(perclos_buf) if perclos_buf else 0.0
        state["perclos"] = perclos

        # ── Alert logic ───────────────────────────────────────────
        alert = alert_type = None
        if microsleep_start and (now - microsleep_start) >= (45 / fps_cam):
            alert      = f"!! MICROSLEEP ({now - microsleep_start:.1f}s) !!"
            alert_type = "MICROSLEEP"
        elif perclos >= PERCLOS_THRESHOLD and len(perclos_buf) == PERCLOS_WINDOW:
            alert      = "DROWSY — Eyes closing!"
            alert_type = "DROWSY"
        elif len(yawn_timestamps) >= YAWN_COUNT_ALERT:
            alert      = f"FATIGUED — {len(yawn_timestamps)} yawns in 60s"
            alert_type = "FATIGUED"

        state["alert"] = alert
        if alert:
            last_alert_sound = play_alert(sound, last_alert_sound)
            if alert_type != prev_alert:
                log_event(conn, alert_type,
                          now - microsleep_start if microsleep_start else 0, alert)
        prev_alert = alert_type

        draw_hud(frame, state)
        cv2.imshow("In-Cabin DMS  v2  |  Q = Quit", frame)
        if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
            break

    cap.release()
    cv2.destroyAllWindows()
    conn.close()
    if mp_ok and face_mesh:
        face_mesh.close()
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
