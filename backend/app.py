"""
app.py  —  Final Version (Dual Stream + Violence + Shared Camera)
- Distraction detection (YOLO distraction model)
- Drowsiness detection (YOLO11n + MediaPipe)
- Violence & Weapon detection (VD-MIL + YOLOv8n)
- Single Camera Producer (cam 0) — distraction + drowsiness share frames
- Separate Camera 2 Producer (cam 1) — violence & weapon
- No race conditions — lock-protected singleton pattern
"""

from flask import Flask, request, jsonify, Response, send_from_directory
from flask_cors import CORS
import cv2
import numpy as np
import base64
import json
import os
import sqlite3
import threading
import time
import collections
from datetime import datetime
from pathlib import Path

from recognition import register_user, login_user, DB_PATH


# ════════════════════════════════════════════════════════════════
#  LAZY LOADING — Distraction Module
# ════════════════════════════════════════════════════════════════
_distraction_module    = None
_distraction_load_error = None
_distraction_lock      = threading.Lock()


def _load_distraction():
    global _distraction_module, _distraction_load_error
    with _distraction_lock:
        if _distraction_module is not None:
            return True
        if _distraction_load_error is not None:
            return False
        try:
            import distraction
            _distraction_module = distraction
            print("[INFO] Distraction module loaded.")
            return True
        except Exception as e:
            _distraction_load_error = str(e)
            print(f"[ERROR] Distraction load failed: {e}")
            return False


def _get_detect_fn():
    return _distraction_module.detect_distraction if _load_distraction() else None


def _get_state_fn():
    return _distraction_module.get_latest_state if _load_distraction() else None


# ════════════════════════════════════════════════════════════════
#  LAZY LOADING — Drowsiness Detector
# ════════════════════════════════════════════════════════════════
_drowsiness_detector   = None
_drowsiness_load_error = None
_drowsiness_lock       = threading.Lock()


def _load_drowsiness():
    global _drowsiness_detector, _drowsiness_load_error
    with _drowsiness_lock:
        if _drowsiness_detector is not None:
            return True
        if _drowsiness_load_error is not None:
            return False
        try:
            _drowsiness_detector = DrowsinessDetector()
            if not _drowsiness_detector.ready:
                raise RuntimeError("Detector init returned not ready")
            print("[INFO] Drowsiness module loaded.")
            return True
        except Exception as e:
            _drowsiness_load_error = str(e)
            _drowsiness_detector = None
            print(f"[ERROR] Drowsiness init failed: {e}")
            return False


# ════════════════════════════════════════════════════════════════
#  DrowsinessDetector Class (no camera — frame-by-frame only)
# ════════════════════════════════════════════════════════════════
class DrowsinessDetector:
    EAR_THRESHOLD     = 0.25
    PERCLOS_WINDOW    = 120
    PERCLOS_THRESHOLD = 0.75
    YAWN_WINDOW_SECS  = 60
    YAWN_COUNT_ALERT  = 3
    MAR_THRESHOLD     = 0.55

    CLASS_AWAKE  = 0
    CLASS_DROWSY = 1

    GREEN  = (0, 220, 100)
    ORANGE = (0, 165, 255)
    RED    = (0, 50, 255)
    WHITE  = (255, 255, 255)

    _LEFT_EYE  = [362, 385, 387, 263, 373, 380]
    _RIGHT_EYE = [33,  160, 158, 133, 153, 144]
    _MOUTH     = [61,  40,  37,  0,  267, 270, 291, 321]

    def __init__(self, weights_path=None):
        self.ready = False
        self.model = None
        self.device = "cpu"
        self.face_mesh = None
        self.mp_ok = False

        self.perclos_buf     = collections.deque(maxlen=self.PERCLOS_WINDOW)
        self.yawn_timestamps = collections.deque()
        self.microsleep_start = None

        self.state = dict(
            drowsy=False, alert=None,
            ear=0.0, mar=0.0, perclos=0.0, yawns=0,
        )

        try:
            from ultralytics import YOLO
            import torch

            if weights_path is None:
                weights_path = r"D:\DEPI GP\models\yolov11n\best.pt"

            weights = Path(weights_path)
            if not weights.exists():
                raise FileNotFoundError(f"Model not found: {weights}")

            self.device = 0 if torch.cuda.is_available() else "cpu"
            print(f"[DROWSY] Loading YOLO11: {weights}  |  device: "
                  f"{'GPU' if self.device == 0 else 'CPU'}")
            self.model = YOLO(str(weights))

            try:
                import mediapipe as mp
                self.face_mesh = mp.solutions.face_mesh.FaceMesh(
                    max_num_faces=1, refine_landmarks=True,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5,
                )
                self.mp_ok = True
            except Exception:
                print("[DROWSY] MediaPipe unavailable — geometric EAR/MAR disabled.")

            self.ready = True
            print("[DROWSY] Detector initialized.")

        except Exception as e:
            print(f"[DROWSY] Detector init failed: {e}")
            raise

    @staticmethod
    def _dist(p1, p2):
        return np.linalg.norm(np.array(p1) - np.array(p2))

    def _compute_ear(self, pts):
        A = self._dist(pts[1], pts[5])
        B = self._dist(pts[2], pts[4])
        C = self._dist(pts[0], pts[3])
        return (A + B) / (2.0 * C + 1e-6)

    def _compute_mar(self, pts):
        A = self._dist(pts[1], pts[7])
        B = self._dist(pts[2], pts[6])
        C = self._dist(pts[3], pts[5])
        D = self._dist(pts[0], pts[4])
        return (A + B + C) / (2.0 * D + 1e-6)

    def _get_ear_mar(self, face_lm, w, h):
        lm = face_lm.landmark
        pt = lambda i: (lm[i].x * w, lm[i].y * h)
        ear = (self._compute_ear([pt(i) for i in self._LEFT_EYE]) +
               self._compute_ear([pt(i) for i in self._RIGHT_EYE])) / 2.0
        mar = self._compute_mar([pt(i) for i in self._MOUTH])
        return ear, mar

    def process_frame(self, frame):
        h, w = frame.shape[:2]
        now = time.time()

        results = self.model.predict(
            source=frame, conf=0.35,
            iou=0.45, device=self.device, verbose=False,
        )[0]

        yolo_drowsy = 0
        yolo_awake  = 0
        for box in results.boxes:
            cls  = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            if cls == self.CLASS_DROWSY:
                yolo_drowsy += 1
                cv2.rectangle(frame, (x1, y1), (x2, y2), self.RED, 2)
                cv2.putText(frame, f"Drowsy {conf:.2f}", (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.RED, 1)
            elif cls == self.CLASS_AWAKE:
                yolo_awake += 1
                cv2.rectangle(frame, (x1, y1), (x2, y2), self.GREEN, 2)
                cv2.putText(frame, f"Awake {conf:.2f}", (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.GREEN, 1)

        eye_closed_yolo = yolo_drowsy > 0 and yolo_awake == 0

        ear, mar      = 0.0, 0.0
        eye_closed_mp = False
        if self.mp_ok:
            try:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_result = self.face_mesh.process(rgb)
                if mp_result.multi_face_landmarks:
                    lm = mp_result.multi_face_landmarks[0]
                    ear, mar = self._get_ear_mar(lm, w, h)
                    eye_closed_mp = ear < self.EAR_THRESHOLD
                    if eye_closed_mp:
                        if self.microsleep_start is None:
                            self.microsleep_start = now
                    else:
                        self.microsleep_start = None
            except Exception:
                pass

        is_closed = eye_closed_yolo or eye_closed_mp
        self.perclos_buf.append(1 if is_closed else 0)

        if mar > self.MAR_THRESHOLD:
            self.yawn_timestamps.append(now)
        while self.yawn_timestamps and (now - self.yawn_timestamps[0]) > self.YAWN_WINDOW_SECS:
            self.yawn_timestamps.popleft()

        perclos = np.mean(self.perclos_buf) if self.perclos_buf else 0.0

        alert = None
        if self.microsleep_start and (now - self.microsleep_start) >= 1.5:
            alert = f"!! MICROSLEEP ({now - self.microsleep_start:.1f}s) !!"
        elif perclos >= self.PERCLOS_THRESHOLD and len(self.perclos_buf) == self.PERCLOS_WINDOW:
            alert = "DROWSY - Eyes closing!"
        elif len(self.yawn_timestamps) >= self.YAWN_COUNT_ALERT:
            alert = f"FATIGUED - {len(self.yawn_timestamps)} yawns in 60s"

        self._draw_hud(frame, ear, mar, perclos,
                       len(self.yawn_timestamps), alert, h, w)

        alert_type = None
        if self.microsleep_start and (now - self.microsleep_start) >= 1.5:
            alert_type = "MICROSLEEP"
        elif perclos >= self.PERCLOS_THRESHOLD and len(self.perclos_buf) == self.PERCLOS_WINDOW:
            alert_type = "DROWSY"
        elif len(self.yawn_timestamps) >= self.YAWN_COUNT_ALERT:
            alert_type = "FATIGUED"

        self.state = dict(
            drowsy     = alert is not None,
            alert      = alert,
            alert_type = alert_type,
            ear        = round(ear, 3),
            mar        = round(mar, 3),
            perclos    = round(perclos, 3),
            yawns      = len(self.yawn_timestamps),
        )
        return frame, self.state

    def _draw_hud(self, frame, ear, mar, perclos, yawns, alert, h, w):
        cv2.putText(frame, "DROWSINESS MONITOR", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, self.WHITE, 2, cv2.LINE_AA)

        ear_col = self.RED if ear < self.EAR_THRESHOLD else self.GREEN
        cv2.putText(frame, f"EAR: {ear:.3f}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, ear_col, 1)

        cv2.putText(frame, f"MAR: {mar:.3f}", (10, 85),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.WHITE, 1)

        col = (self.RED if perclos >= self.PERCLOS_THRESHOLD
               else (self.ORANGE if perclos > 0.4 else self.GREEN))
        cv2.putText(frame, f"PERCLOS: {perclos*100:.0f}%", (10, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)

        cv2.putText(frame, f"Yawns(60s): {yawns}", (10, 135),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    self.RED if yawns >= self.YAWN_COUNT_ALERT else self.WHITE, 1)

        if alert:
            ov = frame.copy()
            cv2.rectangle(ov, (0, h - 60), (w, h), (0, 0, 180), -1)
            cv2.addWeighted(ov, 0.6, frame, 0.4, 0, frame)
            cv2.putText(frame, f"  {alert}", (w // 2 - 220, h - 20),
                        cv2.FONT_HERSHEY_DUPLEX, 1.0, self.WHITE, 2, cv2.LINE_AA)

    def get_state(self):
        return self.state.copy()


# ════════════════════════════════════════════════════════════════
#  Flask App
# ════════════════════════════════════════════════════════════════
app = Flask(__name__)
CORS(app, resources={r"/*": {
    "origins": "*",
    "methods": ["GET", "POST", "OPTIONS"],
    "allow_headers": ["Content-Type"],
}})


@app.before_request
def handle_options():
    if request.method == "OPTIONS":
        return app.make_default_options_response()


DATA_PATH       = r"D:\DEPI GP\data"
FRONTEND_DIR    = r"D:\DEPI GP"
LOG_PATH         = os.path.join(DATA_PATH, "activity_log.json")
DISTRACTION_LOG  = os.path.join(DATA_PATH, "distraction_log.json")
DROWSINESS_LOG   = os.path.join(DATA_PATH, "drowsiness_log.json")
VIOLENCE_LOG     = os.path.join(DATA_PATH, "violence_log.json")
DB_PATH_LOGS     = os.path.join(DATA_PATH, "system_logs.db")
os.makedirs(DATA_PATH, exist_ok=True)


# ════════════════════════════════════════════════════════════════
#  SQLite Database for Logs
# ════════════════════════════════════════════════════════════════
_db_lock = threading.Lock()


def _get_log_db():
    """Return a connection to the log database."""
    conn = sqlite3.connect(DB_PATH_LOGS, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def _init_log_db():
    """Create log tables if they don't exist."""
    with _db_lock:
        conn = _get_log_db()
        try:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS activity_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user TEXT,
                    action TEXT,
                    details TEXT,
                    timestamp TEXT
                );
                CREATE TABLE IF NOT EXISTS distraction_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user TEXT,
                    timestamp TEXT,
                    detections TEXT
                );
                CREATE TABLE IF NOT EXISTS drowsiness_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user TEXT,
                    timestamp TEXT,
                    alert_type TEXT,
                    details TEXT
                );
                CREATE TABLE IF NOT EXISTS violence_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user TEXT,
                    timestamp TEXT,
                    threat_level INTEGER,
                    threat_name TEXT,
                    violence_score REAL,
                    weapon_count INTEGER,
                    alert TEXT
                );
            """)
            conn.commit()
            print("[DB] Log database initialized.")
        finally:
            conn.close()


_init_log_db()


@app.route("/")
def index():
    return send_from_directory(FRONTEND_DIR, "index.html")


@app.route("/<path:filename>")
def static_files(filename):
    return send_from_directory(FRONTEND_DIR, filename)


_json_lock = threading.Lock()


def _load_json(path):
    with _json_lock:
        if not os.path.exists(path):
            return []
        try:
            with open(path) as f:
                c = f.read()
                return json.loads(c) if c else []
        except Exception:
            return []


def _save_json(path, data):
    with _json_lock:
        with open(path, "w") as f:
            json.dump(data, f, indent=4)


def add_activity_log(user, action, details=""):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # Save to JSON
    entries = _load_json(LOG_PATH)
    entries.append({
        "user":      user,
        "action":    action,
        "details":   details,
        "timestamp": ts,
    })
    _save_json(LOG_PATH, entries)
    # Save to SQLite
    try:
        with _db_lock:
            conn = _get_log_db()
            try:
                conn.execute(
                    "INSERT INTO activity_logs (user, action, details, timestamp) VALUES (?,?,?,?)",
                    (user, action, details, ts),
                )
                conn.commit()
            finally:
                conn.close()
    except Exception as e:
        print(f"[DB ERROR] activity_log: {e}")


def add_distraction_log(user, detections):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # Save to JSON
    entries = _load_json(DISTRACTION_LOG)
    entries.append({
        "user":       user,
        "timestamp":  ts,
        "detections": detections,
    })
    _save_json(DISTRACTION_LOG, entries)
    # Save to SQLite
    try:
        with _db_lock:
            conn = _get_log_db()
            try:
                conn.execute(
                    "INSERT INTO distraction_logs (user, timestamp, detections) VALUES (?,?,?)",
                    (user, ts, json.dumps(detections)),
                )
                conn.commit()
            finally:
                conn.close()
    except Exception as e:
        print(f"[DB ERROR] distraction_log: {e}")


def add_drowsiness_log(user, alert_type, details):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # Save to JSON
    entries = _load_json(DROWSINESS_LOG)
    entries.append({
        "user":      user,
        "timestamp": ts,
        "alert_type": alert_type,
        "details":   details,
    })
    _save_json(DROWSINESS_LOG, entries)
    # Save to SQLite
    try:
        with _db_lock:
            conn = _get_log_db()
            try:
                conn.execute(
                    "INSERT INTO drowsiness_logs (user, timestamp, alert_type, details) VALUES (?,?,?,?)",
                    (user, ts, alert_type, details),
                )
                conn.commit()
            finally:
                conn.close()
    except Exception as e:
        print(f"[DB ERROR] drowsiness_log: {e}")


def decode(img_str: str):
    try:
        arr = np.frombuffer(base64.b64decode(img_str), np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except Exception:
        return None


def encode_img(img):
    if img is None:
        return None
    _, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buf).decode('utf-8')


# ════════════════════════════════════════════════════════════════
#  SINGLE CAMERA PRODUCER (Singleton Thread) — Camera 1
# ════════════════════════════════════════════════════════════════
_camera_lock      = threading.Lock()
_camera_producer  = None
_stream_refcount  = 0
_shared_frame     = None
_frame_seq        = 0
_producer_ready   = threading.Event()


class _CameraProducer(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self._running = False
        self.cap = None

    def run(self):
        global _shared_frame, _frame_seq
        print("[CAM1] Opening camera 1...")
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("[CAM1] ERROR: Camera 1 not available!")
            return
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        time.sleep(0.5)
        self._running = True
        print("[CAM1] Camera 1 opened OK.")

        while self._running:
            ret, frame = self.cap.read()
            if ret:
                with _camera_lock:
                    _shared_frame = frame.copy()
                    _frame_seq   += 1
                    _producer_ready.set()
            else:
                time.sleep(0.01)

        self.cap.release()
        print("[CAM1] Camera 1 closed.")

    def stop(self):
        self._running = False


def _start_producer():
    global _camera_producer, _stream_refcount
    with _camera_lock:
        if _stream_refcount == 0:
            if _camera_producer is None or not _camera_producer.is_alive():
                _camera_producer = _CameraProducer()
                _camera_producer.start()
                print("[CAM1] Producer started.")
        _stream_refcount += 1
    _producer_ready.wait(timeout=5)


def _stop_producer():
    global _camera_producer, _stream_refcount
    with _camera_lock:
        _stream_refcount = max(0, _stream_refcount - 1)
        if _stream_refcount == 0:
            if _camera_producer and _camera_producer.is_alive():
                _camera_producer.stop()
                _camera_producer = None
            _producer_ready.clear()
            print("[CAM1] Producer stopped.")


def _get_shared_frame():
    with _camera_lock:
        if _shared_frame is not None:
            return _shared_frame.copy(), _frame_seq
    return None, 0


# ════════════════════════════════════════════════════════════════
#  Routes — Users
# ════════════════════════════════════════════════════════════════
@app.route("/get_users")
def get_users():
    try:
        if not os.path.exists(DB_PATH):
            return jsonify([])
        with open(DB_PATH) as f:
            c = f.read()
            data = json.loads(c) if c else []
            return jsonify(data if isinstance(data, list) else [])
    except Exception:
        return jsonify([])


@app.route("/register", methods=["POST"])
def register():
    data = request.json
    if not data or not data.get("image") or not data.get("name"):
        return jsonify({"success": False, "message": "Invalid input"})
    img = decode(data["image"])
    if img is None:
        return jsonify({"success": False, "message": "Image decode failed"})
    ok, msg, processed = register_user(data["name"], img)
    if ok:
        add_activity_log(data["name"], "register", msg)
    return jsonify({"success": ok, "message": msg,
                    "processed_image": encode_img(processed)})


@app.route("/login", methods=["POST"])
def login():
    data = request.json
    if not data or not data.get("image") or not data.get("name"):
        return jsonify({"success": False, "message": "Invalid input"})
    img = decode(data["image"])
    if img is None:
        return jsonify({"success": False, "message": "Image decode failed"})
    ok, msg, processed = login_user(data["name"], img)
    add_activity_log(data["name"], "login", msg)
    return jsonify({"success": ok, "message": msg,
                    "processed_image": encode_img(processed)})


@app.route("/set_driver", methods=["POST"])
def set_driver():
    global _current_driver, _last_logged_time
    data = request.json
    name = (data or {}).get("name", "driver").strip()
    if name:
        _current_driver   = name
        _last_logged_time = 0.0
        print(f"[INFO] Active driver set to: {name}")
    return jsonify({"success": True})


# ════════════════════════════════════════════════════════════════
#  Routes — Distraction
# ════════════════════════════════════════════════════════════════
@app.route("/distraction_status")
def distraction_status():
    loaded = _load_distraction()
    return jsonify({
        "model_loaded": loaded,
        "message": "Model ready" if loaded else f"Not loaded: {_distraction_load_error}",
    })


@app.route("/detect_distraction", methods=["POST"])
def detect_distraction_route():
    detect_fn = _get_detect_fn()
    if detect_fn is None:
        return jsonify({"success": False,
                        "message": f"Model not loaded: {_distraction_load_error}"})
    data = request.json
    if not data or not data.get("image"):
        return jsonify({"success": False, "message": "No image"})
    img = decode(data["image"])
    if img is None:
        return jsonify({"success": False, "message": "Decode failed"})
    annotated, detections, distracted = detect_fn(img)
    return jsonify({
        "success":         True,
        "annotated_image": encode_img(annotated),
        "detections":      detections,
        "distracted":      distracted,
        "alert":           "Distraction Detected!" if distracted else "",
    })


@app.route("/get_latest_detection")
def get_latest_detection():
    state_fn = _get_state_fn()
    if state_fn is None:
        return jsonify({"distracted": False, "detections": [],
                        "message": "Model not loaded"})
    return jsonify(state_fn())


@app.route("/log_distraction", methods=["POST"])
def log_distraction():
    data = request.json
    if not data or not data.get("user"):
        return jsonify({"success": False, "message": "Missing user"})
    add_distraction_log(data["user"], data.get("detections", []))
    return jsonify({"success": True})


@app.route("/get_distraction_log")
def get_distraction_log():
    return jsonify(list(reversed(_load_json(DISTRACTION_LOG))))


@app.route("/get_drowsiness_log")
def get_drowsiness_log():
    return jsonify(list(reversed(_load_json(DROWSINESS_LOG))))


@app.route("/get_activity_log")
def get_activity_log():
    return jsonify(list(reversed(_load_json(LOG_PATH))))


@app.route("/stop_camera", methods=["POST"])
def stop_camera():
    global _camera_producer, _stream_refcount, _shared_frame, _frame_seq
    with _camera_lock:
        _stream_refcount = 0
        if _camera_producer and _camera_producer.is_alive():
            _camera_producer.stop()
            _camera_producer = None
        _shared_frame = None
        _frame_seq = 0
        _producer_ready.clear()
    print("[CAM1] Camera 1 force-stopped.")
    return jsonify({"success": True})


# ════════════════════════════════════════════════════════════════
#  Camera Capture (for login / register)
# ════════════════════════════════════════════════════════════════
@app.route("/capture_frame")
def capture_frame():
    frame, _ = _get_shared_frame()
    if frame is not None:
        return jsonify({"success": True, "image": encode_img(frame)})

    cap = None
    try:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            cap = cv2.VideoCapture(1)
        if not cap.isOpened():
            return jsonify({"success": False, "message": "Camera not available"})
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        time.sleep(0.8)
        ret, frame = cap.read()
        if not ret:
            return jsonify({"success": False, "message": "Failed to capture frame"})
        return jsonify({"success": True, "image": encode_img(frame)})
    except Exception as e:
        return jsonify({"success": False, "message": f"Camera error: {str(e)}"})
    finally:
        if cap is not None:
            cap.release()


# ════════════════════════════════════════════════════════════════
#  Routes — Drowsiness
# ════════════════════════════════════════════════════════════════
@app.route("/drowsiness_status")
def drowsiness_status():
    loaded = _load_drowsiness()
    return jsonify({
        "model_loaded": loaded,
        "message": "Model ready" if loaded else f"Not loaded: {_drowsiness_load_error}",
    })


@app.route("/get_latest_drowsiness")
def get_latest_drowsiness():
    if not _load_drowsiness():
        return jsonify({"drowsy": False, "alert": None,
                        "message": "Model not loaded"})
    return jsonify(_drowsiness_detector.get_state())


# ════════════════════════════════════════════════════════════════
#  MJPEG Streams (Shared Single Camera)
# ════════════════════════════════════════════════════════════════
_LOG_INTERVAL            = 30.0
_last_logged_time        = 0.0
_last_drowsy_logged_time = 0.0
_current_driver          = "driver"


@app.route("/distraction_stream")
def distraction_stream():
    if not _load_distraction():
        def _err_gen():
            blank = np.zeros((480, 640, 3), np.uint8)
            cv2.putText(blank, f"Model not loaded: {_distraction_load_error}",
                        (10, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            _, buf = cv2.imencode('.jpg', blank, [cv2.IMWRITE_JPEG_QUALITY, 60])
            while True:
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'
                       + buf.tobytes() + b'\r\n')
                time.sleep(1)
        return Response(_err_gen(),
                        mimetype="multipart/x-mixed-replace; boundary=frame",
                        headers={"Cache-Control": "no-cache"})

    _start_producer()
    detect_fn = _get_detect_fn()

    def generate():
        global _last_logged_time
        last_seq = 0
        try:
            while True:
                frame, seq = _get_shared_frame()
                if frame is not None and seq > last_seq:
                    last_seq = seq
                    if detect_fn is not None:
                        try:
                            annotated, detections, distracted = detect_fn(frame)
                        except Exception as ex:
                            print(f"[WARN] detect error: {ex}")
                            annotated = frame
                            distracted = False
                        else:
                            if distracted:
                                now = time.time()
                                if now - _last_logged_time >= _LOG_INTERVAL:
                                    _last_logged_time = now
                                    labels = [
                                        {"label": d["label"],
                                         "confidence": d["confidence"]}
                                        for d in detections
                                        if d.get("is_distraction")
                                    ]
                                    if labels:
                                        add_distraction_log(_current_driver, labels)
                                        add_activity_log(
                                            _current_driver, "distraction",
                                            ", ".join(d["label"] for d in labels)
                                            + " detected")
                    else:
                        annotated = frame

                    _, buf = cv2.imencode('.jpg', annotated,
                                         [cv2.IMWRITE_JPEG_QUALITY, 75])
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n'
                           + buf.tobytes() + b'\r\n')
                time.sleep(0.033)
        finally:
            _stop_producer()

    return Response(
        generate(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-cache, no-store, must-revalidate",
                 "X-Accel-Buffering": "no"},
    )


@app.route("/drowsiness_stream")
def drowsiness_stream():
    if not _load_drowsiness():
        def _err_gen():
            blank = np.zeros((480, 640, 3), np.uint8)
            cv2.putText(blank, f"Model not loaded: {_drowsiness_load_error}",
                        (10, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            _, buf = cv2.imencode('.jpg', blank, [cv2.IMWRITE_JPEG_QUALITY, 60])
            while True:
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'
                       + buf.tobytes() + b'\r\n')
                time.sleep(1)
        return Response(_err_gen(),
                        mimetype="multipart/x-mixed-replace; boundary=frame",
                        headers={"Cache-Control": "no-cache"})

    _start_producer()

    if _load_drowsiness():
        _drowsiness_detector.perclos_buf.clear()
        _drowsiness_detector.microsleep_start = None
        _drowsiness_detector.yawn_timestamps.clear()

    def generate():
        global _last_drowsy_logged_time
        last_seq = 0
        try:
            while True:
                frame, seq = _get_shared_frame()
                if frame is not None and seq > last_seq:
                    last_seq = seq
                    if _load_drowsiness():
                        try:
                            annotated, state = _drowsiness_detector.process_frame(frame)
                        except Exception as ex:
                            print(f"[WARN] drowsiness error: {ex}")
                            annotated = frame
                        else:
                            if state.get("drowsy") and state.get("alert_type"):
                                now = time.time()
                                if now - _last_drowsy_logged_time >= _LOG_INTERVAL:
                                    _last_drowsy_logged_time = now
                                    add_drowsiness_log(
                                        _current_driver,
                                        state["alert_type"],
                                        state["alert"]
                                    )
                                    add_activity_log(
                                        _current_driver, "drowsiness",
                                        state["alert_type"] + " - " + state["alert"]
                                    )
                    else:
                        annotated = frame.copy()
                        cv2.putText(annotated,
                                    f"Drowsiness N/A: {_drowsiness_load_error}",
                                    (10, 240), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.5, (0, 0, 255), 1)

                    _, buf = cv2.imencode('.jpg', annotated,
                                         [cv2.IMWRITE_JPEG_QUALITY, 75])
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n'
                           + buf.tobytes() + b'\r\n')
                time.sleep(0.033)
        finally:
            _stop_producer()

    return Response(
        generate(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-cache, no-store, must-revalidate",
                 "X-Accel-Buffering": "no"},
    )


# ════════════════════════════════════════════════════════════════
#  CAMERA 2 PRODUCER — Violence & Weapon Detection
# ════════════════════════════════════════════════════════════════
_cam2_lock      = threading.Lock()
_cam2_producer  = None
_cam2_refcount  = 0
_cam2_frame     = None
_cam2_seq       = 0
_cam2_ready     = threading.Event()


class _Camera2Producer(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self._running = False
        self.cap = None

    def run(self):
        global _cam2_frame, _cam2_seq
        print("[CAM2] Opening camera 2...")
        for cam_idx in [1, 2, 0]:
            self.cap = cv2.VideoCapture(cam_idx)
            if self.cap.isOpened():
                print(f"[CAM2] Camera opened at index {cam_idx}.")
                break
        else:
            print("[CAM2] ERROR: No camera available!")
            return
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        time.sleep(0.5)
        self._running = True
        print("[CAM2] Camera 2 opened OK.")

        while self._running:
            ret, frame = self.cap.read()
            if ret:
                with _cam2_lock:
                    _cam2_frame = frame.copy()
                    _cam2_seq   += 1
                    _cam2_ready.set()
            else:
                time.sleep(0.01)

        self.cap.release()
        print("[CAM2] Camera 2 closed.")

    def stop(self):
        self._running = False


def _start_cam2():
    global _cam2_producer, _cam2_refcount
    with _cam2_lock:
        if _cam2_refcount == 0:
            if _cam2_producer is None or not _cam2_producer.is_alive():
                _cam2_producer = _Camera2Producer()
                _cam2_producer.start()
                print("[CAM2] Camera 2 producer started.")
        _cam2_refcount += 1
    _cam2_ready.wait(timeout=5)


def _stop_cam2():
    global _cam2_producer, _cam2_refcount
    with _cam2_lock:
        _cam2_refcount = max(0, _cam2_refcount - 1)
        if _cam2_refcount == 0:
            if _cam2_producer and _cam2_producer.is_alive():
                _cam2_producer.stop()
                _cam2_producer = None
            _cam2_ready.clear()
            print("[CAM2] Camera 2 producer stopped.")


def _get_cam2_frame():
    with _cam2_lock:
        if _cam2_frame is not None:
            return _cam2_frame.copy(), _cam2_seq
    return None, 0


# ════════════════════════════════════════════════════════════════
#  LAZY LOADING — Violence & Weapon Detector
# ════════════════════════════════════════════════════════════════
_vw_load_error = None
_vw_lock       = threading.Lock()


def _load_vw():
    global _vw_load_error
    with _vw_lock:
        try:
            import violence_weapon
            if not violence_weapon.load_detector():
                _vw_load_error = "Violence+Weapon detector init failed"
                return False
            return True
        except Exception as e:
            _vw_load_error = str(e)
            print(f"[ERROR] Violence+Weapon load failed: {e}")
            return False


def _vw_process(frame):
    import violence_weapon
    return violence_weapon.process_frame(frame)


def _vw_state():
    import violence_weapon
    return violence_weapon.get_state()


def add_violence_log(user, threat_level, threat_name, violence_score, weapon_count, alert):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # Save to JSON
    entries = _load_json(VIOLENCE_LOG)
    entries.append({
        "user":           user,
        "timestamp":      ts,
        "threat_level":   threat_level,
        "threat_name":    threat_name,
        "violence_score": violence_score,
        "weapon_count":   weapon_count,
        "alert":          alert,
    })
    _save_json(VIOLENCE_LOG, entries)
    # Save to SQLite
    try:
        with _db_lock:
            conn = _get_log_db()
            try:
                conn.execute(
                    "INSERT INTO violence_logs (user, timestamp, threat_level, threat_name, violence_score, weapon_count, alert) VALUES (?,?,?,?,?,?,?)",
                    (user, ts, threat_level, threat_name, violence_score, weapon_count, alert),
                )
                conn.commit()
            finally:
                conn.close()
    except Exception as e:
        print(f"[DB ERROR] violence_log: {e}")


# ════════════════════════════════════════════════════════════════
#  Routes — Violence & Weapon
# ════════════════════════════════════════════════════════════════
@app.route("/violence_weapon_status")
def violence_weapon_status():
    loaded = _load_vw()
    return jsonify({
        "model_loaded": loaded,
        "message": "Model ready" if loaded else f"Not loaded: {_vw_load_error}",
    })


@app.route("/get_latest_violence_weapon")
def get_latest_violence_weapon():
    if not _load_vw():
        return jsonify({
            "threat_level": 0, "threat_name": "OFFLINE",
            "threat_color": "#6b7a90", "violence_score": 0,
            "weapon_count": 0, "weapon_detections": [],
            "fps": 0, "alert": None, "is_running": False,
            "message": "Model not loaded",
        })
    return jsonify(_vw_state())


@app.route("/get_violence_log")
def get_violence_log():
    return jsonify(list(reversed(_load_json(VIOLENCE_LOG))))


@app.route("/stop_camera2", methods=["POST"])
def stop_camera2():
    global _cam2_producer, _cam2_refcount, _cam2_frame, _cam2_seq
    with _cam2_lock:
        _cam2_refcount = 0
        if _cam2_producer and _cam2_producer.is_alive():
            _cam2_producer.stop()
            _cam2_producer = None
        _cam2_frame = None
        _cam2_seq = 0
        _cam2_ready.clear()
    print("[CAM2] Camera 2 force-stopped.")
    return jsonify({"success": True})


@app.route("/stop_all_cameras", methods=["POST"])
def stop_all_cameras():
    global _camera_producer, _stream_refcount, _shared_frame, _frame_seq
    global _cam2_producer, _cam2_refcount, _cam2_frame, _cam2_seq

    with _camera_lock:
        _stream_refcount = 0
        if _camera_producer and _camera_producer.is_alive():
            _camera_producer.stop()
            _camera_producer = None
        _shared_frame = None
        _frame_seq = 0
        _producer_ready.clear()

    with _cam2_lock:
        _cam2_refcount = 0
        if _cam2_producer and _cam2_producer.is_alive():
            _cam2_producer.stop()
            _cam2_producer = None
        _cam2_frame = None
        _cam2_seq = 0
        _cam2_ready.clear()

    print("[CAM] All cameras stopped by Stop button.")
    return jsonify({"success": True})


_last_vw_logged_time = 0.0


@app.route("/violence_weapon_stream")
def violence_weapon_stream():
    if not _load_vw():
        def _err_gen():
            blank = np.zeros((480, 640, 3), np.uint8)
            cv2.putText(blank, f"Model not loaded: {_vw_load_error}",
                        (10, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            _, buf = cv2.imencode('.jpg', blank, [cv2.IMWRITE_JPEG_QUALITY, 60])
            while True:
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'
                       + buf.tobytes() + b'\r\n')
                time.sleep(1)
        return Response(_err_gen(),
                        mimetype="multipart/x-mixed-replace; boundary=frame",
                        headers={"Cache-Control": "no-cache"})

    _start_cam2()

    def generate():
        global _last_vw_logged_time
        last_seq = 0
        try:
            while True:
                frame, seq = _get_cam2_frame()
                if frame is not None and seq > last_seq:
                    last_seq = seq
                    if _load_vw():
                        try:
                            annotated, state = _vw_process(frame)
                        except Exception as ex:
                            print(f"[WARN] violence_weapon error: {ex}")
                            annotated = frame
                        else:
                            if state.get("threat_level", 0) >= 2 and state.get("alert"):
                                now = time.time()
                                if now - _last_vw_logged_time >= _LOG_INTERVAL:
                                    _last_vw_logged_time = now
                                    add_violence_log(
                                        _current_driver,
                                        state["threat_level"],
                                        state["threat_name"],
                                        state["violence_score"],
                                        state["weapon_count"],
                                        state["alert"],
                                    )
                                    add_activity_log(
                                        _current_driver, "violence_weapon",
                                        f"{state['threat_name']} - {state['alert']}"
                                    )
                    else:
                        annotated = frame.copy()
                        cv2.putText(annotated,
                                    f"V&W N/A: {_vw_load_error}",
                                    (10, 240), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.5, (0, 0, 255), 1)

                    _, buf = cv2.imencode('.jpg', annotated,
                                         [cv2.IMWRITE_JPEG_QUALITY, 75])
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n'
                           + buf.tobytes() + b'\r\n')
                time.sleep(0.033)
        finally:
            _stop_cam2()

    return Response(
        generate(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-cache, no-store, must-revalidate",
                 "X-Accel-Buffering": "no"},
    )


@app.route("/admin_tab_violence")
def admin_tab_violence():
    return jsonify(list(reversed(_load_json(VIOLENCE_LOG))))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
