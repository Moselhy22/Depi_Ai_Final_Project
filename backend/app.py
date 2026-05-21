from flask import Flask, request, jsonify, Response, send_from_directory
from flask_cors import CORS
import cv2
import numpy as np
import base64  
import json
import os
import threading
import time
from datetime import datetime

from recognition import register_user, login_user, DB_PATH

# ── Distraction: lazy-loaded after first /distraction_status call ──
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

def _get_yolo_model():
    return _distraction_module.yolo_model if _load_distraction() else None

def _get_state_fn():
    return _distraction_module.get_latest_state if _load_distraction() else None


# ── Flask ────────────────────────────────────────────────────
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

# ── Paths ────────────────────────────────────────────────────
DATA_PATH       = r"D:\DEPI GP\data"
FRONTEND_DIR    = r"D:\DEPI GP"
LOG_PATH        = os.path.join(DATA_PATH, "activity_log.json")
DISTRACTION_LOG = os.path.join(DATA_PATH, "distraction_log.json")
os.makedirs(DATA_PATH, exist_ok=True)

# Serve frontend from Flask so no Mixed-Content / CORS issues
@app.route("/")
def index():
    return send_from_directory(FRONTEND_DIR, "index.html")

@app.route("/<path:filename>")
def static_files(filename):
    return send_from_directory(FRONTEND_DIR, filename)


# ── JSON helpers (thread-safe) ───────────────────────────────
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
    entries = _load_json(LOG_PATH)
    entries.append({
        "user":      user,
        "action":    action,
        "details":   details,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    })
    _save_json(LOG_PATH, entries)

def add_distraction_log(user, detections):
    entries = _load_json(DISTRACTION_LOG)
    entries.append({
        "user":       user,
        "timestamp":  datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "detections": detections,
    })
    _save_json(DISTRACTION_LOG, entries)


# ── Image helpers ────────────────────────────────────────────
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


# ── Shared camera for MJPEG stream ───────────────────────────
_cam_lock  = threading.Lock()
_cam       = None
_cam_users = 0

def _open_cam():
    global _cam, _cam_users
    with _cam_lock:
        if _cam is None or not _cam.isOpened():
            _cam = cv2.VideoCapture(0)
            _cam.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
            _cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            _cam.set(cv2.CAP_PROP_FPS, 30)
            _cam.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        _cam_users += 1

def _close_cam():
    global _cam, _cam_users
    with _cam_lock:
        _cam_users = max(0, _cam_users - 1)
        if _cam_users == 0 and _cam is not None:
            _cam.release()
            _cam = None


# ════════════════════════════════════════════
#  Routes — users
# ════════════════════════════════════════════
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
    return jsonify({"success": ok, "message": msg, "processed_image": encode_img(processed)})


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
    return jsonify({"success": ok, "message": msg, "processed_image": encode_img(processed)})


@app.route("/set_driver", methods=["POST"])
def set_driver():
    global _current_driver, _last_logged_time
    data = request.json
    name = (data or {}).get("name", "driver").strip()
    if name:
        _current_driver   = name
        _last_logged_time = 0.0   # reset so first event logs immediately
        print(f"[INFO] Active driver set to: {name}")
    return jsonify({"success": True})



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
        return jsonify({"success": False, "message": f"Model not loaded: {_distraction_load_error}"})
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
        return jsonify({"distracted": False, "detections": [], "message": "Model not loaded"})
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


@app.route("/get_activity_log")
def get_activity_log():
    return jsonify(list(reversed(_load_json(LOG_PATH))))


# ════════════════════════════════════════════
#  MJPEG Stream
# ════════════════════════════════════════════

# Log once every 30 seconds per confirmed distraction event
_LOG_INTERVAL     = 30.0
_last_logged_time = 0.0
_current_driver   = "driver"   # updated by /set_driver route


def _generate_frames():
    detect_fn = _get_detect_fn()
    if detect_fn is None:
        err = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(err, f"Model error: {_distraction_load_error}",
                    (20, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        _, buf = cv2.imencode('.jpg', err)
        yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n'
        return

    _open_cam()
    global _last_logged_time
    try:
        while True:
            with _cam_lock:
                if _cam is None or not _cam.isOpened():
                    break
                ret, frame = _cam.read()

            if not ret:
                time.sleep(0.05)
                continue

            try:
                annotated, detections, distracted = detect_fn(frame)
            except Exception as ex:
                print(f"[WARN] detect error: {ex}")
                annotated, distracted = frame, False

            # Auto-log confirmed distraction, max once per _LOG_INTERVAL seconds
            if distracted:
                now = time.time()
                if now - _last_logged_time >= _LOG_INTERVAL:
                    _last_logged_time = now
                    labels = [
                        {"label": d["label"], "confidence": d["confidence"]}
                        for d in detections if d.get("is_distraction")
                    ]
                    if labels:
                        add_distraction_log(_current_driver, labels)
                        add_activity_log(
                            _current_driver, "distraction",
                            ", ".join(d["label"] for d in labels) + " detected"
                        )

            _, buf = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 75])
            yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n'

    finally:
        _close_cam()


@app.route("/distraction_stream")
def distraction_stream():
    return Response(
        _generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
        headers={
            "Cache-Control":    "no-cache, no-store, must-revalidate",
            "X-Accel-Buffering": "no",
        },
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
