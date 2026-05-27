import os
import json
import numpy as np
import cv2
from datetime import datetime
from insightface.app import FaceAnalysis
from spoof import check_spoof

LOCAL_MODEL = r"D:\DEPI GP\models\recognistion_models\Arcface Model"
DATA_PATH   = r"D:\DEPI GP\data"
DB_PATH     = os.path.join(DATA_PATH, "users_db.json")

os.makedirs(DATA_PATH, exist_ok=True)
os.makedirs(os.path.join(DATA_PATH, "users"), exist_ok=True)


def _init_app():
    try:
        fa = FaceAnalysis(name="buffalo_l", root=LOCAL_MODEL)
    except Exception:
        fa = FaceAnalysis(name="buffalo_l")
    fa.prepare(ctx_id=-1, det_size=(320, 320), det_thresh=0.5)
    return fa

_face_app = _init_app()


# ── DB helpers ───────────────────────────────────────────────
def load_db():
    if not os.path.exists(DB_PATH):
        return []
    try:
        with open(DB_PATH, "r") as f:
            content = f.read()
            if not content:
                return []
            data = json.loads(content)
            if isinstance(data, dict):
                return [{"name": n, "folder": p, "timestamp": "Unknown"}
                        for n, p in data.items()]
            return data
    except Exception:
        return []


def save_db(db):
    with open(DB_PATH, "w") as f:
        json.dump(db, f, indent=4)


# ── Math ─────────────────────────────────────────────────────
def cosine(a, b):
    a, b = np.asarray(a, dtype=np.float32), np.asarray(b, dtype=np.float32)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom > 0 else 0.0


# ── Face helpers ─────────────────────────────────────────────
def get_face(img):
    faces = _face_app.get(img)
    if not faces:
        return None
    # Return the largest face (most likely the driver)
    return max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))


def draw_box(img, face, text="", color=(0, 255, 0)):
    x1, y1, x2, y2 = map(int, face.bbox)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    if text:
        (w, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(img, (x1, y1 - 25), (x1 + w + 4, y1), color, -1)
        cv2.putText(img, text, (x1 + 2, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)
    return img


# ═══════════════════════════════════════════
#  Register
# ═══════════════════════════════════════════
def register_user(name: str, img: np.ndarray):
    face = get_face(img)
    if face is None:
        return False, "No face detected", None

    user_dir = os.path.join(DATA_PATH, "users", name)
    os.makedirs(user_dir, exist_ok=True)

    np.save(os.path.join(user_dir, "embedding.npy"), face.embedding)

    img_with_box = draw_box(img.copy(), face, name)
    img_path = os.path.join(user_dir, "face.jpg")
    cv2.imwrite(img_path, img_with_box)

    db = load_db()
    db = [u for u in db if u.get("name") != name]   # replace if re-registering
    db.append({
        "name":       name,
        "folder":     user_dir,
        "image_path": img_path,
        "timestamp":  datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    })
    save_db(db)

    return True, "User registered successfully", img_with_box


# ═══════════════════════════════════════════
#  Login
# ═══════════════════════════════════════════
def login_user(name: str, img: np.ndarray):
    db = load_db()
    user_data = next((u for u in db if u.get("name") == name), None)
    if not user_data:
        return False, "User not found", None

    face = get_face(img)
    if face is None:
        return False, "No face detected", None

    # Crop face region for spoof check
    x1, y1, x2, y2 = map(int, face.bbox)
    h, w = img.shape[:2]
    crop = img[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]

    is_real = True
    try:
        is_real = check_spoof(crop)
    except Exception:
        pass   # If spoof check errors, assume real to avoid false lockout

    if not is_real:
        return False, "Spoof detected — use a real face", draw_box(img.copy(), face, "Spoof!", (0, 0, 255))

    ref_path = os.path.join(user_data["folder"], "embedding.npy")
    if not os.path.exists(ref_path):
        return False, "User data corrupted — please re-register", None

    ref   = np.load(ref_path)
    score = cosine(face.embedding, ref)

    if score > 0.6:
        processed = draw_box(img.copy(), face, f"Match {score:.2f}", (0, 255, 0))
        return True, f"Welcome {name}!", processed
    else:
        processed = draw_box(img.copy(), face, f"No match {score:.2f}", (0, 0, 255))
        return False, "Identity mismatch", processed
