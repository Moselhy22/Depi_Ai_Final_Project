import cv2
import threading
import time

YOLO_PATH = r"D:\DEPI GP\runs\detect\train-6\weights\best.pt"

CONF_THRESHOLD_YOLO = 0.25

SUSTAINED_FRAMES_REQUIRED = 6

YOLO_CLASSES = {
    "phone": "Phone",
    "cigarette": "Smoking",
    "vape": "Smoking",
    "smoking": "Smoking",
    "drink": "Drink",
    "bottle": "Drink",
    "cup": "Drink",
}

LABEL_COLORS = {
    "Phone": (255, 0, 0),
    "Smoking": (0, 0, 255),
    "Drink": (0, 255, 255),
}

# ── Lazy YOLO loading (NOT at module level — prevents server hang) ──
_yolo_model = None
_yolo_lock = threading.Lock()


def _get_yolo():
    global _yolo_model
    if _yolo_model is not None:
        return _yolo_model
    with _yolo_lock:
        if _yolo_model is not None:
            return _yolo_model
        print("\n[DESTRACTION] Loading YOLO model...\n")
        from ultralytics import YOLO
        _yolo_model = YOLO(YOLO_PATH)
        print("[DESTRACTION] Model loaded successfully.\n")
        print("MODEL CLASSES:")
        print(_yolo_model.names)
        return _yolo_model


_state_lock = threading.Lock()

_latest_state = {
    "distracted": False,
    "label": "safe",
    "display": "Safe Driving",
    "confidence": 0.0,
    "detections": [],
    "timestamp": "",
}

_consec_distracted = 0


def get_latest_state():
    with _state_lock:
        return dict(_latest_state)


def detect_distraction(frame):
    global _consec_distracted

    model = _get_yolo()

    results = model(
        frame,
        verbose=False,
        conf=CONF_THRESHOLD_YOLO,
        iou=0.45
    )[0]

    yolo_dets = []

    for box in results.boxes:
        cls_name = results.names[
            int(box.cls.item())
        ].lower()

        if cls_name not in YOLO_CLASSES:
            continue

        conf = float(box.conf.item())

        if conf < CONF_THRESHOLD_YOLO:
            continue

        yolo_dets.append({
            "label": YOLO_CLASSES[cls_name],
            "raw_label": cls_name,
            "conf": conf,
            "bbox": tuple(
                map(int, box.xyxy[0].tolist())
            )
        })

    raw_distracted = len(yolo_dets) > 0

    if raw_distracted:
        _consec_distracted += 1
    else:
        _consec_distracted = 0

    is_confirmed = (
        _consec_distracted >=
        SUSTAINED_FRAMES_REQUIRED
    )

    main_label = (
        yolo_dets[0]["label"]
        if yolo_dets
        else "Safe Driving"
    )

    main_conf = (
        yolo_dets[0]["conf"]
        if yolo_dets
        else 0.0
    )

    detections = []

    for d in yolo_dets:
        detections.append({
            "label": d["label"],
            "class_id": d["raw_label"],
            "confidence": round(d["conf"], 4),
            "is_distraction": is_confirmed,
            "source": "yolo"
        })

    if not detections:
        detections.append({
            "label": "Safe Driving",
            "class_id": "safe",
            "confidence": 1.0,
            "is_distraction": False,
            "source": "yolo"
        })

    with _state_lock:
        _latest_state["distracted"] = is_confirmed
        _latest_state["label"] = (
            main_label.lower()
        )
        _latest_state["display"] = main_label
        _latest_state["confidence"] = main_conf
        _latest_state["detections"] = detections
        _latest_state["timestamp"] = time.strftime(
            "%Y-%m-%d %H:%M:%S"
        )

    annotated = frame.copy()

    for d in yolo_dets:
        x1, y1, x2, y2 = d["bbox"]
        color = LABEL_COLORS.get(
            d["label"],
            (0, 0, 255)
        )
        box_color = (
            (0, 0, 255)
            if is_confirmed
            else color
        )

        cv2.rectangle(
            annotated,
            (x1, y1),
            (x2, y2),
            box_color,
            2
        )

        label_text = (
            f"{d['label']} "
            f"{d['conf']:.0%}"
        )

        (tw, th), _ = cv2.getTextSize(
            label_text,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            2
        )

        cv2.rectangle(
            annotated,
            (x1, y1 - th - 10),
            (x1 + tw + 10, y1),
            box_color,
            -1
        )

        cv2.putText(
            annotated,
            label_text,
            (x1 + 5, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

    status_text = (
        "DISTRACTED"
        if is_confirmed
        else "SAFE"
    )

    status_color = (
        (0, 0, 255)
        if is_confirmed
        else (0, 255, 0)
    )

    cv2.putText(
        annotated,
        status_text,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        status_color,
        3,
        cv2.LINE_AA
    )

    return annotated, detections, is_confirmed


if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    print("\nPRESS Q TO EXIT\n")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        annotated, dets, distracted = (
            detect_distraction(frame)
        )
        labels = [
            d["class_id"]
            for d in dets
            if d.get("is_distraction")
        ]
        print(
            f"\r{' | '.join(labels) or 'Safe'} "
            f"| confirmed={distracted}",
            end=""
        )
        cv2.imshow(
            "Driver Distraction Detection",
            annotated
        )
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()
