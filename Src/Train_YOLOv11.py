import os
import shutil
import json
from ultralytics import YOLO

DATASET_YAML = r"D:\DEPI GP\data\labeled data yolov11\data.yaml"

MODEL_DIR = r"D:\DEPI GP\models\destraction_model"

BASE_MODEL_NAME = "yolo11n.pt"
BASE_MODEL_PATH = os.path.join(MODEL_DIR, BASE_MODEL_NAME)

TRAINED_MODEL_NAME = "trained_driver_distraction.pt"
TRAINED_MODEL_PATH = os.path.join(MODEL_DIR, TRAINED_MODEL_NAME)

REPORT_PATH = os.path.join(MODEL_DIR, "evaluation_report.json")

os.makedirs(MODEL_DIR, exist_ok=True)

if not os.path.exists(BASE_MODEL_PATH):
    print("\nDownloading YOLOv11 model...\n")

    temp_model = YOLO(BASE_MODEL_NAME)

    downloaded_path = temp_model.ckpt_path

    shutil.copy(downloaded_path, BASE_MODEL_PATH)

    print(f"Model downloaded to:\n{BASE_MODEL_PATH}\n")

else:
    print(f"\nUsing existing model:\n{BASE_MODEL_PATH}\n")

model = YOLO(BASE_MODEL_PATH)

print("\nStarting Training...\n")

model.train(
    data=DATASET_YAML,
    epochs=100,
    imgsz=640,
    batch=16,
    patience=15,
    project=MODEL_DIR,
    name="training_results",
    exist_ok=True,
    verbose=True
)

BEST_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "training_results",
    "weights",
    "best.pt"
)

if os.path.exists(BEST_MODEL_PATH):
    shutil.copy(BEST_MODEL_PATH, TRAINED_MODEL_PATH)

print("\nStarting Evaluation...\n")

trained_model = YOLO(TRAINED_MODEL_PATH)

metrics = trained_model.val(
    data=DATASET_YAML,
    split="test",
    project=MODEL_DIR,
    name="evaluation_results",
    exist_ok=True
)

evaluation_report = {
    "Precision": float(metrics.box.mp),
    "Recall": float(metrics.box.mr),
    "mAP50": float(metrics.box.map50),
    "mAP50-95": float(metrics.box.map)
}

print("\nEvaluation Results:\n")
print(json.dumps(evaluation_report, indent=4))

with open(REPORT_PATH, "w") as f:
    json.dump(evaluation_report, f, indent=4)

SOURCE_RESULTS_DIR = os.path.join(MODEL_DIR, "training_results")
DEST_RESULTS_DIR = os.path.join(MODEL_DIR, "training_graphs")

if os.path.exists(DEST_RESULTS_DIR):
    shutil.rmtree(DEST_RESULTS_DIR)

shutil.copytree(SOURCE_RESULTS_DIR, DEST_RESULTS_DIR)

print("\nTraining Complete!\n")
print(f"Base Model:\n{BASE_MODEL_PATH}")
print(f"\nTrained Model:\n{TRAINED_MODEL_PATH}")
print(f"\nEvaluation Report:\n{REPORT_PATH}")
print(f"\nGraphs & Results:\n{DEST_RESULTS_DIR}")