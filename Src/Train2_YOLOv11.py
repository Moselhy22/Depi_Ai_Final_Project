from ultralytics import YOLO
import torch
import os
import json


def main():

    BASE_MODEL = r"D:\DEPI GP\models\destraction_model\yolo11n.pt"

    SAVE_DIR = r"D:\DEPI GP\models\destraction_model\train2"

    DATA_YAML = r"D:\DEPI GP\data\final_dataset_clean\data.yaml"

    os.makedirs(SAVE_DIR, exist_ok=True)

    print("\n================ GPU INFO ================\n")

    print("CUDA Available:", torch.cuda.is_available())

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    print("\n================ TRAINING STARTED ================\n")

    model = YOLO(BASE_MODEL)

    results = model.train(

        data=DATA_YAML,

        epochs=100,

        imgsz=640,

        batch=8,

        device=0,

        workers=0,

        freeze=10,

        patience=15,

        lr0=0.001,

        amp=False,

        project=SAVE_DIR,

        name="driver_distraction_yolo11",

        exist_ok=True,

        verbose=True,

        save=True,

        save_period=1
    )

    best_model_path = os.path.join(
        SAVE_DIR,
        "driver_distraction_yolo11",
        "weights",
        "best.pt"
    )

    print("\n================ EVALUATION =================\n")

    model = YOLO(best_model_path)

    metrics = model.val(
        data=DATA_YAML,
        split="test",
        device=0,
        project=SAVE_DIR,
        name="evaluation",
        exist_ok=True
    )

    report = {
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "mAP50": float(metrics.box.map50),
        "mAP50-95": float(metrics.box.map)
    }

    report_path = os.path.join(
        SAVE_DIR,
        "training_report.json"
    )

    with open(report_path, "w") as f:
        json.dump(report, f, indent=4)

    print("\n================ FINAL RESULTS ================\n")

    print(report)

    print("\n✔ Model saved at:")
    print(best_model_path)

    print("\n✔ Report saved at:")
    print(report_path)


if __name__ == "__main__":
    main()