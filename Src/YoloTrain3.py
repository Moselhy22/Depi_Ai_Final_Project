import os
import torch
import mlflow
from ultralytics import YOLO

def main():

    DATA_YAML = r"D:\DEPI GP\data\FINAL DATA\YOLO_DATASET\data.yaml"

    BASE_MODEL = r"D:\DEPI GP\models\yolov8n\yolov8n_final_ft\best.pt"

    SAVE_DIR = r"D:\DEPI GP\models\yolov8n\yolov8n_final_ft_resume"

    os.makedirs(SAVE_DIR, exist_ok=True)

    # MLflow only for logging (NOT controlling training)
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("YOLO_DRIVER_MONITORING")

    model = YOLO(BASE_MODEL)

    with mlflow.start_run(run_name="resume_train_clean"):

        results = model.train(

            data=DATA_YAML,

            epochs=25,
            imgsz=640,
            batch=6,

            device=0 if torch.cuda.is_available() else "cpu",

            workers=0,
            cache=False,

            optimizer="AdamW",
            lr0=0.0005,

            patience=10,

            amp=False,   # safer for GTX 1650

            plots=True,

            save=True,
            save_period=1,

            project=SAVE_DIR,
            name="train_run",
            exist_ok=True,

            val=True,
            verbose=True
        )

        run_path = os.path.join(SAVE_DIR, "train_run")

        # best / last copy (LOCAL ONLY)
        best = os.path.join(run_path, "weights", "best.pt")
        last = os.path.join(run_path, "weights", "last.pt")

        if os.path.exists(best):
            torch.save(model.model.state_dict(), os.path.join(SAVE_DIR, "best_local.pt"))

        if os.path.exists(last):
            torch.save(model.model.state_dict(), os.path.join(SAVE_DIR, "last_local.pt"))

        print("\nTRAINING DONE ✔")
        print("Saved in:", SAVE_DIR)

if __name__ == "__main__":
    main()