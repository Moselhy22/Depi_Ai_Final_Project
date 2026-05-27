import os
import cv2
from ultralytics import YOLO


BASE = r"D:\DEPI GP\data"
INPUT = os.path.join(BASE, "selected_30")
OUTPUT = os.path.join(BASE, "cropped_faces")
MODEL_PATH = r"D:\DEPI GP\models\detection_model\yolov8m-face.pt"

os.makedirs(OUTPUT, exist_ok=True)

model = YOLO(MODEL_PATH)


for person in os.listdir(INPUT):

    in_path = os.path.join(INPUT, person)
    out_path = os.path.join(OUTPUT, person)

    os.makedirs(out_path, exist_ok=True)

    for img_name in os.listdir(in_path):

        img_path = os.path.join(in_path, img_name)
        img = cv2.imread(img_path)

        if img is None:
            continue

        results = model(img)

        saved = False

        for r in results:
            for box in r.boxes.xyxy:

                x1, y1, x2, y2 = map(int, box)

                face = img[y1:y2, x1:x2]

                if face.size > 0:

                    save_path = os.path.join(out_path, img_name)
                    cv2.imwrite(save_path, face)

                    saved = True
                    break

            if saved:
                break

        if not saved:
            cv2.imwrite(os.path.join(out_path, img_name), img)

    print("Done:", person)

print("✅ Finished YOLO face cropping")