import os
import cv2
import albumentations as A
from collections import Counter

BASE = r"D:\DEPI GP\data\final_dataset_clean"

IMG_DIR = os.path.join(BASE, "images", "train")
LBL_DIR = os.path.join(BASE, "labels", "train")

TARGET_CLASSES = [0, 4]  # phone + audio_device
AUG_REPEAT = 2

transform = A.Compose([
    A.RandomBrightnessContrast(p=0.5),
    A.MotionBlur(p=0.3),
    A.GaussNoise(p=0.3),
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=10, p=0.7)
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))


def load_labels(path):
    boxes, labels = [], []

    if not os.path.exists(path):
        return boxes, labels

    with open(path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue

            cls = int(parts[0])
            x,y,w,h = map(float, parts[1:])
            boxes.append([x,y,w,h])
            labels.append(cls)

    return boxes, labels


def save_labels(path, boxes, labels):
    with open(path, "w") as f:
        for b,l in zip(boxes, labels):
            f.write(f"{l} {b[0]} {b[1]} {b[2]} {b[3]}\n")


images = os.listdir(IMG_DIR)

aug_count = 0
class_counter = Counter()

for img_name in images:

    img_path = os.path.join(IMG_DIR, img_name)
    lbl_path = os.path.join(LBL_DIR, img_name.replace(".jpg", ".txt"))

    image = cv2.imread(img_path)
    if image is None:
        continue

    boxes, labels = load_labels(lbl_path)

    if not any(l in TARGET_CLASSES for l in labels):
        continue

    for i in range(AUG_REPEAT):

        try:
            transformed = transform(
                image=image,
                bboxes=boxes,
                class_labels=labels
            )

            aug_img = transformed["image"]
            aug_boxes = transformed["bboxes"]
            aug_labels = transformed["class_labels"]

            new_name = img_name.replace(".jpg", f"_aug{i}.jpg")
            new_lbl = new_name.replace(".jpg", ".txt")

            cv2.imwrite(os.path.join(IMG_DIR, new_name), aug_img)
            save_labels(os.path.join(LBL_DIR, new_lbl), aug_boxes, aug_labels)

            aug_count += 1

            for c in aug_labels:
                class_counter[c] += 1

        except:
            continue


print("\n================ AUGMENTATION REPORT ================\n")

print("✔ Total augmented images:", aug_count)

print("\n📊 Per class augmentation:")
for k,v in class_counter.items():
    print(f"class {k} -> {v}")

print("\n🚀 Done")