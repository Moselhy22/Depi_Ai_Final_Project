import os
import shutil
import random

SRC = r"D:\DEPI GP\data\FINAL DATA\Before\Cigarette2"
DST = r"D:\DEPI GP\data\FINAL DATA\YOLO_DATASET"

TARGET = 2000
SMOKING_CLASS_ID = 1

IMG_EXT = [".jpg", ".png", ".jpeg"]

def safe_mkdir(p):
    os.makedirs(p, exist_ok=True)


def get_samples():
    samples = []

    for split in ["train", "valid", "test"]:
        img_dir = os.path.join(SRC, split, "images")
        lbl_dir = os.path.join(SRC, split, "labels")

        if not os.path.exists(img_dir):
            continue

        for f in os.listdir(img_dir):
            if any(f.endswith(e) for e in IMG_EXT):
                img_path = os.path.join(img_dir, f)
                lbl_path = os.path.join(lbl_dir, f.rsplit(".", 1)[0] + ".txt")

                if os.path.exists(lbl_path):
                    samples.append((img_path, lbl_path))

    return samples


def filter_smoking_labels(label_path):
    with open(label_path, "r") as f:
        lines = f.readlines()

    new_lines = []

    for l in lines:
        parts = l.strip().split()
        if len(parts) < 5:
            continue

        cls = int(parts[0])

        if cls == SMOKING_CLASS_ID:
            parts[0] = "0"
            new_lines.append(" ".join(parts) + "\n")

    return new_lines


def save(img_path, lbl_lines, idx):
    img_out = os.path.join(DST, "images", "train")
    lbl_out = os.path.join(DST, "labels", "train")

    safe_mkdir(img_out)
    safe_mkdir(lbl_out)

    new_name = f"cigarette_extra_{idx}.jpg"

    shutil.copy2(img_path, os.path.join(img_out, new_name))

    with open(os.path.join(lbl_out, new_name.replace(".jpg", ".txt")), "w") as f:
        f.writelines(lbl_lines)


def count_classes():
    label_dir = os.path.join(DST, "labels")

    counts = {}

    for f in os.listdir(label_dir):
        if not f.endswith(".txt"):
            continue

        path = os.path.join(label_dir, f)

        with open(path, "r") as file:
            lines = file.readlines()

        for l in lines:
            parts = l.strip().split()
            if len(parts) < 5:
                continue

            cls = int(parts[0])
            counts[cls] = counts.get(cls, 0) + 1

    print("\n========== CLASS COUNTS ==========")
    for k in sorted(counts.keys()):
        print(f"Class {k}: {counts[k]} boxes")

    print("==================================\n")


def main():
    samples = get_samples()
    random.shuffle(samples)

    print("Total raw samples:", len(samples))

    count = 0
    idx = 0

    for img, lbl in samples:

        if count >= TARGET:
            break

        filtered = filter_smoking_labels(lbl)

        if len(filtered) == 0:
            continue

        save(img, filtered, idx)

        idx += 1
        count += 1

    print(f"\n✔ Added {count} cigarette images to dataset")

    # 👇 NEW: count classes after merge
    count_classes()


if __name__ == "__main__":
    main()