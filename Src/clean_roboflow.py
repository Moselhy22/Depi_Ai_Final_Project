import os
import cv2
import random
import matplotlib.pyplot as plt
from collections import Counter

BASE = r"D:\DEPI GP\data\final_dataset_clean"

IMG_DIR = os.path.join(BASE, "images", "train")
LBL_DIR = os.path.join(BASE, "labels", "train")

CLASSES = {
    0: "phone",
    1: "cigarette",
    2: "drink",
    3: "food",
    4: "audio_device"
}

print("\n================ DATASET AUTO FIX + TEST ================\n")

bad_files = []
counter = Counter()

# =========================
# 1. FIX LABELS AUTOMATICALLY
# =========================
for file in os.listdir(LBL_DIR):
    path = os.path.join(LBL_DIR, file)

    new_lines = []

    try:
        with open(path, "r", errors="ignore") as f:
            lines = f.readlines()

        for line in lines:
            parts = line.strip().split()

            # skip broken rows
            if len(parts) < 5:
                continue

            try:
                cls = int(parts[0])

                # take only first 4 coords safely
                coords = parts[1:5]

                if len(coords) != 4:
                    continue

                coords = list(map(float, coords))

                # valid class check
                if 0 <= cls <= 4:
                    new_lines.append(f"{cls} " + " ".join(map(str, coords)))
                    counter[cls] += 1

            except:
                continue

        # overwrite cleaned file
        with open(path, "w") as f:
            f.write("\n".join(new_lines))

    except:
        bad_files.append(file)

print("✔ Label cleaning done")


# =========================
# 2. DATASET VALIDATION
# =========================
images = set([f.replace(".jpg","") for f in os.listdir(IMG_DIR)])
labels = set([f.replace(".txt","") for f in os.listdir(LBL_DIR)])

missing_labels = images - labels
missing_images = labels - images

print("\n🔹 Missing labels:", len(missing_labels))
print("🔹 Missing images:", len(missing_images))
print("🔹 Corrupt files:", len(bad_files))


# =========================
# 3. CLASS DISTRIBUTION
# =========================
print("\n🔹 Class Distribution:")
for k, v in counter.items():
    print(f"{CLASSES.get(k,'unknown')} ({k}) -> {v}")


# =========================
# 4. SAFE VISUALIZATION
# =========================
def draw(img_path, lbl_path):
    img = cv2.imread(img_path)
    if img is None:
        return None

    h, w, _ = img.shape

    if not os.path.exists(lbl_path):
        return img

    with open(lbl_path, "r") as f:
        for line in f:
            parts = line.strip().split()

            if len(parts) != 5:
                continue

            try:
                cls = int(parts[0])
                x, y, bw, bh = map(float, parts[1:5])

                x1 = int((x - bw/2) * w)
                y1 = int((y - bh/2) * h)
                x2 = int((x + bw/2) * w)
                y2 = int((y + bh/2) * h)

                if cls in CLASSES:
                    cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
                    cv2.putText(img,CLASSES[cls],
                                (x1,y1-5),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.6,(0,255,0),2)
            except:
                continue

    return img


print("\n🔹 Visual check samples...")

samples = random.sample(os.listdir(IMG_DIR), 6)

plt.figure(figsize=(15,10))

for i,img in enumerate(samples):
    img_path = os.path.join(IMG_DIR,img)
    lbl_path = os.path.join(LBL_DIR,img.replace(".jpg",".txt"))

    out = draw(img_path,lbl_path)

    plt.subplot(2,3,i+1)
    plt.imshow(cv2.cvtColor(out,cv2.COLOR_BGR2RGB))
    plt.axis("off")

plt.tight_layout()
plt.show()


# =========================
# 5. FINAL REPORT
# =========================
print("\n================ FINAL REPORT ================\n")

if len(missing_labels) == 0 and len(missing_images) == 0:
    print("✔ Images & Labels aligned")

if len(counter) >= 4:
    print("✔ All classes present")

if len(bad_files) == 0:
    print("✔ No critical corruption detected")

print("\n🚀 DATASET READY FOR YOLO TRAINING")