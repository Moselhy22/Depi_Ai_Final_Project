import os
import random
import shutil

random.seed(42)

sources = {
    "Phone": r"D:\DEPI GP\data\roboflow_yolo_ds\Phone",
    "Smoking": r"D:\DEPI GP\data\roboflow_yolo_ds\Cigarette and Vape",
    "Drink": r"D:\DEPI GP\data\roboflow_yolo_ds\Bottles and Cups"
}

output_root = r"D:\DEPI GP\data\roboflow_yolo_ds\final_merged"
splits = ["train", "valid", "test"]

target = 7000

def collect(path):
    items = []
    for sp in splits:
        img_dir = os.path.join(path, sp, "images")
        lbl_dir = os.path.join(path, sp, "labels")

        if not os.path.exists(img_dir):
            continue

        for f in os.listdir(img_dir):
            if f.endswith((".jpg", ".png", ".jpeg")):
                img = os.path.join(img_dir, f)
                lbl = os.path.join(lbl_dir, os.path.splitext(f)[0] + ".txt")
                if os.path.exists(lbl):
                    items.append((img, lbl, sp))
    return items

def save(img, lbl, sp, cls):
    out_img = os.path.join(output_root, sp, "images")
    out_lbl = os.path.join(output_root, sp, "labels")
    os.makedirs(out_img, exist_ok=True)
    os.makedirs(out_lbl, exist_ok=True)

    name = cls + "_" + os.path.basename(img)

    shutil.copy2(img, os.path.join(out_img, name))
    shutil.copy2(lbl, os.path.join(out_lbl, os.path.splitext(name)[0] + ".txt"))

def sample(data, n):
    random.shuffle(data)
    return data[:n]

phone = sample(collect(sources["Phone"]), target)
smoking = sample(collect(sources["Smoking"]), target)
drink = sample(collect(sources["Drink"]), target)

counts = {
    "Phone": len(phone),
    "Smoking": len(smoking),
    "Drink": len(drink)
}

for img, lbl, sp in phone:
    save(img, lbl, sp, "Phone")

for img, lbl, sp in smoking:
    save(img, lbl, sp, "Smoking")

for img, lbl, sp in drink:
    save(img, lbl, sp, "Drink")

print("FINAL DATASET READY")
for k, v in counts.items():
    print(k, ":", v)