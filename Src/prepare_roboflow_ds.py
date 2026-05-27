import os
import shutil

INPUT = r"D:\DEPI GP\data\roboflow_ds"
OUTPUT = r"D:\DEPI GP\data\final_dataset_clean"

CLASS_MAP = {
    "phone": 0,
    "smoking": 1,

    "bottle": 2,
    "cup": 2,
    "drinks": 2,

    "burger": 3,
    "pizza": 3,
    "sandwich": 3,
    "fries": 3,
    "snack": 3,
    "eating": 3,

    "earphone": 4,
    "headphone": 4
}

for split in ["train", "valid", "test"]:
    os.makedirs(os.path.join(OUTPUT, "images", split), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT, "labels", split), exist_ok=True)


def process_folder(folder):
    folder_path = os.path.join(INPUT, folder)

    for split in ["train", "valid", "test"]:

        img_dir = os.path.join(folder_path, split, "images")
        lbl_dir = os.path.join(folder_path, split, "labels")

        if not os.path.exists(img_dir):
            continue

        for img in os.listdir(img_dir):

            img_path = os.path.join(img_dir, img)
            lbl_path = os.path.join(lbl_dir, img.replace(".jpg", ".txt"))

            out_img = os.path.join(OUTPUT, "images", split, img)
            out_lbl = os.path.join(OUTPUT, "labels", split, img.replace(".jpg", ".txt"))

            shutil.copy(img_path, out_img)

            if os.path.exists(lbl_path):
                with open(lbl_path, "r") as f:
                    lines = f.readlines()

                new_lines = []

                for line in lines:
                    parts = line.strip().split()

                    try:
                        cls = CLASS_MAP[folder]
                        new_lines.append(str(cls) + " " + " ".join(parts[1:]))
                    except:
                        continue

                with open(out_lbl, "w") as f:
                    f.write("\n".join(new_lines))


for f in CLASS_MAP.keys():
    print("Processing:", f)
    process_folder(f)

print("DONE 🚀 Clean dataset ready")