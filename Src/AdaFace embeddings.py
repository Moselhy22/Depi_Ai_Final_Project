import os
import cv2
import torch
import numpy as np
import sys

MODEL_DIR = r"D:\DEPI GP\models\recognistion_models\Adaface Model"
CKPT_PATH = r"D:\DEPI GP\models\recognistion_models\Adaface Model\adaface_ir101_ms1mv2.ckpt"
INPUT_DIR = r"D:\DEPI GP\data\cropped_faces"
OUTPUT_DIR = r"D:\DEPI GP\data\adaface_embeddings"

os.makedirs(OUTPUT_DIR, exist_ok=True)

sys.path.append(MODEL_DIR)
from net import build_model

ckpt = torch.load(CKPT_PATH, map_location='cpu')

model = build_model('ir_101')
model.load_state_dict(ckpt['state_dict'], strict=False)
model.eval()

def preprocess(img):
    img = cv2.resize(img, (112, 112))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = (img / 255.0 - 0.5) / 0.5
    img = np.transpose(img, (2, 0, 1))
    return torch.tensor(img, dtype=torch.float32).unsqueeze(0)

for person in os.listdir(INPUT_DIR):
    person_path = os.path.join(INPUT_DIR, person)
    if not os.path.isdir(person_path):
        continue

    save_dir = os.path.join(OUTPUT_DIR, person)
    os.makedirs(save_dir, exist_ok=True)

    for img_name in os.listdir(person_path):
        img_path = os.path.join(person_path, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue

        tensor = preprocess(img)

        with torch.no_grad():
            emb = model(tensor)[0]

        emb = emb.squeeze().cpu().numpy()

        np.save(os.path.join(save_dir, img_name.replace(".jpg", ".npy")), emb)

print("DONE")