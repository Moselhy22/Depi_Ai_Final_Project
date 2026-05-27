import os
import cv2
import numpy as np
from insightface.app import FaceAnalysis

INPUT = r"D:\DEPI GP\data\selected_30"
OUTPUT = r"D:\DEPI GP\data\arcface_embeddings"
MODEL_ROOT = r"D:\DEPI GP\models\recognistion_models\Arcface Model"

os.makedirs(OUTPUT, exist_ok=True)

os.environ["INSIGHTFACE_HOME"] = MODEL_ROOT

app = FaceAnalysis(name="buffalo_l", root=MODEL_ROOT)
app.prepare(ctx_id=0)

for person in os.listdir(INPUT):

    person_path = os.path.join(INPUT, person)
    if not os.path.isdir(person_path):
        continue

    save_dir = os.path.join(OUTPUT, person)
    os.makedirs(save_dir, exist_ok=True)

    for img_name in os.listdir(person_path):

        img_path = os.path.join(person_path, img_name)
        img = cv2.imread(img_path)

        if img is None:
            continue

        faces = app.get(img)

        if len(faces) > 0:
            emb = faces[0].embedding
            np.save(os.path.join(save_dir, img_name.replace(".jpg", ".npy")), emb)

    print("Done:", person)

print("DONE")