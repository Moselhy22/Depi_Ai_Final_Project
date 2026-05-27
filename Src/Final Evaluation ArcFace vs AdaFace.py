import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report
from sklearn.metrics.pairwise import cosine_similarity

ARC_DIR = r"D:\DEPI GP\data\arcface_embeddings"
ADA_DIR = r"D:\DEPI GP\data\adaface_embeddings"
OUT_DIR = r"D:\DEPI GP\data\evaluation_results"

os.makedirs(OUT_DIR, exist_ok=True)

REPORT_PATH = os.path.join(OUT_DIR, "final_report.txt")

def log(txt):
    print(txt)
    with open(REPORT_PATH, "a", encoding="utf-8") as f:
        f.write(txt + "\n")

def load_embeddings(path):
    data = {}
    for person in os.listdir(path):
        p = os.path.join(path, person)
        if not os.path.isdir(p):
            continue
        vecs = [np.load(os.path.join(p, f)) for f in os.listdir(p)]
        if len(vecs) > 1:
            data[person] = vecs
    return data

def build_balanced_pairs(data, n=2000):
    people = list(data.keys())

    pos, neg = [], []

    for _ in range(n // 2):
        p = np.random.choice(people)
        vecs = data[p]
        if len(vecs) < 2:
            continue
        i, j = np.random.choice(len(vecs), 2, replace=False)
        pos.append(cosine_similarity([vecs[i]], [vecs[j]])[0][0])

    for _ in range(n // 2):
        p1, p2 = np.random.choice(people, 2, replace=False)
        v1 = data[p1][np.random.randint(len(data[p1]))]
        v2 = data[p2][np.random.randint(len(data[p2]))]
        neg.append(cosine_similarity([v1], [v2])[0][0])

    y_true = np.array([1]*len(pos) + [0]*len(neg))
    y_score = np.array(pos + neg)

    return y_true, y_score

def find_best_threshold(y_true, y_score):
    best_t, best_acc = 0, 0
    for t in np.linspace(0, 1, 100):
        pred = (y_score > t).astype(int)
        acc = (pred == y_true).mean()
        if acc > best_acc:
            best_acc = acc
            best_t = t
    return best_t, best_acc

def eer_score(fpr, tpr):
    fnr = 1 - tpr
    return fpr[np.nanargmin(np.abs(fpr - fnr))]

def evaluate(name, data):
    y_true, y_score = build_balanced_pairs(data)

    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    eer = eer_score(fpr, tpr)

    best_t, best_acc = find_best_threshold(y_true, y_score)
    y_pred = (y_score > best_t).astype(int)

    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred)

    log("\n==============================")
    log(f"{name} RESULTS")
    log("==============================")
    log(f"AUC: {roc_auc:.4f}")
    log(f"EER: {eer:.4f}")
    log(f"Best Threshold: {best_t:.3f}")
    log(f"Best Accuracy: {best_acc:.4f}")
    log("\nClassification Report:\n" + report)
    log("Confusion Matrix:\n" + str(cm))

    # ROC
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC={roc_auc:.3f} | EER={eer:.3f}")
    plt.plot([0,1],[0,1],'--')
    plt.title(f"{name} ROC")
    plt.legend()
    plt.savefig(os.path.join(OUT_DIR, f"{name}_roc.png"))
    plt.close()

    # Distribution
    plt.figure()
    plt.hist(y_score[y_true==1], bins=50, alpha=0.5, label="Same")
    plt.hist(y_score[y_true==0], bins=50, alpha=0.5, label="Different")
    plt.title(f"{name} Distribution")
    plt.legend()
    plt.savefig(os.path.join(OUT_DIR, f"{name}_dist.png"))
    plt.close()

    return y_score

arc = load_embeddings(ARC_DIR)
ada = load_embeddings(ADA_DIR)

open(REPORT_PATH, "w").close()

log("START EVALUATION")

arc_scores = evaluate("ArcFace", arc)
ada_scores = evaluate("AdaFace", ada)

plt.figure()
plt.hist(arc_scores, bins=50, alpha=0.5, label="ArcFace")
plt.hist(ada_scores, bins=50, alpha=0.5, label="AdaFace")
plt.legend()
plt.title("Model Comparison")
plt.savefig(os.path.join(OUT_DIR, "comparison.png"))
plt.close()

log("\nDONE")