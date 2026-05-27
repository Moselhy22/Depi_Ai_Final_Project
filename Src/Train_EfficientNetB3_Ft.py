import os
import json
import torch
import mlflow
import mlflow.pytorch
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix, classification_report

import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from torchvision.models import efficientnet_b3


def main():

    MY_DATA_DIR = r"D:\DEPI GP\data\Own Created Data\images"
    BASE_MODEL  = r"D:\DEPI GP\models\efficientnet_driver_distraction_b3\best_model.pth"
    SAVE_DIR    = r"D:\DEPI GP\models\efficientnet_driver_distraction_b3_Ft"
    MODEL_NAME  = "efficientnet_driver_distraction_b3_Ft"

    os.makedirs(SAVE_DIR, exist_ok=True)

    BATCH_SIZE   = 8
    IMAGE_SIZE   = 300
    EPOCHS       = 20
    PATIENCE     = 5
    LR_HEAD      = 1e-3
    LR_BACKBONE  = 1e-5

    CLASS_DISPLAY = {
        "c0":  "Safe Driving",
        "c1":  "Texting Right",
        "c2":  "Phone Call Right",
        "c3":  "Texting Left",
        "c4":  "Phone Call Left",
        "c5":  "Headphones",
        "c6":  "Drinking",
        "c9":  "Talking Passenger",
        "c10": "Smoking",
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        transforms.RandomGrayscale(p=0.05),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    full_dataset = datasets.ImageFolder(MY_DATA_DIR, transform=train_transform)
    class_names  = full_dataset.classes
    num_classes  = len(class_names)

    train_size = int(0.8 * len(full_dataset))
    val_size   = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    val_dataset.dataset.transform = val_transform

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    model = efficientnet_b3(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 10)
    model.load_state_dict(torch.load(BASE_MODEL, map_location=device))

    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

    for param in model.features.parameters():
        param.requires_grad = False
    for param in model.features[-3:].parameters():
        param.requires_grad = True

    model = model.to(device)

    optimizer = optim.Adam([
        {"params": model.features[-3:].parameters(), "lr": LR_BACKBONE},
        {"params": model.classifier.parameters(),    "lr": LR_HEAD},
    ])

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)
    criterion = nn.CrossEntropyLoss()
    scaler    = torch.cuda.amp.GradScaler()

    mlflow.set_experiment("Driver_Distraction_Classification")

    best_val_acc       = 0.0
    early_stop_counter = 0
    best_model_path    = os.path.join(SAVE_DIR, "best_model.pth")

    with mlflow.start_run(run_name=MODEL_NAME):

        mlflow.log_param("batch_size",    BATCH_SIZE)
        mlflow.log_param("image_size",    IMAGE_SIZE)
        mlflow.log_param("epochs",        EPOCHS)
        mlflow.log_param("lr_head",       LR_HEAD)
        mlflow.log_param("lr_backbone",   LR_BACKBONE)
        mlflow.log_param("num_classes",   num_classes)
        mlflow.log_param("base_model",    BASE_MODEL)

        for epoch in range(EPOCHS):

            model.train()
            train_correct = 0
            train_total   = 0

            for images, labels in train_loader:

                images = images.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.cuda.amp.autocast():
                    outputs = model(images)
                    loss    = criterion(outputs, labels)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                _, predicted   = torch.max(outputs, 1)
                train_total   += labels.size(0)
                train_correct += (predicted == labels).sum().item()

            train_acc = 100 * train_correct / train_total

            model.eval()
            val_correct = 0
            val_total   = 0
            all_preds   = []
            all_labels  = []

            with torch.no_grad():

                for images, labels in val_loader:

                    images = images.to(device)
                    labels = labels.to(device)

                    outputs = model(images)

                    _, predicted  = torch.max(outputs, 1)
                    val_total    += labels.size(0)
                    val_correct  += (predicted == labels).sum().item()

                    all_preds.extend(predicted.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())

            val_acc = 100 * val_correct / val_total

            scheduler.step(val_acc)

            print(f"Epoch [{epoch+1}/{EPOCHS}]")
            print(train_acc)
            print(val_acc)

            mlflow.log_metric("train_accuracy",      train_acc, step=epoch)
            mlflow.log_metric("validation_accuracy", val_acc,   step=epoch)

            if val_acc > best_val_acc:
                best_val_acc       = val_acc
                early_stop_counter = 0
                torch.save(model.state_dict(), best_model_path)
            else:
                early_stop_counter += 1

            if early_stop_counter >= PATIENCE:
                break

        model.load_state_dict(torch.load(best_model_path))
        model.eval()
        all_preds  = []
        all_labels = []

        with torch.no_grad():

            for images, labels in val_loader:

                images = images.to(device)
                outputs = model(images)

                _, predicted = torch.max(outputs, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.numpy())

        display_names = [CLASS_DISPLAY.get(c, c) for c in class_names]

        cm = confusion_matrix(all_labels, all_preds)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt="d", xticklabels=display_names, yticklabels=display_names)
        cm_path = os.path.join(SAVE_DIR, "confusion_matrix.png")
        plt.savefig(cm_path)
        plt.close()

        report = classification_report(all_labels, all_preds, target_names=display_names, output_dict=True)
        report_path = os.path.join(SAVE_DIR, "classification_report.json")
        with open(report_path, "w") as f:
            json.dump(report, f, indent=4)

        class_map = {str(i): CLASS_DISPLAY.get(c, c) for i, c in enumerate(class_names)}
        class_map_path = os.path.join(SAVE_DIR, "class_names.json")
        with open(class_map_path, "w", encoding="utf-8") as f:
            json.dump(class_map, f, ensure_ascii=False, indent=4)

        mlflow.log_artifact(cm_path)
        mlflow.log_artifact(report_path)
        mlflow.log_artifact(class_map_path)
        mlflow.pytorch.log_model(model, "model")

    print(best_val_acc)
    print(best_model_path)


if __name__ == "__main__":
    main()
