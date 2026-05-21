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
from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights


def main():

    DATA_DIR = r"D:\DEPI GP\data\Destraction Driver Data\imgs\train"

    SAVE_ROOT = r"D:\DEPI GP\models"

    MODEL_NAME = "efficientnet_driver_distraction_b3"

    SAVE_DIR = os.path.join(SAVE_ROOT, MODEL_NAME)

    os.makedirs(SAVE_DIR, exist_ok=True)

    BATCH_SIZE = 8
    IMAGE_SIZE = 300
    EPOCHS = 30
    PATIENCE = 5
    LEARNING_RATE = 3e-4

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.backends.cudnn.benchmark = True

    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(0.2, 0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    dataset = datasets.ImageFolder(DATA_DIR, transform=transform)

    class_names = dataset.classes

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)

    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    weights = EfficientNet_B3_Weights.DEFAULT

    model = efficientnet_b3(weights=weights)

    for param in model.features.parameters():
        param.requires_grad = False

    for param in model.features[-2:].parameters():
        param.requires_grad = True

    model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(class_names))

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    scaler = torch.cuda.amp.GradScaler()

    mlflow.set_experiment("Driver_Distraction_Classification")

    best_val_acc = 0
    early_stop_counter = 0

    best_model_path = os.path.join(SAVE_DIR, "best_model.pth")

    with mlflow.start_run(run_name=MODEL_NAME):

        mlflow.log_param("batch_size", BATCH_SIZE)
        mlflow.log_param("image_size", IMAGE_SIZE)
        mlflow.log_param("epochs", EPOCHS)
        mlflow.log_param("learning_rate", LEARNING_RATE)

        for epoch in range(EPOCHS):

            model.train()

            train_correct = 0
            train_total = 0

            for images, labels in train_loader:

                images = images.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.cuda.amp.autocast():

                    outputs = model(images)
                    loss = criterion(outputs, labels)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                _, predicted = torch.max(outputs, 1)

                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()

            train_acc = 100 * train_correct / train_total

            model.eval()

            val_correct = 0
            val_total = 0

            all_preds = []
            all_labels = []

            with torch.no_grad():

                for images, labels in val_loader:

                    images = images.to(device)
                    labels = labels.to(device)

                    outputs = model(images)

                    _, predicted = torch.max(outputs, 1)

                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()

                    all_preds.extend(predicted.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())

            val_acc = 100 * val_correct / val_total

            print(f"Epoch [{epoch+1}/{EPOCHS}]")
            print(train_acc)
            print(val_acc)

            mlflow.log_metric("train_accuracy", train_acc, step=epoch)
            mlflow.log_metric("validation_accuracy", val_acc, step=epoch)

            if val_acc > best_val_acc:

                best_val_acc = val_acc
                torch.save(model.state_dict(), best_model_path)
                early_stop_counter = 0

            else:

                early_stop_counter += 1

            if early_stop_counter >= PATIENCE:

                break

        model.load_state_dict(torch.load(best_model_path))

        model.eval()

        all_preds = []
        all_labels = []

        with torch.no_grad():

            for images, labels in val_loader:

                images = images.to(device)

                outputs = model(images)

                _, predicted = torch.max(outputs, 1)

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.numpy())

        cm = confusion_matrix(all_labels, all_preds)

        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt="d", xticklabels=class_names, yticklabels=class_names)

        cm_path = os.path.join(SAVE_DIR, "confusion_matrix.png")
        plt.savefig(cm_path)

        report = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)

        report_path = os.path.join(SAVE_DIR, "classification_report.json")

        with open(report_path, "w") as f:
            json.dump(report, f, indent=4)

        mlflow.log_artifact(cm_path)
        mlflow.log_artifact(report_path)
        mlflow.pytorch.log_model(model, "model")

    print(best_val_acc)
    print(best_model_path)


if __name__ == "__main__":
    main()