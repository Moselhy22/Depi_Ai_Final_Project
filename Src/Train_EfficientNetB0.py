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
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights


def main():

    DATA_DIR = r"D:\DEPI GP\data\Destraction Driver Data\imgs\train"

    SAVE_ROOT = r"D:\DEPI GP\models"

    MODEL_NAME = "efficientnet_driver_distraction"

    SAVE_DIR = os.path.join(SAVE_ROOT, MODEL_NAME)

    os.makedirs(SAVE_DIR, exist_ok=True)

    BATCH_SIZE = 16
    IMAGE_SIZE = 224
    EPOCHS = 30
    PATIENCE = 5
    LEARNING_RATE = 1e-3

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.backends.cudnn.benchmark = True

    print("\n================ GPU INFO ================\n")

    print("CUDA Available:", torch.cuda.is_available())

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    dataset = datasets.ImageFolder(
        DATA_DIR,
        transform=transform
    )

    class_names = dataset.classes

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size]
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    weights = EfficientNet_B0_Weights.DEFAULT

    model = efficientnet_b0(weights=weights)

    for param in model.features.parameters():
        param.requires_grad = False

    model.classifier[1] = nn.Linear(
        model.classifier[1].in_features,
        len(class_names)
    )

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        model.classifier.parameters(),
        lr=LEARNING_RATE
    )

    scaler = torch.cuda.amp.GradScaler()

    mlflow.set_experiment("Driver_Distraction_Classification")

    best_val_acc = 0
    early_stop_counter = 0

    best_model_path = os.path.join(
        SAVE_DIR,
        "best_model.pth"
    )

    with mlflow.start_run(run_name=MODEL_NAME):

        mlflow.log_param("batch_size", BATCH_SIZE)
        mlflow.log_param("image_size", IMAGE_SIZE)
        mlflow.log_param("epochs", EPOCHS)
        mlflow.log_param("learning_rate", LEARNING_RATE)

        for epoch in range(EPOCHS):

            model.train()

            train_correct = 0
            train_total = 0
            train_loss = 0

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

                train_loss += loss.item()

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

            print(f"\nEpoch [{epoch+1}/{EPOCHS}]")
            print(f"Train Accuracy: {train_acc:.2f}%")
            print(f"Validation Accuracy: {val_acc:.2f}%")

            mlflow.log_metric("train_accuracy", train_acc, step=epoch)
            mlflow.log_metric("validation_accuracy", val_acc, step=epoch)

            if val_acc > best_val_acc:

                best_val_acc = val_acc

                torch.save(
                    model.state_dict(),
                    best_model_path
                )

                early_stop_counter = 0

                print("\nBest model saved!\n")

            else:

                early_stop_counter += 1

                print(f"\nNo improvement count: {early_stop_counter}/{PATIENCE}")

            if early_stop_counter >= PATIENCE:

                print("\nEarly Stopping Triggered!\n")

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

        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            xticklabels=class_names,
            yticklabels=class_names
        )

        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("Confusion Matrix")

        confusion_matrix_path = os.path.join(
            SAVE_DIR,
            "confusion_matrix.png"
        )

        plt.savefig(confusion_matrix_path)

        report = classification_report(
            all_labels,
            all_preds,
            target_names=class_names,
            output_dict=True
        )

        report_path = os.path.join(
            SAVE_DIR,
            "classification_report.json"
        )

        with open(report_path, "w") as f:

            json.dump(report, f, indent=4)

        mlflow.log_artifact(confusion_matrix_path)

        mlflow.log_artifact(report_path)

        mlflow.pytorch.log_model(
            model,
            "model"
        )

    print("\n================ FINAL RESULTS ================\n")

    print("Best Validation Accuracy:", best_val_acc)

    print("\nModel Saved At:")
    print(best_model_path)

    print("\nConfusion Matrix Saved At:")
    print(confusion_matrix_path)

    print("\nClassification Report Saved At:")
    print(report_path)


if __name__ == "__main__":
    main()