#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ============================================================================
# SETUP AND IMPORTS
# ============================================================================

import os
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

import torchvision.transforms as T
from torchvision.datasets import ImageFolder
import timm

from torch.cuda.amp import GradScaler, autocast


# ============================================================================
# CONFIGURATION
# ============================================================================

BATCH_SIZE = 24
EPOCHS = 50
IMG_SIZE = 224
ROOT_DIR = r"C:\Users\LENOVO\Desktop\RAF-DB\DATASET"
BEST_MODEL_PATH = "rafdb_efficientnetv2s_best.pth"


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def check_folders(root_dir):
    train_dir = os.path.join(root_dir, "train")
    test_dir = os.path.join(root_dir, "test")
    if not os.path.isdir(train_dir):
        raise FileNotFoundError(f"Train folder not found: {train_dir}")
    if not os.path.isdir(test_dir):
        raise FileNotFoundError(f"Test folder not found: {test_dir}")
    print("Found train and test folders:")
    print("  Train:", train_dir)
    print("  Test :", test_dir)


def compute_class_weights_from_imagefolder(dataset, num_classes):
    targets = np.array(dataset.targets)
    class_counts = np.bincount(targets, minlength=num_classes)

    class_weights = 1.0 / (class_counts + 1e-6)
    class_weights = class_weights * (num_classes / class_weights.sum())
    return torch.tensor(class_weights, dtype=torch.float)


# ============================================================================
# MODEL DEFINITION
# ============================================================================

def build_model(num_classes):
    model = timm.create_model(
        "tf_efficientnetv2_s_in21k",
        pretrained=True,
        num_classes=num_classes,
        drop_rate=0.4,
        drop_path_rate=0.2,
    )
    return model


# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

def train_one_epoch_amp(model, loader, optimizer, criterion, device, scaler):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()

        with autocast():
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


# ============================================================================
# MAIN TRAINING PIPELINE
# ============================================================================

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    torch.backends.cudnn.benchmark = True

    check_folders(ROOT_DIR)
    train_dir = os.path.join(ROOT_DIR, "train")
    test_dir = os.path.join(ROOT_DIR, "test")

    train_transform = T.Compose([
        T.Resize((256, 256)),
        T.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),
        T.RandomHorizontalFlip(),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        T.RandomRotation(10),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])

    test_transform = T.Compose([
        T.Resize((IMG_SIZE, IMG_SIZE)),
        T.CenterCrop(IMG_SIZE),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])

    full_train = ImageFolder(train_dir, transform=train_transform)
    test_dataset = ImageFolder(test_dir, transform=test_transform)

    num_classes = len(full_train.classes)
    print("Classes:", full_train.classes)
    print("Number of classes:", num_classes)

    val_ratio = 0.1
    val_size = int(len(full_train) * val_ratio)
    train_size = len(full_train) - val_size
    train_dataset, val_dataset = random_split(full_train, [train_size, val_size])

    print(f"Train: {len(train_dataset)}  Val: {len(val_dataset)}  Test: {len(test_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=4, pin_memory=True)

    model = build_model(num_classes).to(device)

    class_weights = compute_class_weights_from_imagefolder(full_train, num_classes).to(device)
    print("Class weights:", class_weights.cpu().numpy())

    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)

    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    scaler = GradScaler()

    best_val_acc = 0.0

    for epoch in range(EPOCHS):
        train_loss, train_acc = train_one_epoch_amp(
            model, train_loader, optimizer, criterion, device, scaler
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        scheduler.step()

        print(f"Epoch {epoch + 1:02d}/{EPOCHS} "
              f"Train loss {train_loss:.4f} acc {train_acc:.4f} "
              f"Val loss {val_loss:.4f} acc {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print(f"  New best model saved with val acc {best_val_acc:.4f}")

    print("Loading best model from:", BEST_MODEL_PATH)
    model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device))

    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"Test loss {test_loss:.4f}  Test acc {test_acc:.4f}")


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()