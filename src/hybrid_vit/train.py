from __future__ import annotations

import time
from typing import Tuple

import torch
import torch.nn as nn


def train_one_epoch(model, loader, optimizer, criterion, device) -> Tuple[float, float, float]:
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    start_time = time.time()

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * labels.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_time = time.time() - start_time
    avg_loss = running_loss / total
    acc = correct / total * 100.0
    return avg_loss, acc, epoch_time


@torch.no_grad()
def evaluate(model, loader, criterion, device) -> Tuple[float, float]:
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)

        running_loss += loss.item() * labels.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    avg_loss = running_loss / total
    acc = correct / total * 100.0
    return avg_loss, acc


def fit(model, train_loader, test_loader, optimizer, criterion, device, epochs: int):

    history = []
    best_test_acc = 0.0
    total_train_time = 0.0

    for epoch in range(1, epochs + 1):
        train_loss, train_acc, t_time = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )
        total_train_time += t_time

        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        best_test_acc = max(best_test_acc, test_acc)

        print(f"Epoch {epoch:02d} | "
              f"Train Loss {train_loss:.3f} Acc {train_acc:.2f}% | "
              f"Test Loss {test_loss:.3f} Acc {test_acc:.2f}% | "
              f"Time {t_time:.1f}s")

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "test_loss": test_loss,
            "test_acc": test_acc,
            "epoch_time": t_time
        })

    return history, best_test_acc, total_train_time