from __future__ import annotations

from copy import deepcopy
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn


def _run_epoch(model, loader, device, optimizer=None) -> Tuple[float, float]:
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_correct = 0
    total_count = 0

    with torch.set_grad_enabled(is_train):
        for batch in loader:
            xb, yb = batch[0], batch[1]
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            if yb.ndim > 1:
                yb = yb.argmax(dim=1)

            logits = model(xb)
            loss = criterion(logits, yb)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * xb.size(0)
            total_correct += (logits.argmax(dim=1) == yb).sum().item()
            total_count += xb.size(0)

    return total_loss / total_count, total_correct / total_count


def fit(model, train_loader, val_loader, optimizer, device, epochs: int = 10, early_stopping_patience: int = 5):
    history: Dict[str, List[float]] = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    best_state = deepcopy(model.state_dict())
    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = _run_epoch(model, train_loader, device, optimizer=optimizer)
        val_loss, val_acc = _run_epoch(model, val_loader, device, optimizer=None)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(
            f"Epoch {epoch:02d}/{epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if early_stopping_patience > 0 and patience_counter >= early_stopping_patience:
            print(f"Early stopping at epoch {epoch:02d}; best validation loss was at epoch {best_epoch:02d}.")
            break

    model.load_state_dict(best_state)
    history["best_epoch"] = best_epoch
    history["best_val_loss"] = None if best_epoch == 0 else best_val_loss
    return model, history


def evaluate(model, loader, device):
    loss, acc = _run_epoch(model, loader, device, optimizer=None)
    return {"loss": loss, "acc": acc}


def predict(model, loader, device):
    model.eval()
    y_true, y_pred, snrs = [], [], []

    with torch.no_grad():
        for batch in loader:
            xb, yb = batch[0], batch[1]
            snr = batch[2] if len(batch) > 2 else torch.zeros_like(yb)
            xb = xb.to(device, non_blocking=True)
            logits = model(xb)

            y_true.append(yb.cpu().numpy())
            y_pred.append(logits.argmax(dim=1).cpu().numpy())
            snrs.append(snr.cpu().numpy())

    return np.concatenate(y_true), np.concatenate(y_pred), np.concatenate(snrs)
