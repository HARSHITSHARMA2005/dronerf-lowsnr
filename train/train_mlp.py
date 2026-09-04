"""MLP baseline for DroneRF 4-class classification (PyTorch, CPU).

Architecture (calibration-study baseline): a plain 3-layer MLP on the raw
2048-bin log-spectrum vector (no reshape). Linear(2048, 512) -> ReLU ->
Dropout(0.2) -> Linear(512, 128) -> ReLU -> Dropout(0.2) -> Linear(128, 4).
Training config mirrors train_cnn.py exactly (Adam lr=3e-4, batch 128, 50
epochs, early stopping patience=10 on val_loss, ReduceLROnPlateau
factor=0.5 patience=4). Resumable checkpointing protects against Windows
killing the process mid-run.
"""
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import load_splits, save_results, RESULTS_DIR

RANDOM_STATE = 42
BATCH_SIZE = 128
EPOCHS = 50
LR = 3e-4
PATIENCE = 10
NUM_CLASSES = 4
MAX_TRAIN_SECONDS = 10800  # abort guard: 3 hours
DROPOUT = 0.2
CLASS_NAMES = {0: "background", 1: "Bebop", 2: "AR", 3: "Phantom"}

OUT_DIR = RESULTS_DIR / "baseline_mlp"

torch.manual_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)


class MLP(nn.Module):
    def __init__(self, in_dim=2048, num_classes=NUM_CLASSES, dropout=DROPOUT):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def make_loader(X, y, batch_size, shuffle):
    X_t = torch.from_numpy(X).float()
    y_t = torch.from_numpy(y).long()
    return DataLoader(TensorDataset(X_t, y_t), batch_size=batch_size, shuffle=shuffle)


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = correct = total = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            total_loss += criterion(logits, yb).item() * xb.size(0)
            correct += (logits.argmax(1) == yb).sum().item()
            total += xb.size(0)
    return total_loss / total, correct / total


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu")
    X_train, X_val, X_test, y_train, y_val, y_test = load_splits()
    print(f"Loaded splits: train={X_train.shape}, val={X_val.shape}, test={X_test.shape}", flush=True)

    train_loader = make_loader(X_train, y_train, BATCH_SIZE, True)
    val_loader = make_loader(X_val, y_val, BATCH_SIZE, False)
    test_loader = make_loader(X_test, y_test, BATCH_SIZE, False)

    model = MLP().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=4)

    best_val_loss = float("inf")
    best_val_acc = 0.0
    best_epoch = 0
    best_state = None
    epochs_no_improve = 0
    start_epoch = 1
    prev_elapsed = 0.0

    ckpt_path = OUT_DIR / "resume_ckpt.pt"
    if ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        scheduler.load_state_dict(ck["scheduler"])
        best_val_loss = ck["best_val_loss"]
        best_val_acc = ck["best_val_acc"]
        best_epoch = ck["best_epoch"]
        best_state = ck["best_state"]
        epochs_no_improve = ck["epochs_no_improve"]
        start_epoch = ck["epoch"] + 1
        prev_elapsed = ck["elapsed"]
        print(f"Resumed from epoch {ck['epoch']} (best_epoch={best_epoch}, "
              f"best_val_loss={best_val_loss:.4f})", flush=True)

    log_mode = "a" if ckpt_path.exists() else "w"
    log_fh = open(OUT_DIR / "training_log.csv", log_mode, newline="")
    writer = csv.writer(log_fh)
    if log_mode == "w":
        writer.writerow(["epoch", "train_loss", "val_loss", "val_acc", "lr"])

    t0 = time.time()
    for epoch in range(start_epoch, EPOCHS + 1):
        model.train()
        running_loss = n = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * xb.size(0)
            n += xb.size(0)
        train_loss = running_loss / n

        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_loss)
        lr_now = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch:03d}/{EPOCHS} | train_loss={train_loss:.4f} | "
              f"val_loss={val_loss:.4f} | val_acc={val_acc * 100:.2f}% | lr={lr_now:.2e}", flush=True)
        writer.writerow([epoch, f"{train_loss:.6f}", f"{val_loss:.6f}", f"{val_acc:.6f}", f"{lr_now:.2e}"])
        log_fh.flush()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            best_epoch = epoch
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
            stop = False
        else:
            epochs_no_improve += 1
            stop = epochs_no_improve >= PATIENCE

        elapsed = prev_elapsed + (time.time() - t0)
        torch.save({
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_val_loss": best_val_loss,
            "best_val_acc": best_val_acc,
            "best_epoch": best_epoch,
            "best_state": best_state,
            "epochs_no_improve": epochs_no_improve,
            "elapsed": elapsed,
        }, ckpt_path)

        if stop:
            print(f"Early stopping at epoch {epoch}.", flush=True)
            break

        if elapsed > MAX_TRAIN_SECONDS:
            print(f"ABORT: training exceeded {MAX_TRAIN_SECONDS}s.", flush=True)
            break

    train_time = prev_elapsed + (time.time() - t0)
    log_fh.close()

    if best_state is not None:
        model.load_state_dict(best_state)
    print(f"Best epoch: {best_epoch} | best val_loss={best_val_loss:.4f} | "
          f"best val_acc={best_val_acc * 100:.2f}%", flush=True)

    model.eval()
    all_logits = []
    with torch.no_grad():
        for xb, _ in test_loader:
            all_logits.append(model(xb.to(device)).numpy())
    test_logits = np.concatenate(all_logits, axis=0)
    y_pred = test_logits.argmax(axis=1)

    torch.save(model.state_dict(), OUT_DIR / "model.pt")
    # save_results writes test_logits.npy / test_predictions.npy / test_labels.npy
    base = save_results("mlp", y_test, y_pred, test_logits, train_time)

    metrics = {
        "test_accuracy": base["accuracy"],
        "macro_F1": base["macro_f1"],
        "per_class_F1": base["per_class_f1"],
        "confusion_matrix": base["confusion_matrix"],
        "train_time_seconds": train_time,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "best_val_accuracy": best_val_acc,
    }
    with open(OUT_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    for cls, f1 in enumerate(metrics["per_class_F1"]):
        print(f"  F1[{CLASS_NAMES[cls]}] = {f1:.4f}", flush=True)

    if ckpt_path.exists():
        ckpt_path.unlink()


if __name__ == "__main__":
    main()
