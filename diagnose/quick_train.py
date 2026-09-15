"""EXPLORATORY — backs no reported result in the paper.

Quick trainer for diagnostics TEST 2 and TEST 3.

Usage:
  python diagnose/quick_train.py --model cnn --data data/processed_v2 --epochs 30
  python diagnose/quick_train.py --model mlp --data data/processed    --epochs 30

Reports best val accuracy (best epoch by val_loss) and per-class F1.
No checkpoints saved -- this is a probe, not a baseline.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, TensorDataset

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "train"))
from train_cnn import CNN1D  # current 4-block architecture

RANDOM_STATE = 42
CLASS_NAMES = {0: "background", 1: "Bebop", 2: "AR", 3: "Phantom"}


class MLP(nn.Module):
    def __init__(self, in_dim=2048, num_classes=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 512), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(512, 128), nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def make_loader(X, y, batch_size, shuffle, conv):
    X_t = torch.from_numpy(X).float()
    if conv:
        X_t = X_t.unsqueeze(1)
    ds = TensorDataset(X_t, torch.from_numpy(y).long())
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["cnn", "mlp"], required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    args = ap.parse_args()

    torch.manual_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)
    data_dir = REPO_ROOT / args.data

    X_train = np.load(data_dir / "X_train.npy")
    y_train = np.load(data_dir / "y_train.npy")
    X_val = np.load(data_dir / "X_val.npy")
    y_val = np.load(data_dir / "y_val.npy")
    X_test = np.load(data_dir / "X_test.npy")
    y_test = np.load(data_dir / "y_test.npy")
    print(f"[{args.model}] data={args.data}  train={X_train.shape}  "
          f"val={X_val.shape}  epochs={args.epochs}", flush=True)

    conv = args.model == "cnn"
    train_loader = make_loader(X_train, y_train, args.batch, True, conv)
    val_loader = make_loader(X_val, y_val, args.batch, False, conv)
    test_loader = make_loader(X_test, y_test, args.batch, False, conv)

    model = CNN1D() if conv else MLP()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3)

    best_val_loss = float("inf")
    best_val_acc = 0.0
    best_epoch = 0
    best_state = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        for xb, yb in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

        model.eval()
        vloss = vc = vt = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                logits = model(xb)
                vloss += criterion(logits, yb).item() * xb.size(0)
                vc += (logits.argmax(1) == yb).sum().item()
                vt += xb.size(0)
        vloss /= vt
        vacc = vc / vt
        scheduler.step(vloss)
        print(f"  epoch {epoch:02d}/{args.epochs}  val_loss={vloss:.4f}  "
              f"val_acc={vacc * 100:.2f}%  ({time.time() - t0:.0f}s)", flush=True)
        if vloss < best_val_loss:
            best_val_loss, best_val_acc, best_epoch = vloss, vacc, epoch
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    model.eval()
    preds = []
    with torch.no_grad():
        for xb, _ in test_loader:
            preds.append(model(xb).argmax(1).numpy())
    y_pred = np.concatenate(preds)
    test_acc = (y_pred == y_test).mean()
    per_class = f1_score(y_test, y_pred, average=None)
    macro = f1_score(y_test, y_pred, average="macro")

    print(f"\nRESULT [{args.model} / {args.data}]")
    print(f"  best_epoch={best_epoch}  best_val_acc={best_val_acc * 100:.2f}%")
    print(f"  test_acc={test_acc * 100:.2f}%  macro_F1={macro:.4f}")
    for c, f in enumerate(per_class):
        print(f"  F1[{CLASS_NAMES[c]}] = {f:.4f}")


if __name__ == "__main__":
    main()
