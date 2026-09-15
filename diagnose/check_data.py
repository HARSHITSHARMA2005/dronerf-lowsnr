"""TEST 1 - data integrity checks for the DroneRF baseline pipeline.

Checks:
  1. X_train / y_train row-count agreement.
  2. Class distribution vs. the expected train split.
  3. Spot-check 10 random samples: feature values + label, and global mean/std.
  4. Shuffled-label control: train the current 4-block CNN for 3 epochs on
     randomly permuted labels. Val accuracy must collapse to ~25% (chance for
     4 classes). Anything > 30% implies a data leak (e.g. row misalignment,
     duplicated samples across splits, or a label baked into the features).
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "train"))
from train_cnn import CNN1D, make_loader  # current 4-block architecture

PROCESSED_DIR = REPO_ROOT / "data" / "processed"
RANDOM_STATE = 42
EXPECTED_TRAIN = {0: 2870, 1: 5880, 2: 5670, 3: 1470}
CLASS_NAMES = {0: "background", 1: "Bebop", 2: "AR", 3: "Phantom"}


def main():
    rng = np.random.default_rng(RANDOM_STATE)
    torch.manual_seed(RANDOM_STATE)

    X_train = np.load(PROCESSED_DIR / "X_train.npy")
    y_train = np.load(PROCESSED_DIR / "y_train.npy")
    X_val = np.load(PROCESSED_DIR / "X_val.npy")
    y_val = np.load(PROCESSED_DIR / "y_val.npy")

    print("=== TEST 1: DATA INTEGRITY ===")
    print(f"X_train shape: {X_train.shape}   y_train shape: {y_train.shape}")
    assert X_train.shape[0] == y_train.shape[0], "row count mismatch X_train vs y_train"
    print(f"Row counts agree: {X_train.shape[0]} == {y_train.shape[0]}  [OK]")

    vc = pd.Series(y_train).value_counts().sort_index()
    print("\ny_train class distribution:")
    for label, count in vc.items():
        exp = EXPECTED_TRAIN.get(int(label), "?")
        print(f"  {int(label)} ({CLASS_NAMES.get(int(label), '?'):10s}): {count:5d}   expected ~{exp}")

    print(f"\nGlobal X_train mean={X_train.mean():.5f}  std={X_train.std():.5f}  "
          f"min={X_train.min():.3f}  max={X_train.max():.3f}")

    print("\n10 random samples (first 10 feature values + label):")
    idx = rng.choice(X_train.shape[0], size=10, replace=False)
    for i in idx:
        vals = " ".join(f"{v:+.3f}" for v in X_train[i, :10])
        print(f"  i={i:5d}  y={y_train[i]} ({CLASS_NAMES[int(y_train[i])]:10s})  x[:10]= {vals}")

    # --- shuffled-label control -------------------------------------------
    print("\n=== Shuffled-label control (3 epochs, current 4-block CNN) ===")
    y_shuf = y_train.copy()
    rng.shuffle(y_shuf)
    frac_moved = (y_shuf != y_train).mean()
    print(f"Fraction of labels moved by shuffle: {frac_moved:.3f}")

    device = torch.device("cpu")
    train_loader = make_loader(X_train, y_shuf, 128, True)
    val_loader = make_loader(X_val, y_val, 128, False)  # real val labels

    model = CNN1D().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

    for epoch in range(1, 4):
        model.train()
        t0 = time.time()
        for xb, yb in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

        model.eval()
        correct = total = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                correct += (model(xb).argmax(1) == yb).sum().item()
                total += xb.size(0)
        # train accuracy against the (shuffled) labels the model was fed
        # (2000-sample slice, just to see whether the net is memorising)
        with torch.no_grad():
            xb = torch.from_numpy(X_train[:2000]).float().unsqueeze(1)
            tr_pred = model(xb).argmax(1).numpy()
        tr_acc = (tr_pred == y_shuf[:2000]).mean() * 100
        print(f"  epoch {epoch}: train_acc(shuffled,2k)={tr_acc:.2f}%  "
              f"val_acc(real)={correct / total * 100:.2f}%  ({time.time() - t0:.0f}s)")

    val_acc = correct / total * 100
    print(f"\nSHUFFLED-LABEL RESULT: val_acc={val_acc:.2f}%  "
          f"({'OK (~chance, no leak)' if val_acc < 30 else 'WARNING: >30% -> possible data leak'})")


if __name__ == "__main__":
    main()
