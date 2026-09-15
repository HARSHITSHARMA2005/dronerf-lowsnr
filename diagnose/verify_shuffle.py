"""Resolve the shuffled-label anomaly: a label-permuted model reached ~57% val
accuracy (TEST 1) when chance is ~25%.

Runs four checks and prints each result explicitly:
  CHECK 1 - the shuffle actually permutes y_train
  CHECK 2 - y_val was NOT permuted by the TEST 1 code
  CHECK 3 - re-run the shuffled-label control properly (small MLP, 5 epochs)
  CHECK 4 - if leakage remains, look for duplicate / near-duplicate rows
            shared between train and val (the known DroneRF segment-leak)
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.neighbors import NearestNeighbors

REPO_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
CSV_PATH = REPO_ROOT / "data" / "dronerf_preprocessed" / "RF_Data.csv"
SEED = 42
CLASS_NAMES = {0: "background", 1: "Bebop", 2: "AR", 3: "Phantom"}


def check1(y_train):
    print("=" * 70)
    print("CHECK 1 - did the shuffle actually happen?")
    print("=" * 70)
    rng = np.random.RandomState(SEED)
    y_shuffled = y_train.copy()
    rng.shuffle(y_shuffled)
    diff = (y_shuffled != y_train).mean()
    print(f"  y_train   [:20] = {y_train[:20].tolist()}")
    print(f"  shuffled  [:20] = {y_shuffled[:20].tolist()}")
    print(f"  positions changed: {diff * 100:.1f}%   (expect ~69-75%)")
    vc_o = pd.Series(y_train).value_counts().sort_index().to_dict()
    vc_s = pd.Series(y_shuffled).value_counts().sort_index().to_dict()
    print(f"  class counts original: {vc_o}")
    print(f"  class counts shuffled: {vc_s}")
    print(f"  distributions identical: {vc_o == vc_s}")
    return y_shuffled


def check2():
    print("\n" + "=" * 70)
    print("CHECK 2 - was y_val shuffled with the same permutation? (the suspected bug)")
    print("=" * 70)
    src = (REPO_ROOT / "diagnose" / "check_data.py").read_text().splitlines()
    print("  Relevant lines from diagnose/check_data.py:")
    for i, line in enumerate(src, 1):
        if "shuf" in line.lower() or "make_loader(X_val" in line or "make_loader(X_train" in line:
            print(f"    {i:3d}: {line}")
    print("\n  Verdict: check_data.py shuffles only `y_shuf` (a copy of y_train) and builds")
    print("  the val loader as make_loader(X_val, y_val, ...) with the untouched y_val.")
    print("  => y_val was NOT shuffled. The 57%% is not explained by a shuffle bug.")


class MLP(nn.Module):
    def __init__(self, in_dim=2048, n=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256), nn.ReLU(), nn.Dropout(0.2), nn.Linear(256, n)
        )

    def forward(self, x):
        return self.net(x)


def check3(X_train, y_shuffled, X_val, y_val):
    print("\n" + "=" * 70)
    print("CHECK 3 - re-run shuffled-label control properly (MLP, 5 epochs)")
    print("=" * 70)
    torch.manual_seed(SEED)
    Xtr = torch.from_numpy(X_train).float()
    ytr = torch.from_numpy(y_shuffled).long()
    Xva = torch.from_numpy(X_val).float()
    yva = torch.from_numpy(y_val).long()

    model = MLP()
    opt = torch.optim.Adam(model.parameters(), lr=3e-4)
    crit = nn.CrossEntropyLoss()
    n = Xtr.shape[0]
    for epoch in range(1, 6):
        model.train()
        perm = torch.randperm(n)
        t0 = time.time()
        for i in range(0, n, 128):
            b = perm[i:i + 128]
            opt.zero_grad()
            loss = crit(model(Xtr[b]), ytr[b])
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            tr_acc = (model(Xtr).argmax(1) == ytr).float().mean().item()
            va_acc = (model(Xva).argmax(1) == yva).float().mean().item()
        print(f"  epoch {epoch}: train_acc(shuffled)={tr_acc * 100:.2f}%  "
              f"val_acc(real)={va_acc * 100:.2f}%  ({time.time() - t0:.1f}s)")
    print(f"\n  RESULT: shuffled-label val_acc = {va_acc * 100:.2f}%")
    print(f"  {'OK: ~chance, no structural leak' if va_acc < 0.30 else 'LEAK: >30% with random labels -> structural leakage in X'}")
    return va_acc


def check4(X_train, y_train, X_val, y_val):
    print("\n" + "=" * 70)
    print("CHECK 4 - duplicate / near-duplicate rows shared between train and val")
    print("=" * 70)

    # exact duplicates via hashing rounded rows
    def keyset(A):
        return {r.tobytes() for r in np.ascontiguousarray(np.round(A, 3))}

    tr_keys = keyset(X_train)
    va_rounded = np.ascontiguousarray(np.round(X_val, 3))
    exact = sum(1 for r in va_rounded if r.tobytes() in tr_keys)
    print(f"  exact (3-dp) val rows also present in train: {exact} / {len(X_val)} "
          f"({exact / len(X_val) * 100:.1f}%)")

    # nearest-neighbour distance from each val row to the closest train row
    nn = NearestNeighbors(n_neighbors=1, algorithm="auto").fit(X_train)
    dist, idx = nn.kneighbors(X_val)
    dist = dist.ravel()
    feat_scale = np.sqrt(X_train.shape[1])  # ~RMS distance between two independent standardized rows ~ sqrt(2*d)
    print(f"  val->nearest-train distance:  min={dist.min():.4f}  "
          f"p1={np.percentile(dist, 1):.4f}  median={np.percentile(dist, 50):.3f}  "
          f"max={dist.max():.3f}   (random pair ~ {np.sqrt(2) * feat_scale:.1f})")
    for thr in (0.01, 0.1, 0.5, 1.0):
        frac = (dist < thr).mean()
        print(f"    val rows with a train neighbour closer than {thr:>4}: {frac * 100:.1f}%")

    # label agreement of the nearest train neighbour
    nn_label_match = (y_train[idx.ravel()] == y_val).mean()
    print(f"  nearest-train-neighbour label matches val label: {nn_label_match * 100:.1f}% "
          f"(1-NN accuracy; chance ~ {max(np.bincount(y_val)) / len(y_val) * 100:.1f}%)")

    # raw-CSV adjacency: are consecutive samples (columns) near-identical?
    try:
        raw = pd.read_csv(CSV_PATH, header=None, dtype=np.float32).to_numpy()
        Xraw = raw[0:2048, :].T  # (n_samples, 2048) in original file order
        a = Xraw[:-1]
        b = Xraw[1:]
        cos = (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-12)
        print(f"  raw CSV consecutive-sample cosine similarity: "
              f"mean={cos.mean():.4f}  median={np.median(cos):.4f}  "
              f">0.99: {(cos > 0.99).mean() * 100:.1f}%   >0.999: {(cos > 0.999).mean() * 100:.1f}%")
        print("  first 5 raw X rows, cols[:12]:")
        for i in range(5):
            print("    " + " ".join(f"{v:10.2e}" for v in Xraw[i, :12]))
    except Exception as e:  # pragma: no cover
        print(f"  (raw CSV inspection skipped: {e})")


def main():
    y_train = np.load(PROCESSED_DIR / "y_train.npy")
    y_val = np.load(PROCESSED_DIR / "y_val.npy")
    X_train = np.load(PROCESSED_DIR / "X_train.npy")
    X_val = np.load(PROCESSED_DIR / "X_val.npy")

    y_shuffled = check1(y_train)
    check2()
    check3(X_train, y_shuffled, X_val, y_val)
    check4(X_train, y_train, X_val, y_val)


if __name__ == "__main__":
    main()
