"""EXPLORATORY — backs no reported result in the paper.

TEST 2 - preprocessing variant: StandardScaler ONLY (no log10 transform).

Identical splits (same seed) to load_and_split.py, but the log10 step is
removed so we can isolate whether the log transform is what caps model
accuracy. Writes to data/processed_v2/ and does NOT touch data/processed/.
"""
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = REPO_ROOT / "data" / "dronerf_preprocessed" / "RF_Data.csv"
OUT_DIR = REPO_ROOT / "data" / "processed_v2"

RANDOM_STATE = 42
TEST_FRAC = 0.15
VAL_FRAC = 0.15


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    df = pd.read_csv(CSV_PATH, header=None, dtype=np.float32)
    print(f"Loaded CSV: shape={df.shape}, load_time={time.time() - t0:.2f}s")

    data = df.to_numpy(dtype=np.float32)
    del df
    X = data[0:2048, :].T
    y = data[2049, :].astype(int)
    print(f"X shape: {X.shape}, y shape: {y.shape}")

    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=TEST_FRAC, random_state=RANDOM_STATE, stratify=y
    )
    val_frac_of_remaining = VAL_FRAC / (1 - TEST_FRAC)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=val_frac_of_remaining,
        random_state=RANDOM_STATE, stratify=y_train_full
    )

    print("(a) Raw (X_train): "
          f"mean={X_train.mean():.6e}, std={X_train.std():.6e}, "
          f"min={X_train.min():.6e}, max={X_train.max():.6e}")

    # NO log10 here -- StandardScaler only.
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_val = scaler.transform(X_val).astype(np.float32)
    X_test = scaler.transform(X_test).astype(np.float32)

    print("(b) After StandardScaler only (X_train): "
          f"mean={X_train.mean():.6f}, std={X_train.std():.6f}, "
          f"min={X_train.min():.4f}, max={X_train.max():.4f}")

    joblib.dump(scaler, OUT_DIR / "scaler.joblib")
    splits = {"train": (X_train, y_train), "val": (X_val, y_val), "test": (X_test, y_test)}
    for name, (Xs, ys) in splits.items():
        np.save(OUT_DIR / f"X_{name}.npy", Xs.astype(np.float32))
        np.save(OUT_DIR / f"y_{name}.npy", ys.astype(int))

    metadata = {
        "variant": "standardscaler_only_no_log",
        "total_samples": int(X.shape[0]),
        "feature_dim": int(X.shape[1]),
        "per_split_sample_count": {n: int(Xs.shape[0]) for n, (Xs, _) in splits.items()},
        "random_seed": RANDOM_STATE,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    with open(OUT_DIR / "split_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print("\nFiles written to", OUT_DIR)
    for name, (Xs, ys) in splits.items():
        print(f"  {name}: X={Xs.shape}, y={ys.shape}")


if __name__ == "__main__":
    main()
