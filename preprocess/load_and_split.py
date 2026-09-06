"""Load the DroneRF preprocessed CSV and produce a stratified train/val/test split."""
import json
import time
import tracemalloc
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = REPO_ROOT / "data" / "dronerf_preprocessed" / "RF_Data.csv"
OUT_DIR = REPO_ROOT / "data" / "processed"

RANDOM_STATE = 42
TEST_FRAC = 0.15
VAL_FRAC = 0.15  # of total; computed as a fraction of the remaining 85% below

# Expected class-size -> drone-name mapping, keyed by count (not label id).
COUNT_TO_DRONE = {
    4100: "background",
    8400: "Bebop",
    8100: "AR",
    2100: "Phantom",
}


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    tracemalloc.start()
    t0 = time.time()
    df = pd.read_csv(CSV_PATH, header=None, dtype=np.float32)
    load_time = time.time() - t0
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print(f"Loaded CSV: shape={df.shape}, load_time={load_time:.2f}s, "
          f"peak_memory={peak / 1e9:.2f} GB")

    data = df.to_numpy(dtype=np.float32)
    del df

    X = data[0:2048, :].T  # (22700, 2048)
    y = data[2049, :].astype(int)  # 4-class label

    print(f"X shape: {X.shape}, y shape: {y.shape}")

    value_counts = pd.Series(y).value_counts().sort_index()
    print("Class distribution (raw label id -> count):")
    print(value_counts)

    inferred_mapping = {}
    for label_id, count in value_counts.items():
        drone = COUNT_TO_DRONE.get(int(count), "UNKNOWN")
        inferred_mapping[int(label_id)] = drone
    print(f"Inferred label -> drone mapping: {inferred_mapping}")

    t1 = time.time()
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=TEST_FRAC, random_state=RANDOM_STATE, stratify=y
    )
    val_frac_of_remaining = VAL_FRAC / (1 - TEST_FRAC)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=val_frac_of_remaining,
        random_state=RANDOM_STATE, stratify=y_train_full
    )
    split_time = time.time() - t1

    print("\n(a) Raw (X_train):")
    print(f"  mean={X_train.mean():.6f}, std={X_train.std():.6f}, "
          f"min={X_train.min():.6e}, max={X_train.max():.6e}")

    # Keep the pre-log-transform raw power spectra around: the SNR degradation
    # study (eval/snr_degradation.py) injects AWGN on this raw signal, not on
    # the already log10+standardized features, to simulate realistic RF
    # signal degradation rather than artificial post-processing noise.
    X_train_raw = X_train.copy()
    X_val_raw = X_val.copy()
    X_test_raw = X_test.copy()

    LOG_EPSILON = 1e-10
    X_train = np.log10(X_train + LOG_EPSILON)
    X_val = np.log10(X_val + LOG_EPSILON)
    X_test = np.log10(X_test + LOG_EPSILON)

    print("(b) After log10 (X_train):")
    print(f"  mean={X_train.mean():.6f}, std={X_train.std():.6f}, "
          f"min={X_train.min():.6e}, max={X_train.max():.6e}")

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_val = scaler.transform(X_val).astype(np.float32)
    X_test = scaler.transform(X_test).astype(np.float32)

    print("(c) After log10 + standardization (X_train):")
    print(f"  mean={X_train.mean():.6f}, std={X_train.std():.6f}, "
          f"min={X_train.min():.6e}, max={X_train.max():.6e}")

    scaler_path = OUT_DIR / "scaler.joblib"
    joblib.dump(scaler, scaler_path)

    splits = {
        "train": (X_train, y_train),
        "val": (X_val, y_val),
        "test": (X_test, y_test),
    }
    raw_splits = {
        "train": X_train_raw,
        "val": X_val_raw,
        "test": X_test_raw,
    }

    paths = {}
    for name, (Xs, ys) in splits.items():
        x_path = OUT_DIR / f"X_{name}.npy"
        y_path = OUT_DIR / f"y_{name}.npy"
        np.save(x_path, Xs.astype(np.float32))
        np.save(y_path, ys.astype(int))
        paths[f"X_{name}"] = str(x_path)
        paths[f"y_{name}"] = str(y_path)

    for name, Xr in raw_splits.items():
        x_raw_path = OUT_DIR / f"X_{name}_raw.npy"
        np.save(x_raw_path, Xr.astype(np.float32))
        paths[f"X_{name}_raw"] = str(x_raw_path)

    per_split_class_counts = {
        name: {int(k): int(v) for k, v in pd.Series(ys).value_counts().sort_index().items()}
        for name, (_, ys) in splits.items()
    }

    metadata = {
        "total_samples": int(X.shape[0]),
        "feature_dim": int(X.shape[1]),
        "per_split_sample_count": {name: int(Xs.shape[0]) for name, (Xs, _) in splits.items()},
        "per_class_count_per_split": per_split_class_counts,
        "inferred_label_to_drone_mapping": inferred_mapping,
        "random_seed": RANDOM_STATE,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    metadata_path = OUT_DIR / "split_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    paths["split_metadata"] = str(metadata_path)
    paths["scaler"] = str(scaler_path)

    print("\n=== Summary ===")
    print(f"Load time: {load_time:.2f}s | Split time: {split_time:.2f}s | Peak memory: {peak / 1e9:.2f} GB")
    for name, (Xs, ys) in splits.items():
        print(f"{name}: X={Xs.shape}, y={ys.shape}, class_counts={per_split_class_counts[name]}")
    print("\nFiles written:")
    for key, p in paths.items():
        print(f"  {key}: {p}")


if __name__ == "__main__":
    main()
