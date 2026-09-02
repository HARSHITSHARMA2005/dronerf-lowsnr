"""SVM baseline (RBF kernel) for DroneRF 4-class classification."""
import sys
import time
import multiprocessing as mp
from pathlib import Path

import joblib
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import load_splits, save_results, RESULTS_DIR

RANDOM_STATE = 42
FULL_FIT_TIMEOUT_SECONDS = 600  # 10 minutes
SUBSET_SIZE = 5000


def _fit_worker(X, y, queue):
    try:
        model = SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE)
        t0 = time.time()
        model.fit(X, y)
        elapsed = time.time() - t0
        queue.put(("ok", model, elapsed))
    except Exception as e:
        queue.put(("error", str(e), 0.0))


def fit_with_timeout(X, y, timeout):
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    proc = ctx.Process(target=_fit_worker, args=(X, y, queue))
    proc.start()
    proc.join(timeout)
    if proc.is_alive():
        proc.terminate()
        proc.join()
        return None, None
    status, payload, elapsed = queue.get()
    if status == "error":
        raise RuntimeError(f"SVM fit failed: {payload}")
    return payload, elapsed


def main():
    X_train, X_val, X_test, y_train, y_val, y_test = load_splits()
    print(f"Loaded splits: train={X_train.shape}, val={X_val.shape}, test={X_test.shape}")

    print(f"Attempting full-data SVM fit on {X_train.shape[0]} samples "
          f"(timeout={FULL_FIT_TIMEOUT_SECONDS}s)...")
    model, train_time = fit_with_timeout(X_train, y_train, FULL_FIT_TIMEOUT_SECONDS)

    if model is None:
        print(f"Full fit exceeded {FULL_FIT_TIMEOUT_SECONDS}s — aborting and "
              f"falling back to a stratified subset of {SUBSET_SIZE} samples.")
        X_sub, _, y_sub, _ = train_test_split(
            X_train, y_train, train_size=SUBSET_SIZE,
            random_state=RANDOM_STATE, stratify=y_train
        )
        t0 = time.time()
        model = SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE)
        model.fit(X_sub, y_sub)
        train_time = time.time() - t0
        print(f"Subset fit complete in {train_time:.1f}s on {X_sub.shape[0]} samples.")
    else:
        print(f"Full fit complete in {train_time:.1f}s.")

    y_pred = model.predict(X_test)
    test_scores = model.decision_function(X_test)  # shape (n_test, 4) for one-vs-one/rest
    print(f"Test decision_function shape: {test_scores.shape}")

    val_pred = model.predict(X_val)
    val_acc = (val_pred == y_val).mean()
    print(f"Validation accuracy: {val_acc * 100:.2f}%")

    out_dir = RESULTS_DIR / "baseline_svm"
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out_dir / "model.joblib")

    save_results("svm", y_test, y_pred, test_scores, train_time)


if __name__ == "__main__":
    main()
