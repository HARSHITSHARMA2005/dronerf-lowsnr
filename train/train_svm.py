"""SVM baseline (RBF kernel) for DroneRF 4-class classification.

Strategy:
  1. Try a full-data RBF fit (C=10) with a large kernel cache and a 45-min timeout.
  2. If that times out, grid-search C in [1, 10, 100] (gamma='scale', RBF) on a
     stratified 5000-sample subset and keep the best-on-validation model.
  3. Always also fit a LinearSVC on the full data as a fast fallback.
  4. Pick whichever variant has the best validation accuracy.
Regardless of which variant wins, save decision_function outputs on the test set
(shape [n_test, 4]) as test_scores.npy via save_results().
"""
import sys
import time
import multiprocessing as mp
from pathlib import Path

import joblib
from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import load_splits, save_results, RESULTS_DIR

RANDOM_STATE = 42
FULL_FIT_TIMEOUT_SECONDS = 2700  # 45 minutes
SUBSET_SIZE = 5000
CACHE_SIZE_MB = 2000
C_GRID = [10, 100, 1000]

OUT_DIR = RESULTS_DIR / "baseline_svm"
LOG_PATH = OUT_DIR / "train_log.txt"
_log_fh = None


def log(msg):
    print(msg, flush=True)
    if _log_fh is not None:
        _log_fh.write(str(msg) + "\n")
        _log_fh.flush()


def _fit_worker(C, gamma, X, y, model_path, queue):
    # The trained model is written to disk (never passed through the Queue —
    # a large object in a mp.Queue can deadlock parent/child on the pipe).
    try:
        model = SVC(kernel="rbf", C=C, gamma=gamma, cache_size=CACHE_SIZE_MB,
                    decision_function_shape="ovr", random_state=RANDOM_STATE)
        t0 = time.time()
        model.fit(X, y)
        elapsed = time.time() - t0
        joblib.dump(model, model_path)
        queue.put(("ok", elapsed))
    except Exception as e:  # pragma: no cover
        queue.put(("error", str(e)))


def fit_rbf_with_timeout(C, gamma, X, y, timeout):
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    model_path = OUT_DIR / f"_rbf_C{C}.joblib"
    proc = ctx.Process(target=_fit_worker, args=(C, gamma, X, y, str(model_path), queue))
    proc.start()
    proc.join(timeout)
    if proc.is_alive():
        proc.terminate()
        proc.join()
        return None, None
    status, payload = queue.get()
    if status == "error":
        raise RuntimeError(f"SVM fit failed: {payload}")
    model = joblib.load(model_path)
    model_path.unlink(missing_ok=True)
    return model, payload


def evaluate(model, X, y):
    return float((model.predict(X) == y).mean())


def main():
    global _log_fh
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    _log_fh = open(LOG_PATH, "w")

    X_train, X_val, X_test, y_train, y_val, y_test = load_splits()
    log(f"Loaded splits: train={X_train.shape}, val={X_val.shape}, test={X_test.shape}")

    candidates = []  # (name, model, train_time, val_acc)

    # --- 1. Full-data RBF grid search over C (gamma='scale') -----------------
    # Full-data RBF fits take ~2-6 min each here, so we grid-search C directly
    # on all 15890 samples. Only if a fit genuinely exceeds the timeout do we
    # drop to the stratified subset for the remaining C values.
    log(f"Grid-searching full-data RBF SVM over C={C_GRID} on {X_train.shape[0]} samples "
        f"(gamma=scale, cache={CACHE_SIZE_MB}MB, per-fit timeout={FULL_FIT_TIMEOUT_SECONDS}s)...")
    X_sub = y_sub = None
    for C in C_GRID:
        model, ttime = fit_rbf_with_timeout(C, "scale", X_train, y_train, FULL_FIT_TIMEOUT_SECONDS)
        if model is not None:
            va = evaluate(model, X_val, y_val)
            log(f"  full C={C}: fit {ttime:.1f}s | val_acc={va * 100:.2f}%")
            candidates.append((f"rbf_full_C{C}", model, ttime, va))
            continue
        log(f"  full C={C} exceeded {FULL_FIT_TIMEOUT_SECONDS}s — retrying on a "
            f"stratified {SUBSET_SIZE}-sample subset.")
        if X_sub is None:
            X_sub, _, y_sub, _ = train_test_split(
                X_train, y_train, train_size=SUBSET_SIZE,
                random_state=RANDOM_STATE, stratify=y_train,
            )
        m, tt = fit_rbf_with_timeout(C, "scale", X_sub, y_sub, FULL_FIT_TIMEOUT_SECONDS)
        if m is None:
            log(f"  C={C}: timed out even on subset, skipping.")
            continue
        va = evaluate(m, X_val, y_val)
        log(f"  subset C={C}: fit {tt:.1f}s | val_acc={va * 100:.2f}%")
        candidates.append((f"rbf_sub{SUBSET_SIZE}_C{C}", m, tt, va))

    # --- 2. LinearSVC fallback on full data ---------------------------------
    log("Fitting LinearSVC fallback on full data...")
    t0 = time.time()
    lin = LinearSVC(C=1.0, random_state=RANDOM_STATE, max_iter=5000)
    lin.fit(X_train, y_train)
    lt = time.time() - t0
    lva = evaluate(lin, X_val, y_val)
    log(f"  LinearSVC fit {lt:.1f}s | val_acc={lva * 100:.2f}%")
    candidates.append(("linear_svc_full", lin, lt, lva))

    # --- 3. Pick best on validation ---------------------------------------
    name, model, train_time, val_acc = max(candidates, key=lambda c: c[3])
    log(f"\nSelected variant: {name} (val_acc={val_acc * 100:.2f}%, train_time={train_time:.1f}s)")

    y_pred = model.predict(X_test)
    test_scores = model.decision_function(X_test)  # (n_test, 4)
    log(f"Test decision_function shape: {test_scores.shape}")

    joblib.dump(model, OUT_DIR / "model.joblib")
    metrics = save_results("svm", y_test, y_pred, test_scores, train_time)
    metrics["selected_variant"] = name
    metrics["validation_accuracy"] = val_acc
    import json
    with open(OUT_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    _log_fh.close()


if __name__ == "__main__":
    main()
