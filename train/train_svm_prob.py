"""SVM baseline (RBF, C=10) retrained WITH probability=True, to obtain
properly Platt-scaled probability estimates for comparison against the
pseudo-logit (decision_function + softmax) approach used in
results/baseline_svm/.

Why this exists: train/train_svm.py's SVC is fit WITHOUT probability=True, so
eval/calibration.py has to treat its decision_function output (signed
distances to hyperplanes, arbitrary scale) as pseudo-logits and softmax them.
That conversion is a plausible source of the reported 23% clean ECE, not
necessarily a real property of the SVM's confidence. This script fits the
same model WITH probability=True so we can compare directly.

sklearn's probability=True wraps an internal 5-fold cross-validation (Platt
scaling) around the SVC fit, so this is substantially slower than the
original fit (~2-6 min) -- allow up to 45 minutes.

Same kernel (RBF), C=10, gamma='scale', random_state=42, and full training
split as the original baseline_svm (selected_variant="rbf_full_C10", see
results/baseline_svm/metrics.json), so the two models are otherwise directly
comparable -- this is NOT a fresh grid search, just the winning configuration
refit with probability=True.

Saves to results/baseline_svm_prob/ (results/baseline_svm/ is left untouched):
  - model.joblib
  - test_predictions.npy, test_labels.npy
  - test_scores.npy   (decision_function output, same as the original baseline)
  - test_proba.npy    (predict_proba output, shape [n_test, 4] -- Platt-scaled)
  - val_proba.npy     (predict_proba on the validation split, needed for
                        fitting a temperature on top of the Platt probabilities
                        in eval/calibration.py)
  - metrics.json      (same schema as the other baselines' metrics.json)

Run with: python train\\train_svm_prob.py   (after activating venv)
Do NOT run this automatically -- the user runs it manually.
"""
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.svm import SVC

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import load_splits, RESULTS_DIR

RANDOM_STATE = 42
C = 10
GAMMA = "scale"
CACHE_SIZE_MB = 2000

OUT_DIR = RESULTS_DIR / "baseline_svm_prob"


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    X_train, X_val, X_test, y_train, y_val, y_test = load_splits()
    print(f"Loaded splits: train={X_train.shape}, val={X_val.shape}, test={X_test.shape}")

    print(f"Fitting SVC(kernel=rbf, C={C}, gamma={GAMMA}, probability=True) on full "
          f"training data ({X_train.shape[0]} samples). probability=True wraps an "
          f"internal 5-fold CV for Platt scaling -- this can take up to ~45 min...")
    model = SVC(kernel="rbf", C=C, gamma=GAMMA, probability=True,
                cache_size=CACHE_SIZE_MB, decision_function_shape="ovr",
                random_state=RANDOM_STATE)
    t0 = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - t0
    print(f"Fit complete in {train_time:.1f}s ({train_time / 60:.1f} min)")

    y_pred = model.predict(X_test)
    test_scores = model.decision_function(X_test)  # (n_test, 4) -- same as original baseline
    test_proba = model.predict_proba(X_test)        # (n_test, 4) -- Platt-scaled
    val_proba = model.predict_proba(X_val)           # (n_val, 4) -- for temperature fitting

    print(f"Test decision_function shape: {test_scores.shape}")
    print(f"Test predict_proba shape:     {test_proba.shape}")
    print(f"Val predict_proba shape:      {val_proba.shape}")

    joblib.dump(model, OUT_DIR / "model.joblib")
    np.save(OUT_DIR / "test_predictions.npy", y_pred)
    np.save(OUT_DIR / "test_labels.npy", y_test)
    np.save(OUT_DIR / "test_scores.npy", test_scores)
    np.save(OUT_DIR / "test_proba.npy", test_proba)
    np.save(OUT_DIR / "val_proba.npy", val_proba)

    accuracy = accuracy_score(y_test, y_pred)
    per_class_f1 = f1_score(y_test, y_pred, average=None).tolist()
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    cm = confusion_matrix(y_test, y_pred).tolist()

    metrics = {
        "model_name": "svm_prob",
        "accuracy": accuracy,
        "per_class_f1": per_class_f1,
        "macro_f1": macro_f1,
        "confusion_matrix": cm,
        "train_time_seconds": train_time,
        "C": C,
        "gamma": GAMMA,
        "probability": True,
        "kernel": "rbf",
        "random_state": RANDOM_STATE,
        "note": "Same config as baseline_svm's selected_variant=rbf_full_C10, "
                "refit with probability=True for Platt-scaled predict_proba output.",
    }
    with open(OUT_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"MODEL svm_prob: train_time={train_time:.1f}s, "
          f"test_accuracy={accuracy * 100:.2f}%, macro_F1={macro_f1:.4f}")
    print(f"Saved model + outputs to {OUT_DIR}")


if __name__ == "__main__":
    main()
