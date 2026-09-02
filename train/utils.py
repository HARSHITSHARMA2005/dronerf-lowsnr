"""Shared helpers for baseline model training scripts."""
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score

REPO_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
RESULTS_DIR = REPO_ROOT / "results"


def load_splits():
    X_train = np.load(PROCESSED_DIR / "X_train.npy")
    X_val = np.load(PROCESSED_DIR / "X_val.npy")
    X_test = np.load(PROCESSED_DIR / "X_test.npy")
    y_train = np.load(PROCESSED_DIR / "y_train.npy")
    y_val = np.load(PROCESSED_DIR / "y_val.npy")
    y_test = np.load(PROCESSED_DIR / "y_test.npy")
    return X_train, X_val, X_test, y_train, y_val, y_test


def save_results(model_name, y_test, y_pred, logits_or_scores, train_time):
    out_dir = RESULTS_DIR / f"baseline_{model_name}"
    out_dir.mkdir(parents=True, exist_ok=True)

    y_test = np.asarray(y_test)
    y_pred = np.asarray(y_pred)
    logits_or_scores = np.asarray(logits_or_scores)

    np.save(out_dir / "test_predictions.npy", y_pred)
    is_svm = model_name.lower() == "svm"
    scores_name = "test_scores.npy" if is_svm else "test_logits.npy"
    np.save(out_dir / scores_name, logits_or_scores)
    np.save(out_dir / "test_labels.npy", y_test)

    accuracy = accuracy_score(y_test, y_pred)
    per_class_f1 = f1_score(y_test, y_pred, average=None).tolist()
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    cm = confusion_matrix(y_test, y_pred).tolist()

    metrics = {
        "model_name": model_name,
        "accuracy": accuracy,
        "per_class_f1": per_class_f1,
        "macro_f1": macro_f1,
        "confusion_matrix": cm,
        "train_time_seconds": train_time,
    }

    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"MODEL {model_name}: train_time={train_time:.1f} s, "
          f"test_accuracy={accuracy * 100:.2f}%, macro_F1={macro_f1:.4f}")

    return metrics
