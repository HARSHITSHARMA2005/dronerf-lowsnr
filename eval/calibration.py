"""Calibration analysis for the three DroneRF baselines (SVM, CNN, MLP).

Core paper contribution: measures whether softmax/decision-function confidence
scores on the clean test set are trustworthy (ECE, MCE, Brier, NLL), draws
reliability diagrams, and applies temperature scaling (Guo et al. 2017) to fix
miscalibration without changing predictions (argmax is temperature-invariant).

SVM handling note: train/train_svm.py trains a plain SVC/LinearSVC WITHOUT
probability=True (see model.decision_function usage there), so there is no
Platt-scaling predict_proba available and none is needed. The saved
test_scores.npy / val "scores" are the raw decision_function output, shape
(n, 4) for the one-vs-rest-shaped multiclass decision function. We treat these
margin scores as pseudo-logits and apply softmax over them, exactly as if they
were logits from a neural net. Temperature scaling on an SVM's decision
function is unconventional in the strict Platt-scaling sense, but it is a
principled, consistent way to compare calibration across all three model
families using one methodology (Guo et al.'s temperature scaling framework is
agnostic to how the "logits" were produced).

Run with: python eval\\calibration.py   (after activating venv)
Do NOT run this automatically — the user runs it manually.
"""
import json
import sys
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "train"))
from utils import load_splits, RESULTS_DIR  # noqa: E402
from train_mlp import MLP  # noqa: E402
from train_cnn import CNN1D  # noqa: E402

OUT_DIR = RESULTS_DIR / "calibration"
N_BINS_ECE = 15
N_BINS_PLOT = 10
CLASS_NAMES = ["background", "Bebop", "AR", "Phantom"]
NUM_CLASSES = 4


# ---------------------------------------------------------------------------
# Core calibration metrics
# ---------------------------------------------------------------------------

def softmax(x, axis=1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def compute_ece_mce(probs, labels, n_bins=N_BINS_ECE):
    """Expected/Maximum Calibration Error with equal-width confidence bins.

    ECE = sum_b (|B_b| / N) * |acc(B_b) - conf(B_b)|
    MCE = max_b |acc(B_b) - conf(B_b)|
    Confidence = max predicted probability; accuracy = correctness of argmax.
    """
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    correctness = (predictions == labels).astype(np.float64)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    n = len(labels)
    ece = 0.0
    mce = 0.0
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if i == n_bins - 1:
            mask = (confidences >= lo) & (confidences <= hi)
        else:
            mask = (confidences >= lo) & (confidences < hi)
        bin_size = mask.sum()
        if bin_size == 0:
            continue
        avg_conf = confidences[mask].mean()
        avg_acc = correctness[mask].mean()
        gap = abs(avg_conf - avg_acc)
        ece += (bin_size / n) * gap
        mce = max(mce, gap)
    return float(ece), float(mce)


def compute_brier(probs, labels, num_classes=NUM_CLASSES):
    one_hot = np.eye(num_classes)[labels]
    return float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))


def compute_nll(probs, labels, eps=1e-12):
    p_true = np.clip(probs[np.arange(len(labels)), labels], eps, 1.0)
    return float(-np.mean(np.log(p_true)))


def compute_all_metrics(probs, labels):
    accuracy = float((probs.argmax(axis=1) == labels).mean())
    ece, mce = compute_ece_mce(probs, labels)
    return {
        "ECE": ece,
        "MCE": mce,
        "Brier": compute_brier(probs, labels),
        "NLL": compute_nll(probs, labels),
        "accuracy": accuracy,
    }


# ---------------------------------------------------------------------------
# Reliability diagrams
# ---------------------------------------------------------------------------

def plot_reliability(probs, labels, ece, model_name, tag, out_dir, n_bins=N_BINS_PLOT):
    """tag is 'uncalibrated' or 'calibrated'."""
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    correctness = (predictions == labels).astype(np.float64)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_width = 1.0 / n_bins

    bin_acc = np.zeros(n_bins)
    bin_count = np.zeros(n_bins, dtype=int)
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if i == n_bins - 1:
            mask = (confidences >= lo) & (confidences <= hi)
        else:
            mask = (confidences >= lo) & (confidences < hi)
        bin_count[i] = mask.sum()
        bin_acc[i] = correctness[mask].mean() if mask.sum() > 0 else 0.0

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(6, 7), gridspec_kw={"height_ratios": [3, 1]}, sharex=True)

    ax1.bar(bin_centers, bin_acc, width=bin_width * 0.9, color="#4C72B0",
            edgecolor="black", linewidth=0.5, label="Accuracy")
    ax1.plot([0, 1], [0, 1], linestyle="--", color="red", linewidth=1.5,
              label="Perfect calibration")
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.set_ylabel("Accuracy")
    ax1.set_title(f"Reliability Diagram — {model_name.upper()} ({tag})")
    ax1.text(0.03, 0.92, f"ECE = {ece:.4f}", transform=ax1.transAxes,
              fontsize=11, verticalalignment="top",
              bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
    ax1.legend(loc="lower right")
    ax1.grid(alpha=0.3)

    ax2.bar(bin_centers, bin_count, width=bin_width * 0.9, color="#55A868",
            edgecolor="black", linewidth=0.5)
    ax2.set_xlabel("Confidence")
    ax2.set_ylabel("# samples")
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    png_path = out_dir / f"reliability_{model_name}_{tag}.png"
    pdf_path = out_dir / f"reliability_{model_name}_{tag}.pdf"
    fig.savefig(png_path, dpi=300)
    fig.savefig(pdf_path)
    plt.close(fig)
    return png_path, pdf_path


# ---------------------------------------------------------------------------
# Temperature scaling
# ---------------------------------------------------------------------------

def fit_temperature(val_logits, val_labels):
    """Optimize scalar T minimizing NLL on validation logits via LBFGS.

    T is reparameterized as softplus(raw) to keep it strictly positive without
    a hard clamp (avoids zero-gradient regions from clamping/abs at 0).
    """
    val_logits_t = torch.from_numpy(val_logits).float()
    val_labels_t = torch.from_numpy(val_labels).long()

    raw_T = torch.nn.Parameter(torch.tensor([1.5]))
    optimizer = torch.optim.LBFGS([raw_T], lr=0.01, max_iter=50)
    criterion = nn.CrossEntropyLoss()

    def closure():
        optimizer.zero_grad()
        T = torch.nn.functional.softplus(raw_T) + 1e-3  # T > 0
        loss = criterion(val_logits_t / T, val_labels_t)
        loss.backward()
        return loss

    optimizer.step(closure)
    T_final = (torch.nn.functional.softplus(raw_T) + 1e-3).item()
    return T_final


# ---------------------------------------------------------------------------
# Per-model pipelines
# ---------------------------------------------------------------------------

def get_val_logits_mlp(X_val):
    device = torch.device("cpu")
    model = MLP().to(device)
    model.load_state_dict(torch.load(RESULTS_DIR / "baseline_mlp" / "model.pt", map_location=device))
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(X_val).float()).numpy()
    return logits


def get_val_logits_cnn(X_val):
    device = torch.device("cpu")
    model = CNN1D().to(device)
    model.load_state_dict(torch.load(RESULTS_DIR / "baseline_cnn" / "model.pt", map_location=device))
    model.eval()
    X_val_r = X_val.reshape(X_val.shape[0], 1, -1)
    with torch.no_grad():
        logits = model(torch.from_numpy(X_val_r).float()).numpy()
    return logits


def get_val_scores_svm(X_val):
    model = joblib.load(RESULTS_DIR / "baseline_svm" / "model.joblib")
    return model.decision_function(X_val)


def run_model(name, test_scores, test_labels, val_scores_fn, X_val, y_val, out_dir):
    print(f"\n{'=' * 60}\n{name.upper()}\n{'=' * 60}")

    # --- Part A: uncalibrated probabilities ---------------------------------
    test_probs_uncal = softmax(test_scores, axis=1)
    uncal_metrics = compute_all_metrics(test_probs_uncal, test_labels)
    print(f"[uncalibrated] ECE={uncal_metrics['ECE']:.4f} MCE={uncal_metrics['MCE']:.4f} "
          f"Brier={uncal_metrics['Brier']:.4f} NLL={uncal_metrics['NLL']:.4f} "
          f"acc={uncal_metrics['accuracy'] * 100:.2f}%")

    plot_reliability(test_probs_uncal, test_labels, uncal_metrics["ECE"], name, "uncalibrated", out_dir)

    # --- Part D: temperature scaling (fit on val, apply to test) -----------
    val_scores = val_scores_fn(X_val)
    T_star = fit_temperature(val_scores, y_val)
    print(f"  T* = {T_star:.4f} ({'overconfident' if T_star > 1 else 'underconfident' if T_star < 1 else 'well-calibrated'})")

    test_probs_cal = softmax(test_scores / T_star, axis=1)

    # --- Part E: calibrated metrics + accuracy-invariance check ------------
    cal_metrics = compute_all_metrics(test_probs_cal, test_labels)
    cal_metrics["temperature"] = T_star
    if abs(cal_metrics["accuracy"] - uncal_metrics["accuracy"]) > 1e-9:
        print(f"  WARNING: accuracy changed after temperature scaling! "
              f"{uncal_metrics['accuracy']} -> {cal_metrics['accuracy']}")
    print(f"[calibrated]   ECE={cal_metrics['ECE']:.4f} MCE={cal_metrics['MCE']:.4f} "
          f"Brier={cal_metrics['Brier']:.4f} NLL={cal_metrics['NLL']:.4f} "
          f"acc={cal_metrics['accuracy'] * 100:.2f}%")

    plot_reliability(test_probs_cal, test_labels, cal_metrics["ECE"], name, "calibrated", out_dir)

    return {"uncalibrated": uncal_metrics, "calibrated": cal_metrics}


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    _, X_val, _, _, y_val, _ = load_splits()

    results = {}

    # SVM
    svm_scores = np.load(RESULTS_DIR / "baseline_svm" / "test_scores.npy")
    svm_labels = np.load(RESULTS_DIR / "baseline_svm" / "test_labels.npy")
    print(f"SVM test_scores shape={svm_scores.shape} dtype={svm_scores.dtype}")
    results["svm"] = run_model("svm", svm_scores, svm_labels, get_val_scores_svm, X_val, y_val, OUT_DIR)

    # CNN
    cnn_logits = np.load(RESULTS_DIR / "baseline_cnn" / "test_logits.npy")
    cnn_labels = np.load(RESULTS_DIR / "baseline_cnn" / "test_labels.npy")
    results["cnn"] = run_model("cnn", cnn_logits, cnn_labels, get_val_logits_cnn, X_val, y_val, OUT_DIR)

    # MLP
    mlp_logits = np.load(RESULTS_DIR / "baseline_mlp" / "test_logits.npy")
    mlp_labels = np.load(RESULTS_DIR / "baseline_mlp" / "test_labels.npy")
    results["mlp"] = run_model("mlp", mlp_logits, mlp_labels, get_val_logits_mlp, X_val, y_val, OUT_DIR)

    with open(OUT_DIR / "calibration_metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    # --- Part G: summary table -----------------------------------------------
    print(f"\n{'=' * 100}")
    header = f"{'MODEL':<6} | {'Uncal ECE':>10} | {'Cal ECE':>8} | {'ECE reduction':>13} | " \
             f"{'Uncal Brier':>11} | {'Cal Brier':>9} | {'Accuracy':>8} | {'T*':>6}"
    print(header)
    print("-" * len(header))
    for name in ["svm", "cnn", "mlp"]:
        u = results[name]["uncalibrated"]
        c = results[name]["calibrated"]
        reduction = u["ECE"] / c["ECE"] if c["ECE"] > 0 else float("inf")
        print(f"{name.upper():<6} | {u['ECE']:>10.4f} | {c['ECE']:>8.4f} | {reduction:>11.2f}x | "
              f"{u['Brier']:>11.4f} | {c['Brier']:>9.4f} | {c['accuracy'] * 100:>7.2f}% | {c['temperature']:>6.3f}")
    print("=" * 100)
    print(f"\nSaved: {OUT_DIR / 'calibration_metrics.json'}")
    print(f"Saved: reliability diagrams (.png @300dpi + .pdf) in {OUT_DIR}")


if __name__ == "__main__":
    main()
