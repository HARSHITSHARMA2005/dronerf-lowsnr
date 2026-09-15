"""Calibration analysis for the five DroneRF baselines: SVM, CNN (GroupNorm),
CNN-BN (BatchNorm ablation), MLP (LayerNorm-free), MLP-BN (BatchNorm
ablation).

Core paper contribution: measures whether softmax/decision-function confidence
scores on the clean test set are trustworthy (ECE, MCE, Brier, NLL), draws
reliability diagrams, and applies temperature scaling (Guo et al. 2017) to fix
miscalibration without changing predictions (argmax is temperature-invariant).
The cnn_bn/mlp_bn variants exist to test the ablation hypothesis that
BatchNorm-based classifiers (as used in prior DroneRF work, e.g. Al-Sa'd 2019,
Al-Emadi 2020) are more severely miscalibrated than our GroupNorm/LayerNorm-
free main baselines.

SVM handling note: train/train_svm.py trains a plain SVC/LinearSVC WITHOUT
probability=True (see model.decision_function usage there). Its saved
test_scores.npy / val "scores" are the raw decision_function output, shape
(n, 4) for the one-vs-rest-shaped multiclass decision function -- signed
distances to the hyperplanes, with arbitrary scale, NOT logits. We treat
these margin scores as pseudo-logits and apply softmax over them ("svm" /
"svm_pseudologit" below), exactly as if they were logits from a neural net.
This is unconventional and a scikit-learn-literate reviewer will flag it, so
we also report two variants trained WITH probability=True (see
train/train_svm_prob.py, results/baseline_svm_prob/), to check whether the
severe miscalibration seen under the pseudo-logit approach is a real property
of the SVM or an artifact of that conversion:
  - svm_pseudologit : decision_function + softmax (the original approach above)
  - svm_platt       : predict_proba() output used directly -- already
                       Platt-scaled via sklearn's internal 5-fold CV
  - svm_platt_temp  : svm_platt, ADDITIONALLY temperature-scaled on top (a
                       double calibration -- see run_svm_proba_variants() for
                       why this may not be a meaningful thing to do, kept for
                       completeness/transparency)
If results/baseline_svm_prob/ does not exist yet (train_svm_prob.py has not
been run), the svm_platt / svm_platt_temp variants are skipped and only the
original svm/svm_pseudologit results are produced.

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
from train_mlp_bn import MLPBN  # noqa: E402
from train_cnn_bn import CNN1DBN  # noqa: E402

OUT_DIR = RESULTS_DIR / "calibration"
N_BINS_ECE = 15
N_BINS_PLOT = 10
CLASS_NAMES = ["background", "Bebop", "AR", "Phantom"]
NUM_CLASSES = 4

MODEL_DISPLAY_NAMES = {
    "svm": "SVM (decision softmax)",
    "svm_platt": "SVM (Platt)",
    "svm_platt_temp": "SVM (Platt + T*)",
    "cnn": "CNN-GN",
    "cnn_bn": "CNN-BN",
    "mlp": "MLP",
    "mlp_bn": "MLP-BN",
}


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
    display_name = MODEL_DISPLAY_NAMES.get(model_name, model_name.upper())
    ax1.set_title(f"Reliability Diagram — {display_name} ({tag})", fontsize=11)
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

    Follows the reference temperature-scaling implementation (Guo et al. 2017):
    T is optimized directly (no softplus reparameterization — that dampens the
    gradient via the chain rule and was verified to leave LBFGS unconverged,
    e.g. true val-NLL-minimizing T~1.2 for MLP but the softplus version
    returned T~1.45 after 50 iterations, making calibration *worse*).
    `line_search_fn="strong_wolfe"` and 3 repeated .step(closure) calls make
    convergence reliable; T is only clamped (min=0.05) inside the loss to
    guard against numerical issues, never reparameterized.
    """
    val_logits_t = torch.from_numpy(val_logits).float()
    val_labels_t = torch.from_numpy(val_labels).long()

    T = torch.nn.Parameter(torch.ones(1) * 1.0)
    optimizer = torch.optim.LBFGS([T], lr=0.01, max_iter=100, line_search_fn="strong_wolfe")
    criterion = nn.CrossEntropyLoss()

    def closure():
        optimizer.zero_grad()
        loss = criterion(val_logits_t / T.clamp(min=0.05), val_labels_t)
        loss.backward()
        return loss

    with torch.no_grad():
        nll_before = criterion(val_logits_t / T.clamp(min=0.05), val_labels_t).item()

    for _ in range(3):
        optimizer.step(closure)

    with torch.no_grad():
        T_final = float(T.clamp(min=0.05).item())
        nll_after = criterion(val_logits_t / T.clamp(min=0.05), val_labels_t).item()

    if nll_after > nll_before + 1e-6:
        print(f"  WARNING: temperature fit did not improve val NLL "
              f"({nll_before:.4f} -> {nll_after:.4f}); check convergence.")

    return T_final


# ---------------------------------------------------------------------------
# Per-model pipelines
# ---------------------------------------------------------------------------

def make_val_logits_fn(model_class, result_dir_name, reshape_conv):
    """Build a get_val_logits(X_val) closure for a torch model checkpoint."""
    def fn(X_val):
        device = torch.device("cpu")
        model = model_class().to(device)
        model.load_state_dict(torch.load(RESULTS_DIR / result_dir_name / "model.pt", map_location=device))
        model.eval()
        X_in = X_val.reshape(X_val.shape[0], 1, -1) if reshape_conv else X_val
        with torch.no_grad():
            logits = model(torch.from_numpy(X_in).float()).numpy()
        return logits
    return fn


def get_val_scores_svm(X_val):
    model = joblib.load(RESULTS_DIR / "baseline_svm" / "model.joblib")
    return model.decision_function(X_val)


# Each entry: (display_name, results_subdir, reshape_for_conv, val_fn_builder)
# val_fn_builder is called with no args to get the get_val_scores(X_val) fn,
# except for svm which uses the dedicated get_val_scores_svm directly.
MODEL_SPECS = [
    {"name": "svm", "dir": "baseline_svm", "scores_file": "test_scores.npy",
     "val_fn": get_val_scores_svm},
    {"name": "cnn", "dir": "baseline_cnn", "scores_file": "test_logits.npy",
     "val_fn": make_val_logits_fn(CNN1D, "baseline_cnn", reshape_conv=True)},
    {"name": "cnn_bn", "dir": "baseline_cnn_bn", "scores_file": "test_logits.npy",
     "val_fn": make_val_logits_fn(CNN1DBN, "baseline_cnn_bn", reshape_conv=True)},
    {"name": "mlp", "dir": "baseline_mlp", "scores_file": "test_logits.npy",
     "val_fn": make_val_logits_fn(MLP, "baseline_mlp", reshape_conv=False)},
    {"name": "mlp_bn", "dir": "baseline_mlp_bn", "scores_file": "test_logits.npy",
     "val_fn": make_val_logits_fn(MLPBN, "baseline_mlp_bn", reshape_conv=False)},
]


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


def run_svm_proba_variants(test_proba, test_labels, val_proba, y_val, out_dir):
    """Produces the svm_platt and svm_platt_temp results.

    svm_platt: predict_proba() output from an SVC(probability=True) fit, used
    directly. This is already a calibrated probability estimate from
    sklearn's internal Platt scaling (5-fold CV on the training data) -- NOT
    logits, so it is used as-is, with no softmax.

    svm_platt_temp: the same Platt-scaled probabilities, additionally
    temperature-scaled (fit on val_proba, applied to test_proba). Temperature
    scaling expects logit-like inputs, so we convert back to pseudo-logits via
    log(p): since sum_i p_i = 1, softmax(log(p)) == p exactly at T=1, and for
    T != 1, softmax(log(p)/T)_i = p_i^(1/T) / sum_j p_j^(1/T) -- a monotonic
    power rescaling of the original probabilities, so argmax (accuracy) is
    still preserved for any T > 0, same as standard temperature scaling on
    true logits.
    NOTE (double calibration): svm_platt is already calibrated by Platt
    scaling's internal CV. Fitting a second, independent calibration step
    (temperature scaling) on top of it, on a different data split (our val
    set) is not a standard or obviously principled thing to do -- there is no
    guarantee stacking two calibrators improves anything, and it could in
    principle overcorrect. It is reported here only so we can see the number,
    not as a recommended pipeline.
    """
    print(f"\n{'=' * 60}\nSVM_PLATT / SVM_PLATT_TEMP\n{'=' * 60}")
    eps = 1e-12

    # --- svm_platt: Platt-scaled probabilities, used as-is ------------------
    uncal_metrics = compute_all_metrics(test_proba, test_labels)
    print(f"[svm_platt]      ECE={uncal_metrics['ECE']:.4f} MCE={uncal_metrics['MCE']:.4f} "
          f"Brier={uncal_metrics['Brier']:.4f} NLL={uncal_metrics['NLL']:.4f} "
          f"acc={uncal_metrics['accuracy'] * 100:.2f}%")
    plot_reliability(test_proba, test_labels, uncal_metrics["ECE"], "svm_platt", "uncalibrated", out_dir)

    # --- svm_platt_temp: additionally temperature-scale via log-pseudologits
    val_pseudo_logits = np.log(np.clip(val_proba, eps, 1.0))
    T_star = fit_temperature(val_pseudo_logits, y_val)
    print(f"  T* (double-calibration, on top of Platt scaling) = {T_star:.4f}")

    test_pseudo_logits = np.log(np.clip(test_proba, eps, 1.0))
    test_probs_cal = softmax(test_pseudo_logits / T_star, axis=1)
    cal_metrics = compute_all_metrics(test_probs_cal, test_labels)
    cal_metrics["temperature"] = T_star
    if abs(cal_metrics["accuracy"] - uncal_metrics["accuracy"]) > 1e-9:
        print(f"  WARNING: accuracy changed after double calibration! "
              f"{uncal_metrics['accuracy']} -> {cal_metrics['accuracy']}")
    print(f"[svm_platt_temp] ECE={cal_metrics['ECE']:.4f} MCE={cal_metrics['MCE']:.4f} "
          f"Brier={cal_metrics['Brier']:.4f} NLL={cal_metrics['NLL']:.4f} "
          f"acc={cal_metrics['accuracy'] * 100:.2f}%")
    plot_reliability(test_probs_cal, test_labels, cal_metrics["ECE"], "svm_platt_temp", "calibrated", out_dir)

    return uncal_metrics, cal_metrics


def plot_only():
    """Redraw reliability diagrams from already-saved scores + calibration_metrics.json
    T* values, without refitting temperatures or re-running inference."""
    metrics_path = OUT_DIR / "calibration_metrics.json"
    with open(metrics_path) as f:
        results = json.load(f)

    titles = []
    for spec in MODEL_SPECS:
        name, subdir, scores_file = spec["name"], spec["dir"], spec["scores_file"]
        scores = np.load(RESULTS_DIR / subdir / scores_file)
        labels = np.load(RESULTS_DIR / subdir / "test_labels.npy")
        T_star = results[name]["calibrated"]["temperature"]

        probs_uncal = softmax(scores, axis=1)
        _, pdf1 = plot_reliability(probs_uncal, labels, results[name]["uncalibrated"]["ECE"],
                                    name, "uncalibrated", OUT_DIR)
        probs_cal = softmax(scores / T_star, axis=1)
        _, pdf2 = plot_reliability(probs_cal, labels, results[name]["calibrated"]["ECE"],
                                    name, "calibrated", OUT_DIR)
        display_name = MODEL_DISPLAY_NAMES.get(name, name.upper())
        titles.append(f"Reliability Diagram — {display_name} (uncalibrated)")
        titles.append(f"Reliability Diagram — {display_name} (calibrated)")

    svm_prob_dir = RESULTS_DIR / "baseline_svm_prob"
    test_proba_path = svm_prob_dir / "test_proba.npy"
    if test_proba_path.exists() and "svm_platt" in results:
        test_proba = np.load(test_proba_path)
        test_labels_svm_prob_path = svm_prob_dir / "test_labels.npy"
        test_labels_svm_prob = (np.load(test_labels_svm_prob_path)
                                 if test_labels_svm_prob_path.exists()
                                 else np.load(RESULTS_DIR / "baseline_svm" / "test_labels.npy"))
        plot_reliability(test_proba, test_labels_svm_prob, results["svm_platt"]["ECE"],
                          "svm_platt", "uncalibrated", OUT_DIR)
        titles.append(f"Reliability Diagram — {MODEL_DISPLAY_NAMES['svm_platt']} (uncalibrated)")

        T_star = results["svm_platt_temp"]["temperature"]
        eps = 1e-12
        pseudo_logits = np.log(np.clip(test_proba, eps, 1.0))
        probs_cal = softmax(pseudo_logits / T_star, axis=1)
        plot_reliability(probs_cal, test_labels_svm_prob, results["svm_platt_temp"]["ECE"],
                          "svm_platt_temp", "calibrated", OUT_DIR)
        titles.append(f"Reliability Diagram — {MODEL_DISPLAY_NAMES['svm_platt_temp']} (calibrated)")

    print(f"Regenerated reliability diagrams from {metrics_path} (--plot-only, "
          f"no refit/inference).")
    print("\nFinal reliability-diagram titles:")
    for t in titles:
        print(f"  {t!r}")


def main():
    if "--plot-only" in sys.argv:
        plot_only()
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    _, X_val, _, _, y_val, _ = load_splits()

    results = {}
    for spec in MODEL_SPECS:
        name, subdir, scores_file, val_fn = spec["name"], spec["dir"], spec["scores_file"], spec["val_fn"]
        scores = np.load(RESULTS_DIR / subdir / scores_file)
        labels = np.load(RESULTS_DIR / subdir / "test_labels.npy")
        print(f"{name.upper()} {scores_file} shape={scores.shape} dtype={scores.dtype}")
        results[name] = run_model(name, scores, labels, val_fn, X_val, y_val, OUT_DIR)

    # --- SVM Platt-scaled variants (optional -- requires results/baseline_svm_prob/,
    # produced by train/train_svm_prob.py) --------------------------------------
    results["svm_pseudologit"] = results["svm"]  # alias: same data, clearer 3-way name

    svm_prob_dir = RESULTS_DIR / "baseline_svm_prob"
    test_proba_path = svm_prob_dir / "test_proba.npy"
    val_proba_path = svm_prob_dir / "val_proba.npy"
    if test_proba_path.exists() and val_proba_path.exists():
        print(f"\n{'#' * 60}\nSVM Platt-scaled variants ({svm_prob_dir})\n{'#' * 60}")
        test_proba = np.load(test_proba_path)
        val_proba = np.load(val_proba_path)
        test_labels_svm_prob_path = svm_prob_dir / "test_labels.npy"
        test_labels_svm_prob = (np.load(test_labels_svm_prob_path)
                                 if test_labels_svm_prob_path.exists()
                                 else np.load(RESULTS_DIR / "baseline_svm" / "test_labels.npy"))
        svm_platt, svm_platt_temp = run_svm_proba_variants(
            test_proba, test_labels_svm_prob, val_proba, y_val, OUT_DIR)
        results["svm_platt"] = svm_platt
        results["svm_platt_temp"] = svm_platt_temp
    else:
        print(f"\n({svm_prob_dir} not found -- skipping svm_platt / svm_platt_temp "
              f"variants. Run train/train_svm_prob.py first to enable the 3-way "
              f"SVM calibration comparison.)")

    with open(OUT_DIR / "calibration_metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    # --- SVM 3-way comparison: is the 23% ECE real, or a pseudo-logit artifact? --
    if "svm_platt" in results:
        print(f"\n{'=' * 100}\nSVM 3-WAY CALIBRATION COMPARISON (clean test set)\n{'=' * 100}")
        header2 = (f"{'VARIANT':<22} | {'ECE':>8} | {'MCE':>8} | {'Brier':>8} | "
                   f"{'NLL':>8} | {'Accuracy':>9}")
        print(header2)
        print("-" * len(header2))
        rows = [
            ("svm_pseudologit", results["svm_pseudologit"]["uncalibrated"]),
            ("svm_pseudologit + T*", results["svm_pseudologit"]["calibrated"]),
            ("svm_platt", results["svm_platt"]),
            ("svm_platt + T* (dbl-cal)", results["svm_platt_temp"]),
        ]
        for label, m in rows:
            print(f"{label:<22} | {m['ECE']:>8.4f} | {m['MCE']:>8.4f} | {m['Brier']:>8.4f} | "
                  f"{m['NLL']:>8.4f} | {m['accuracy'] * 100:>8.2f}%")
        print("=" * 100)

    # --- Part G: summary table -----------------------------------------------
    print(f"\n{'=' * 100}")
    header = f"{'MODEL':<8} | {'Uncal ECE':>10} | {'Cal ECE':>8} | {'ECE reduction':>13} | " \
             f"{'Uncal Brier':>11} | {'Cal Brier':>9} | {'Accuracy':>8} | {'T*':>6}"
    print(header)
    print("-" * len(header))
    for spec in MODEL_SPECS:
        name = spec["name"]
        u = results[name]["uncalibrated"]
        c = results[name]["calibrated"]
        reduction = u["ECE"] / c["ECE"] if c["ECE"] > 0 else float("inf")
        print(f"{name.upper():<8} | {u['ECE']:>10.4f} | {c['ECE']:>8.4f} | {reduction:>11.2f}x | "
              f"{u['Brier']:>11.4f} | {c['Brier']:>9.4f} | {c['accuracy'] * 100:>7.2f}% | {c['temperature']:>6.3f}")
    print("=" * 100)
    print(f"\nSaved: {OUT_DIR / 'calibration_metrics.json'}")
    print(f"Saved: reliability diagrams (.png @300dpi + .pdf) in {OUT_DIR}")


if __name__ == "__main__":
    main()
