"""SNR degradation study for the five DroneRF baselines: SVM, CNN (GroupNorm),
CNN-BN, MLP (LayerNorm-free), MLP-BN.

Secondary paper contribution: measures how test accuracy AND calibration
(ECE, Brier, NLL) degrade as Additive White Gaussian Noise (AWGN) is injected
into the raw power spectrum at progressively lower SNR levels. Calibration is
reported both uncalibrated and with the clean-data temperature T* (loaded from
results/calibration/calibration_metrics.json) applied WITHOUT re-fitting per
SNR level — re-fitting per level would leak test-time noise-level information
into the calibrator and defeat the purpose of testing robustness.

Noise is added to the pre-log-transform raw power spectrum (X_test_raw.npy,
produced by preprocess/load_and_split.py), then log10 + the training
StandardScaler are re-applied, mirroring the original preprocessing pipeline
exactly. This simulates realistic RF signal degradation rather than injecting
noise into already-transformed features.

Run with: python eval\\snr_degradation.py   (after activating venv)
Requires: data/processed/X_test_raw.npy (re-run preprocess/load_and_split.py
first if it does not exist yet) and results/calibration/calibration_metrics.json
(run eval/calibration.py first).
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

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "train"))
from utils import PROCESSED_DIR, RESULTS_DIR  # noqa: E402
from train_mlp import MLP  # noqa: E402
from train_cnn import CNN1D  # noqa: E402
from train_mlp_bn import MLPBN  # noqa: E402
from train_cnn_bn import CNN1DBN  # noqa: E402

OUT_DIR = RESULTS_DIR / "snr_degradation"
CALIBRATION_METRICS_PATH = RESULTS_DIR / "calibration" / "calibration_metrics.json"
LOG_EPSILON = 1e-10
NUM_CLASSES = 4

SNR_LEVELS_DB = [20, 15, 10, 5, 0, -5, -10, -15]
NOISE_SEEDS = [0, 1, 2, 3, 4]

MODEL_COLORS = {
    "svm": "#C44E52",
    "cnn": "#4C72B0",
    "cnn_bn": "#8172B2",
    "mlp": "#55A868",
    "mlp_bn": "#CCB974",
}
MODEL_DISPLAY_NAMES = {
    "svm": "SVM",
    "cnn": "CNN (GroupNorm)",
    "cnn_bn": "CNN-BN",
    "mlp": "MLP (LayerNorm-free)",
    "mlp_bn": "MLP-BN",
}


# ---------------------------------------------------------------------------
# Metrics (mirrors eval/calibration.py)
# ---------------------------------------------------------------------------

def softmax(x, axis=1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def compute_ece(probs, labels, n_bins=15):
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    correctness = (predictions == labels).astype(np.float64)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    n = len(labels)
    ece = 0.0
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
        ece += (bin_size / n) * abs(avg_conf - avg_acc)
    return float(ece)


def compute_brier(probs, labels, num_classes=NUM_CLASSES):
    one_hot = np.eye(num_classes)[labels]
    return float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))


def compute_nll(probs, labels, eps=1e-12):
    p_true = np.clip(probs[np.arange(len(labels)), labels], eps, 1.0)
    return float(-np.mean(np.log(p_true)))


def compute_accuracy(probs, labels):
    return float((probs.argmax(axis=1) == labels).mean())


# ---------------------------------------------------------------------------
# Noise injection + preprocessing
# ---------------------------------------------------------------------------

def add_awgn(x_raw, snr_db, rng):
    """Inject AWGN into the raw power spectrum at the given per-sample SNR."""
    signal_power = np.mean(x_raw ** 2, axis=1, keepdims=True)
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = rng.standard_normal(x_raw.shape) * np.sqrt(noise_power)
    return x_raw + noise


def preprocess_raw(x_raw, scaler, snr_db="clean"):
    # Power spectrum values are physically non-negative, but AWGN injection can produce
    # negative values when signal magnitude is small relative to noise. Clip to LOG_EPSILON
    # before log to preserve physical validity and prevent NaN. This matches standard
    # practice in RF signal processing under low-SNR simulation.
    x_noisy_clipped = np.maximum(x_raw, LOG_EPSILON)
    x_log = np.log10(x_noisy_clipped)
    x_standardized = scaler.transform(x_log).astype(np.float32)
    assert not np.any(np.isnan(x_standardized)), f"NaN detected after standardization at SNR={snr_db}"
    return x_standardized


# ---------------------------------------------------------------------------
# Model loading / inference
# ---------------------------------------------------------------------------

def load_torch_model(model_class, result_dir_name):
    device = torch.device("cpu")
    model = model_class().to(device)
    model.load_state_dict(torch.load(RESULTS_DIR / result_dir_name / "model.pt", map_location=device))
    model.eval()
    return model


def torch_infer(model, X, reshape_conv):
    X_in = X.reshape(X.shape[0], 1, -1) if reshape_conv else X
    with torch.no_grad():
        logits = model(torch.from_numpy(X_in).float()).numpy()
    return logits


def build_model_specs():
    svm_model = joblib.load(RESULTS_DIR / "baseline_svm" / "model.joblib")
    cnn_model = load_torch_model(CNN1D, "baseline_cnn")
    cnn_bn_model = load_torch_model(CNN1DBN, "baseline_cnn_bn")
    mlp_model = load_torch_model(MLP, "baseline_mlp")
    mlp_bn_model = load_torch_model(MLPBN, "baseline_mlp_bn")

    return [
        {"name": "svm", "infer": lambda X: svm_model.decision_function(X)},
        {"name": "cnn", "infer": lambda X: torch_infer(cnn_model, X, reshape_conv=True)},
        {"name": "cnn_bn", "infer": lambda X: torch_infer(cnn_bn_model, X, reshape_conv=True)},
        {"name": "mlp", "infer": lambda X: torch_infer(mlp_model, X, reshape_conv=False)},
        {"name": "mlp_bn", "infer": lambda X: torch_infer(mlp_bn_model, X, reshape_conv=False)},
    ]


# ---------------------------------------------------------------------------
# Main degradation loop
# ---------------------------------------------------------------------------

def evaluate_at_level(scores, labels, T_star):
    probs_uncal = softmax(scores, axis=1)
    probs_cal = softmax(scores / T_star, axis=1)
    return {
        "accuracy": compute_accuracy(probs_uncal, labels),
        "ece_uncalibrated": compute_ece(probs_uncal, labels),
        "ece_calibrated": compute_ece(probs_cal, labels),
        "brier": compute_brier(probs_uncal, labels),
        "brier_calibrated": compute_brier(probs_cal, labels),
        "nll": compute_nll(probs_uncal, labels),
        "nll_calibrated": compute_nll(probs_cal, labels),
    }


def aggregate(metric_dicts):
    keys = metric_dicts[0].keys()
    agg = {}
    for k in keys:
        vals = np.array([m[k] for m in metric_dicts])
        agg[f"{k}_mean"] = float(vals.mean())
        agg[f"{k}_std"] = float(vals.std())
    return agg


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    X_test_raw = np.load(PROCESSED_DIR / "X_test_raw.npy")
    y_test = np.load(PROCESSED_DIR / "y_test.npy")
    scaler = joblib.load(PROCESSED_DIR / "scaler.joblib")

    with open(CALIBRATION_METRICS_PATH) as f:
        calibration_metrics = json.load(f)
    temperatures = {name: calibration_metrics[name]["calibrated"]["temperature"]
                     for name in MODEL_DISPLAY_NAMES}

    model_specs = build_model_specs()

    results = {spec["name"]: {} for spec in model_specs}

    # --- Clean (infinite SNR) reference -------------------------------------
    print(f"\n{'=' * 60}\nCLEAN (reference)\n{'=' * 60}")
    X_test_clean = preprocess_raw(X_test_raw, scaler)
    for spec in model_specs:
        name = spec["name"]
        scores = spec["infer"](X_test_clean)
        metrics = evaluate_at_level(scores, y_test, temperatures[name])
        results[name]["clean"] = metrics
        print(f"  {name:<8} acc={metrics['accuracy'] * 100:.2f}% "
              f"ECE_uncal={metrics['ece_uncalibrated']:.4f} "
              f"ECE_cal={metrics['ece_calibrated']:.4f}")

    # --- Noisy levels, averaged over multiple noise seeds -------------------
    for snr_db in SNR_LEVELS_DB:
        print(f"\n{'=' * 60}\nSNR = {snr_db} dB\n{'=' * 60}")
        per_model_runs = {spec["name"]: [] for spec in model_specs}

        for seed in NOISE_SEEDS:
            rng = np.random.default_rng(seed)
            X_noisy_raw = add_awgn(X_test_raw, snr_db, rng)
            X_noisy = preprocess_raw(X_noisy_raw, scaler, snr_db=snr_db)

            for spec in model_specs:
                name = spec["name"]
                scores = spec["infer"](X_noisy)
                metrics = evaluate_at_level(scores, y_test, temperatures[name])
                per_model_runs[name].append(metrics)

        for spec in model_specs:
            name = spec["name"]
            agg = aggregate(per_model_runs[name])
            results[name][str(snr_db)] = agg
            print(f"  {name:<8} acc={agg['accuracy_mean'] * 100:.2f}%±{agg['accuracy_std'] * 100:.2f} "
                  f"ECE_uncal={agg['ece_uncalibrated_mean']:.4f}±{agg['ece_uncalibrated_std']:.4f} "
                  f"ECE_cal={agg['ece_calibrated_mean']:.4f}±{agg['ece_calibrated_std']:.4f}")

    metrics_path = OUT_DIR / "degradation_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {metrics_path}")

    make_plots(results)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _series(results, name, key, mean_suffix="_mean", std_suffix="_std"):
    """Return (snr_axis, means, stds) for a metric across clean + noisy levels.

    snr_axis uses 25 dB as a placeholder x-position for the "clean" reference
    (visually to the right of the highest tested SNR) since clean data has no
    finite SNR value.
    """
    snr_axis, means, stds = [], [], []
    if "clean" in results[name]:
        snr_axis.append(25)
        d = results[name]["clean"]
        means.append(d[key])
        stds.append(0.0)
    for snr_db in SNR_LEVELS_DB:
        d = results[name][str(snr_db)]
        snr_axis.append(snr_db)
        means.append(d[f"{key}{mean_suffix}"])
        stds.append(d[f"{key}{std_suffix}"])
    return np.array(snr_axis), np.array(means), np.array(stds)


def _style_axis(ax, ylabel, title):
    ax.set_xlabel("SNR (dB)  [25 = clean / no noise]")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize=9)


def plot_accuracy_vs_snr(results):
    fig, ax = plt.subplots(figsize=(7, 5))
    for name in MODEL_DISPLAY_NAMES:
        snr_axis, means, stds = _series(results, name, "accuracy")
        ax.errorbar(snr_axis, means * 100, yerr=stds * 100, marker="o", markersize=4,
                    linewidth=1.8, capsize=3, color=MODEL_COLORS[name],
                    label=MODEL_DISPLAY_NAMES[name])
    _style_axis(ax, "Accuracy (%)", "Accuracy vs. SNR (AWGN on raw spectrum)")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "accuracy_vs_snr.png", dpi=300)
    fig.savefig(OUT_DIR / "accuracy_vs_snr.pdf")
    plt.close(fig)


def plot_ece_vs_snr(results, key, out_name, title):
    fig, ax = plt.subplots(figsize=(7, 5))
    for name in MODEL_DISPLAY_NAMES:
        snr_axis, means, stds = _series(results, name, key)
        ax.errorbar(snr_axis, means, yerr=stds, marker="o", markersize=4,
                    linewidth=1.8, capsize=3, color=MODEL_COLORS[name],
                    label=MODEL_DISPLAY_NAMES[name])
    _style_axis(ax, "ECE", title)
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"{out_name}.png", dpi=300)
    fig.savefig(OUT_DIR / f"{out_name}.pdf")
    plt.close(fig)


def plot_comparison(results):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 9), sharex=True)

    for name in MODEL_DISPLAY_NAMES:
        snr_axis, means, stds = _series(results, name, "accuracy")
        ax1.errorbar(snr_axis, means * 100, yerr=stds * 100, marker="o", markersize=4,
                     linewidth=1.8, capsize=3, color=MODEL_COLORS[name],
                     label=MODEL_DISPLAY_NAMES[name])
    ax1.set_ylabel("Accuracy (%)")
    ax1.set_title("Accuracy and Calibration Degradation vs. SNR")
    ax1.grid(alpha=0.3)
    ax1.legend(loc="best", fontsize=8)

    for name in MODEL_DISPLAY_NAMES:
        snr_axis, means_u, stds_u = _series(results, name, "ece_uncalibrated")
        _, means_c, stds_c = _series(results, name, "ece_calibrated")

        ax2.errorbar(snr_axis, means_u, yerr=stds_u, marker="o", markersize=4,
                     linewidth=1.6, linestyle="-", capsize=3, color=MODEL_COLORS[name],
                     label=f"{MODEL_DISPLAY_NAMES[name]} (uncal.)")
        ax2.errorbar(snr_axis, means_c, yerr=stds_c, marker="s", markersize=4,
                     linewidth=1.6, linestyle="--", capsize=3, color=MODEL_COLORS[name],
                     label=f"{MODEL_DISPLAY_NAMES[name]} (cal., clean T*)")

    ax2.set_xlabel("SNR (dB)  [25 = clean / no noise]")
    ax2.set_ylabel("ECE")
    ax2.grid(alpha=0.3)
    ax2.legend(loc="best", fontsize=7, ncol=2)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "degradation_comparison.png", dpi=300)
    fig.savefig(OUT_DIR / "degradation_comparison.pdf")
    plt.close(fig)


def make_plots(results):
    plt.rcParams.update({"font.size": 11})
    plot_accuracy_vs_snr(results)
    plot_ece_vs_snr(results, "ece_uncalibrated", "ece_uncalibrated_vs_snr",
                     "Uncalibrated ECE vs. SNR")
    plot_ece_vs_snr(results, "ece_calibrated", "ece_calibrated_vs_snr",
                     "Calibrated ECE (clean-data T*) vs. SNR")
    plot_comparison(results)
    print(f"Saved plots (.png @300dpi + .pdf) in {OUT_DIR}")


if __name__ == "__main__":
    main()
