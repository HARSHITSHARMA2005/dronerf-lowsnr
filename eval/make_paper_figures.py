"""
Generate publication-quality figures from existing diagnostic results.

Plotting only — no new experiments are run. Reads:
  - results/diagnostics/multiseed_eval.json
  - results/diagnostics/bn_stored_stats_mechanism.json
  - results/diagnostics/instancenorm_control.json
  - results/baseline_cnn_bn/test_labels.npy (true class prevalence)

Writes to results/paper_figures/:
  - fig_multiseed_accuracy.{png,pdf}
  - fig_bn_mechanism.{png,pdf}
  - fig_prediction_collapse.{png,pdf}
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DIAG_DIR = ROOT / "results" / "diagnostics"
OUT_DIR = ROOT / "results" / "paper_figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = ["background", "Bebop", "AR", "Phantom"]

plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 100,
    "axes.grid": True,
    "grid.alpha": 0.3,
})

CHANCE = 25.0


def save_fig(fig, name):
    png_path = OUT_DIR / f"{name}.png"
    pdf_path = OUT_DIR / f"{name}.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    print(f"Saved {png_path}")
    print(f"Saved {pdf_path}")


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# FIGURE A — fig_multiseed_accuracy
# ---------------------------------------------------------------------------
def make_fig_multiseed_accuracy():
    data = load_json(DIAG_DIR / "multiseed_eval.json")

    seeds = ["42", "43", "44"]
    snr_levels = ["10", "5", "0"]  # clean handled separately, placed last
    x_labels = ["10", "5", "0", "clean"]
    x_pos = np.arange(len(x_labels))

    def gather(model_prefix):
        # returns dict: level -> (mean_of_seed_means, std_across_seeds)
        # for "clean" the seed-level value is a single accuracy (not mean/std
        # across noise seeds), so we take mean/std across the 3 training seeds
        # of that single clean accuracy.
        result = {}
        for level, key in [("10", "10"), ("5", "5"), ("0", "0")]:
            seed_vals = [
                data[f"{model_prefix}_s{s}"][key]["accuracy_mean"] for s in seeds
            ]
            result[level] = (float(np.mean(seed_vals)), float(np.std(seed_vals)))
        clean_vals = [data[f"{model_prefix}_s{s}"]["clean"]["accuracy"] for s in seeds]
        result["clean"] = (float(np.mean(clean_vals)), float(np.std(clean_vals)))
        return result

    bn_stats = gather("cnn_bn")
    in_stats = gather("cnn_in")

    print("\n=== FIGURE A: fig_multiseed_accuracy ===")
    for model_name, stats in [("CNN-BN", bn_stats), ("CNN-IN", in_stats)]:
        for level in ["10", "5", "0", "clean"]:
            mean_pct = stats[level][0] * 100
            std_pct = stats[level][1] * 100
            print(f"{model_name} @ {level} dB: mean={mean_pct:.3f}% std={std_pct:.3f}%")

    fig, ax = plt.subplots(figsize=(5.0, 3.6))

    for model_name, stats, color, marker in [
        ("CNN-BN", bn_stats, "tab:red", "o"),
        ("CNN-IN", in_stats, "tab:blue", "s"),
    ]:
        means = [stats[lvl][0] * 100 for lvl in ["10", "5", "0", "clean"]]
        stds = [stats[lvl][1] * 100 for lvl in ["10", "5", "0", "clean"]]
        ax.errorbar(
            x_pos, means, yerr=stds, label=model_name, color=color,
            marker=marker, markersize=6, capsize=4, capthick=1.2,
            linewidth=1.6, elinewidth=1.2,
        )

    ax.axhline(CHANCE, color="grey", linestyle="--", linewidth=1.2,
               label="random chance (4 classes)")

    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels)
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Accuracy vs SNR across three training seeds")
    ax.legend(loc="best", frameon=True)
    fig.tight_layout()

    save_fig(fig, "fig_multiseed_accuracy")
    plt.close(fig)


# ---------------------------------------------------------------------------
# FIGURE B — fig_bn_mechanism
# ---------------------------------------------------------------------------
def make_fig_bn_mechanism():
    data = load_json(DIAG_DIR / "bn_stored_stats_mechanism.json")

    conditions = ["clean", "10", "5", "0"]
    x_labels = ["clean", "10", "5", "0"]

    def acc(mode, cond):
        entry = data[mode][cond]
        if "accuracy" in entry:
            return entry["accuracy"] * 100
        return entry["accuracy_mean"] * 100

    stored = [acc("mode1_stored_stats", c) for c in conditions]
    batch = [acc("mode2_batch_stats", c) for c in conditions]

    print("\n=== FIGURE B: fig_bn_mechanism ===")
    for c, s, b in zip(x_labels, stored, batch):
        print(f"{c} dB: stored stats={s:.3f}% test-time stats={b:.3f}% "
              f"recovery={b - s:+.1f} pts")

    x_pos = np.arange(len(x_labels))
    width = 0.36

    fig, ax = plt.subplots(figsize=(5.0, 3.6))
    bars1 = ax.bar(x_pos - width / 2, stored, width,
                    label="stored statistics (standard eval)", color="tab:red")
    bars2 = ax.bar(x_pos + width / 2, batch, width,
                    label="test-time batch statistics", color="tab:blue")

    ax.axhline(CHANCE, color="grey", linestyle="--", linewidth=1.2,
               label="random chance (4 classes)")

    for i, c in enumerate(x_labels):
        if c in ("5", "0"):
            recovery = batch[i] - stored[i]
            ax.annotate(
                f"+{recovery:.1f} pts",
                xy=(x_pos[i] + width / 2, batch[i]),
                xytext=(x_pos[i] + width / 2, batch[i] + 3),
                ha="center", fontsize=9, fontweight="bold",
                color="black", zorder=10,
            )

    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels)
    ax.set_xlabel("SNR condition")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 100)
    ax.set_title("CNN-BN: same weights, normalisation statistics source swapped")
    ax.legend(loc="lower left", frameon=True, fontsize=8.5)
    fig.tight_layout()

    save_fig(fig, "fig_bn_mechanism")
    plt.close(fig)


# ---------------------------------------------------------------------------
# FIGURE C — fig_prediction_collapse
# ---------------------------------------------------------------------------
def make_fig_prediction_collapse():
    bn_data = load_json(DIAG_DIR / "bn_stored_stats_mechanism.json")
    in_data = load_json(DIAG_DIR / "instancenorm_control.json")

    true_labels_path = ROOT / "results" / "baseline_cnn_bn" / "test_labels.npy"
    y_test = np.load(true_labels_path)
    counts = np.bincount(y_test, minlength=4)
    true_prevalence = counts / counts.sum() * 100

    stored_frac = np.array(
        bn_data["predicted_class_distribution_5db"]["mode1_stored_stats"]["predicted_class_frac"]
    ) * 100
    batch_frac = np.array(
        bn_data["predicted_class_distribution_5db"]["mode2_batch_stats"]["predicted_class_frac"]
    ) * 100
    in_frac = np.array(
        in_data["cnn_in_predicted_class_distribution"]["5"]["predicted_class_frac"]
    ) * 100

    print("\n=== FIGURE C: fig_prediction_collapse (5 dB SNR) ===")
    for i, cls in enumerate(CLASS_NAMES):
        print(
            f"{cls}: true={true_prevalence[i]:.2f}% "
            f"CNN-BN(stored)={stored_frac[i]:.2f}% "
            f"CNN-BN(test-time)={batch_frac[i]:.2f}% "
            f"CNN-IN={in_frac[i]:.2f}%"
        )

    x_pos = np.arange(len(CLASS_NAMES))
    width = 0.2

    fig, ax = plt.subplots(figsize=(5.4, 3.8))
    ax.bar(x_pos - 1.5 * width, true_prevalence, width,
           label="true prevalence", color="tab:grey")
    ax.bar(x_pos - 0.5 * width, stored_frac, width,
           label="CNN-BN, stored statistics", color="tab:red")
    ax.bar(x_pos + 0.5 * width, batch_frac, width,
           label="CNN-BN, test-time statistics", color="tab:orange")
    ax.bar(x_pos + 1.5 * width, in_frac, width,
           label="CNN-IN", color="tab:blue")

    ax.set_xticks(x_pos)
    ax.set_xticklabels(CLASS_NAMES)
    ax.set_xlabel("Class")
    ax.set_ylabel("Fraction of predictions (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Predicted class distribution at 5 dB SNR")
    ax.legend(loc="upper right", frameon=True, fontsize=8)
    fig.tight_layout()

    save_fig(fig, "fig_prediction_collapse")
    plt.close(fig)


if __name__ == "__main__":
    make_fig_multiseed_accuracy()
    make_fig_bn_mechanism()
    make_fig_prediction_collapse()
