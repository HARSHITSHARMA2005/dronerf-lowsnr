"""DECISIVE CONTROL EXPERIMENT -- pooling-axis hypothesis for BatchNorm collapse.

The BN-staleness hypothesis is dead (results/diagnostics/bn_shift.json: CNN-BN's
normalised statistics divergence stays below 0.5 at all tested SNRs, while
MLP-BN's raw drift is ~7x larger yet MLP-BN survives -- 78.43% at 5 dB -- while
CNN-BN collapses -- 14.17% at 5 dB). New hypothesis: the problem is the POOLING
AXIS, not the stored statistics. BatchNorm1d in a conv net pools per-channel
across all 2048 frequency positions. The DroneRF log-spectrum is sparse -- a
few high-power bins carry the drone signature, most positions sit near the
floor. An additive noise floor lifts thousands of low-power positions, and
those dominate the per-channel pooled statistic, corrupting normalisation for
the informative positions too. MLP-BN pools over learned global projections
instead, where a uniform noise floor is largely a constant offset that
BatchNorm absorbs.

InstanceNorm1d pools across positions exactly like BatchNorm (same axis) but
computes statistics per-sample at inference with NO stored running statistics.
If pooling-axis is the cause, InstanceNorm-CNN (CNN-IN) should collapse like
CNN-BN. If staleness were the cause (already ruled out), InstanceNorm should be
fine, like GroupNorm.

This script:
  1. Evaluates CNN-IN across the SNR grid {clean, 20, 15, 10, 5, 0, -5, -10,
     -15} dB using add_noise_floor from eval/snr_degradation.py, 5 seeds each.
     Reports accuracy and uncalibrated ECE (no temperature scaling -- CNN-IN
     was never calibration-fitted, and this control is about the pooling-axis
     mechanism, not calibration).
  2. Prints a 3-way comparison table: CNN-GroupNorm vs CNN-BN vs CNN-IN,
     accuracy and ECE at every SNR (GroupNorm/BN loaded from
     results/snr_degradation_v2/degradation_metrics.json).
  3. Reports CNN-IN's predicted-class distribution at 5 dB and 0 dB, to check
     for the same Phantom-funnelling seen in CNN-BN (65.67% of predictions
     into a 9.25%-prevalence class).
  4. ADDITIONAL DIAGNOSTIC (CNN-BN only): splits the 2048 frequency positions
     into high-power (top 10%) and low-power (bottom 50%) groups by clean
     training-set mean power per position, then reports BN input mean/variance
     per group per BN layer at clean, 10, 5, 0 dB. Prediction under the
     pooling-axis hypothesis: at clean the two groups have distinguishable
     statistics; as the noise floor rises, the low-power group's statistics
     shift substantially and, since it is the majority of positions, dominates
     the pooled per-channel statistic.

Run with: python eval\\instancenorm_control.py   (after activating venv)
Requires: results/baseline_cnn_in/model.pt (train/train_cnn_in.py must be run
first), results/baseline_cnn/model.pt, results/baseline_cnn_bn/model.pt,
results/snr_degradation_v2/degradation_metrics.json,
data/processed/{X_train_raw,X_test_raw,y_test}.npy, data/processed/scaler.joblib.
Do NOT run this automatically -- the user runs it manually.
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "eval"))
sys.path.insert(0, str(REPO_ROOT / "train"))
import snr_degradation as sd  # noqa: E402  (add_noise_floor, preprocess_raw, compute_ece, softmax)
from utils import PROCESSED_DIR, RESULTS_DIR  # noqa: E402
from train_cnn_in import CNN1DIN  # noqa: E402
from train_cnn_bn import CNN1DBN  # noqa: E402

OUT_DIR = RESULTS_DIR / "diagnostics"
SNR_DEGRADATION_V2_PATH = RESULTS_DIR / "snr_degradation_v2" / "degradation_metrics.json"
SNR_LEVELS_DB = [20, 15, 10, 5, 0, -5, -10, -15]
NOISE_SEEDS = [0, 1, 2, 3, 4]
CLASS_NAMES = ["background", "Bebop", "AR", "Phantom"]
CLASS_PREVALENCE = {0: 0.1807, 1: 0.3700, 2: 0.3568, 3: 0.0925}  # 4100/8400/8100/2100 of 22700

HIGH_POWER_FRAC = 0.10
LOW_POWER_FRAC = 0.50
GROUP_CONDITIONS = ["clean", 10, 5, 0]


# ---------------------------------------------------------------------------
# Model loading / inference
# ---------------------------------------------------------------------------

def load_torch_model(model_class, result_dir_name):
    device = torch.device("cpu")
    model = model_class().to(device)
    model.load_state_dict(torch.load(RESULTS_DIR / result_dir_name / "model.pt", map_location=device))
    model.eval()
    return model


def torch_infer(model, X):
    X_in = X.reshape(X.shape[0], 1, -1)
    with torch.no_grad():
        logits = model(torch.from_numpy(X_in).float()).numpy()
    return logits


# ---------------------------------------------------------------------------
# PART 1 -- CNN-IN SNR sweep (accuracy + uncalibrated ECE)
# ---------------------------------------------------------------------------

def evaluate_at_level(logits, labels):
    probs = sd.softmax(logits, axis=1)
    return {
        "accuracy": sd.compute_accuracy(probs, labels),
        "ece_uncalibrated": sd.compute_ece(probs, labels),
    }


def aggregate(metric_dicts):
    keys = metric_dicts[0].keys()
    agg = {}
    for k in keys:
        vals = np.array([m[k] for m in metric_dicts])
        agg[f"{k}_mean"] = float(vals.mean())
        agg[f"{k}_std"] = float(vals.std())
    return agg


def run_cnn_in_sweep(cnn_in, X_test_raw, y_test, scaler):
    results = {}

    print(f"\n{'=' * 60}\nCNN-IN: CLEAN (reference)\n{'=' * 60}")
    X_clean = sd.preprocess_raw(X_test_raw, scaler)
    logits = torch_infer(cnn_in, X_clean)
    metrics = evaluate_at_level(logits, y_test)
    results["clean"] = metrics
    print(f"  acc={metrics['accuracy'] * 100:.2f}%  ECE_uncal={metrics['ece_uncalibrated']:.4f}")

    for snr_db in SNR_LEVELS_DB:
        print(f"\n{'=' * 60}\nCNN-IN: SNR = {snr_db} dB\n{'=' * 60}")
        runs = []
        for seed in NOISE_SEEDS:
            rng = np.random.default_rng(seed)
            X_noisy_raw = sd.add_noise_floor(X_test_raw, snr_db, rng)
            X_noisy = sd.preprocess_raw(X_noisy_raw, scaler, snr_db=snr_db)
            logits = torch_infer(cnn_in, X_noisy)
            runs.append(evaluate_at_level(logits, y_test))
        agg = aggregate(runs)
        results[str(snr_db)] = agg
        print(f"  acc={agg['accuracy_mean'] * 100:.2f}%±{agg['accuracy_std'] * 100:.2f} "
              f"ECE_uncal={agg['ece_uncalibrated_mean']:.4f}±{agg['ece_uncalibrated_std']:.4f}")

    return results


# ---------------------------------------------------------------------------
# PART 2 -- 3-way comparison table (GroupNorm / BN / IN)
# ---------------------------------------------------------------------------

def print_3way_comparison(cnn_gn_results, cnn_bn_results, cnn_in_results):
    print(f"\n{'=' * 92}\n3-WAY COMPARISON: CNN-GroupNorm vs CNN-BN vs CNN-IN "
          f"(accuracy / uncalibrated ECE)\n{'=' * 92}")
    header = (f"  {'SNR (dB)':>10} | {'GN acc%':>8} {'BN acc%':>8} {'IN acc%':>8} | "
              f"{'GN ECE':>7} {'BN ECE':>7} {'IN ECE':>7}")
    print(header)
    print("  " + "-" * (len(header) - 2))

    def row_gn_bn(results, key):
        acc = results[key]["accuracy_mean"] * 100
        ece = results[key]["ece_uncalibrated_mean"]
        return acc, ece

    # clean row
    gn_acc, gn_ece = cnn_gn_results["clean"]["accuracy"] * 100, cnn_gn_results["clean"]["ece_uncalibrated"]
    bn_acc, bn_ece = cnn_bn_results["clean"]["accuracy"] * 100, cnn_bn_results["clean"]["ece_uncalibrated"]
    in_acc, in_ece = cnn_in_results["clean"]["accuracy"] * 100, cnn_in_results["clean"]["ece_uncalibrated"]
    print(f"  {'clean':>10} | {gn_acc:>8.2f} {bn_acc:>8.2f} {in_acc:>8.2f} | "
          f"{gn_ece:>7.4f} {bn_ece:>7.4f} {in_ece:>7.4f}")

    for snr_db in SNR_LEVELS_DB:
        key = str(snr_db)
        gn_acc, gn_ece = row_gn_bn(cnn_gn_results, key)
        bn_acc, bn_ece = row_gn_bn(cnn_bn_results, key)
        in_acc = cnn_in_results[key]["accuracy_mean"] * 100
        in_ece = cnn_in_results[key]["ece_uncalibrated_mean"]
        print(f"  {snr_db:>10d} | {gn_acc:>8.2f} {bn_acc:>8.2f} {in_acc:>8.2f} | "
              f"{gn_ece:>7.4f} {bn_ece:>7.4f} {in_ece:>7.4f}")


# ---------------------------------------------------------------------------
# PART 3 -- CNN-IN predicted-class distribution (Phantom-funnelling check)
# ---------------------------------------------------------------------------

def predicted_class_distribution(cnn_in, X_test_raw, scaler, snr_levels=(5, 0), seed=0):
    out = {}
    for snr_db in snr_levels:
        rng = np.random.default_rng(seed)
        X_noisy_raw = sd.add_noise_floor(X_test_raw, snr_db, rng)
        X_noisy = sd.preprocess_raw(X_noisy_raw, scaler, snr_db=snr_db)
        logits = torch_infer(cnn_in, X_noisy)
        probs = sd.softmax(logits, axis=1)
        preds = probs.argmax(axis=1)
        n = X_noisy.shape[0]
        counts = [int((preds == c).sum()) for c in range(probs.shape[1])]
        out[snr_db] = {"n_samples": n, "predicted_class_counts": counts,
                        "predicted_class_frac": [c / n for c in counts]}
    return out


def print_predicted_class_distribution(results):
    print(f"\n{'=' * 78}\nCNN-IN: predicted-class distribution "
          f"(Phantom-funnelling check)\n{'=' * 78}")
    for snr_db, d in results.items():
        print(f"\n  SNR = {snr_db} dB (n={d['n_samples']})")
        print(f"    {'class':<12}{'prevalence':>12}{'pred_count':>12}{'pred_frac':>12}")
        for c, name in enumerate(CLASS_NAMES):
            print(f"    {name:<12}{CLASS_PREVALENCE[c] * 100:>11.2f}%"
                  f"{d['predicted_class_counts'][c]:>12d}{d['predicted_class_frac'][c] * 100:>11.2f}%")


# ---------------------------------------------------------------------------
# PART 4 -- CNN-BN high-power vs low-power position group diagnostic
# ---------------------------------------------------------------------------

def compute_power_groups(X_train_raw, high_frac=HIGH_POWER_FRAC, low_frac=LOW_POWER_FRAC):
    """Splits the 2048 frequency positions into high-power (top `high_frac`)
    and low-power (bottom `low_frac`) groups, ranked by clean training-set
    mean power per position (raw, pre-log power spectrum)."""
    mean_power = X_train_raw.mean(axis=0)  # (2048,)
    n_pos = mean_power.shape[0]
    order = np.argsort(mean_power)[::-1]  # descending power
    n_high = int(round(n_pos * high_frac))
    n_low = int(round(n_pos * low_frac))
    high_idx = order[:n_high]
    low_idx = order[-n_low:]
    print(f"  Position groups: high-power = top {n_high}/{n_pos} positions "
          f"(mean power {mean_power[high_idx].mean():.4g}), "
          f"low-power = bottom {n_low}/{n_pos} positions "
          f"(mean power {mean_power[low_idx].mean():.4g})")
    return high_idx, low_idx


class GroupInputHook:
    """Forward-pre-hook that records the BN layer's input, split into the
    high-power and low-power position groups, per batch."""

    def __init__(self, high_idx, low_idx):
        self.high_idx = high_idx
        self.low_idx = low_idx
        self.high_sum = None
        self.high_sumsq = None
        self.high_count = 0
        self.low_sum = None
        self.low_sumsq = None
        self.low_count = 0
        self._length = None  # conv-layer sequence length, to map raw-position idx -> layer-position idx

    def __call__(self, module, inp):
        x = inp[0].detach()  # (N, C, L)
        length = x.shape[2]
        if self._length is None:
            self._length = length
        # Map the 2048 raw-frequency-position indices down to this layer's
        # sequence length via integer scaling (positions pool by factor
        # 2048 / L as the network downsamples with MaxPool1d(2)).
        scale = length / 2048.0
        high_pos = np.unique((self.high_idx * scale).astype(np.int64))
        high_pos = high_pos[high_pos < length]
        low_pos = np.unique((self.low_idx * scale).astype(np.int64))
        low_pos = low_pos[low_pos < length]

        x_high = x[:, :, high_pos]
        x_low = x[:, :, low_pos]

        s_h = x_high.sum(dim=(0, 2))
        ss_h = (x_high ** 2).sum(dim=(0, 2))
        n_h = x_high.shape[0] * x_high.shape[2]
        s_l = x_low.sum(dim=(0, 2))
        ss_l = (x_low ** 2).sum(dim=(0, 2))
        n_l = x_low.shape[0] * x_low.shape[2]

        if self.high_sum is None:
            self.high_sum, self.high_sumsq = s_h.clone(), ss_h.clone()
            self.low_sum, self.low_sumsq = s_l.clone(), ss_l.clone()
        else:
            self.high_sum += s_h
            self.high_sumsq += ss_h
            self.low_sum += s_l
            self.low_sumsq += ss_l
        self.high_count += n_h
        self.low_count += n_l

    def stats(self):
        high_mean = (self.high_sum / self.high_count)
        high_var = (self.high_sumsq / self.high_count - high_mean ** 2)
        low_mean = (self.low_sum / self.low_count)
        low_var = (self.low_sumsq / self.low_count - low_mean ** 2)
        return {
            "high_power_mean": float(high_mean.mean().item()),
            "high_power_var": float(high_var.mean().item()),
            "low_power_mean": float(low_mean.mean().item()),
            "low_power_var": float(low_var.mean().item()),
        }


def get_bn_layers(model):
    return [(name, m) for name, m in model.named_modules() if isinstance(m, nn.BatchNorm1d)]


def forward_full(model, X, batch_size=256):
    model.eval()
    with torch.no_grad():
        for i in range(0, X.shape[0], batch_size):
            xb = torch.from_numpy(X[i:i + batch_size]).float().reshape(-1, 1, X.shape[1])
            model(xb)


def cnn_bn_group_diagnostic(cnn_bn, X_train_raw, X_test_raw, y_test, scaler):
    print(f"\n{'=' * 78}\nADDITIONAL DIAGNOSTIC: CNN-BN high-power vs. low-power "
          f"position group statistics\n{'=' * 78}")
    high_idx, low_idx = compute_power_groups(X_train_raw)

    bn_layers = get_bn_layers(cnn_bn)
    results = {li: {} for li in range(len(bn_layers))}

    for cond in GROUP_CONDITIONS:
        if cond == "clean":
            X = sd.preprocess_raw(X_test_raw, scaler)
        else:
            rng = np.random.default_rng(0)
            X_noisy_raw = sd.add_noise_floor(X_test_raw, cond, rng)
            X = sd.preprocess_raw(X_noisy_raw, scaler, snr_db=cond)

        hooks_obj = [GroupInputHook(high_idx, low_idx) for _ in bn_layers]
        handles = [bn.register_forward_pre_hook(h) for (_, bn), h in zip(bn_layers, hooks_obj)]
        forward_full(cnn_bn, X)
        for handle in handles:
            handle.remove()

        for li, ((name, _), h) in enumerate(zip(bn_layers, hooks_obj)):
            results[li][str(cond)] = {"layer_name": name, **h.stats()}

    print(f"\n  {'Layer':<20}{'Condition':>10}{'high_mean':>12}{'high_var':>12}"
          f"{'low_mean':>12}{'low_var':>12}")
    for li in sorted(results.keys()):
        for cond in GROUP_CONDITIONS:
            r = results[li][str(cond)]
            print(f"  [{li}] {r['layer_name']:<15}{str(cond):>10}{r['high_power_mean']:>12.4f}"
                  f"{r['high_power_var']:>12.4f}{r['low_power_mean']:>12.4f}{r['low_power_var']:>12.4f}")

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    X_train_raw = np.load(PROCESSED_DIR / "X_train_raw.npy")
    X_test_raw = np.load(PROCESSED_DIR / "X_test_raw.npy")
    y_test = np.load(PROCESSED_DIR / "y_test.npy")
    scaler = joblib.load(PROCESSED_DIR / "scaler.joblib")

    cnn_in = load_torch_model(CNN1DIN, "baseline_cnn_in")
    cnn_bn = load_torch_model(CNN1DBN, "baseline_cnn_bn")

    # --- Part 1: CNN-IN SNR sweep -------------------------------------------
    cnn_in_results = run_cnn_in_sweep(cnn_in, X_test_raw, y_test, scaler)

    with open(SNR_DEGRADATION_V2_PATH) as f:
        v2_results = json.load(f)
    cnn_gn_results = v2_results["cnn"]
    cnn_bn_results = v2_results["cnn_bn"]

    # --- Part 2: 3-way comparison table -------------------------------------
    print_3way_comparison(cnn_gn_results, cnn_bn_results, cnn_in_results)

    # --- Part 3: CNN-IN predicted-class distribution ------------------------
    pred_dist = predicted_class_distribution(cnn_in, X_test_raw, scaler, snr_levels=(5, 0))
    print_predicted_class_distribution(pred_dist)

    # --- Part 4: CNN-BN high/low power group diagnostic ---------------------
    group_results = cnn_bn_group_diagnostic(cnn_bn, X_train_raw, X_test_raw, y_test, scaler)

    out = {
        "cnn_in_snr_sweep": cnn_in_results,
        "cnn_in_predicted_class_distribution": {str(k): v for k, v in pred_dist.items()},
        "cnn_bn_power_group_diagnostic": group_results,
    }
    out_path = OUT_DIR / "instancenorm_control.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
