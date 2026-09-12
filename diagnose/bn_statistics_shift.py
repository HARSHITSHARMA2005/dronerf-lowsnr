"""DIAGNOSTIC ONLY -- measures whether BatchNorm's stored running statistics
go stale under noise, as a candidate explanation for CNN-BN's below-chance
collapse (14.17% acc at 5 dB, 11.69% at 0 dB) under the corrected noise-floor
model in results/snr_degradation_v2/, versus the architecturally identical
GroupNorm CNN's smooth degradation (67.45% at 5 dB, 39.20% at 0 dB).

Hypothesis under test: BatchNorm1d layers normalise at inference using
running_mean/running_var estimated on CLEAN training data. Under noise the
true activation statistics shift, so the model normalises with stale
parameters, and this error compounds layer by layer. GroupNorm normalises
per-sample at inference (no stored statistics), so it cannot go stale the
same way.

This script does NOT modify any model, implement any fix, or change
eval/snr_degradation.py. It only measures and reports numbers.

Run with: python diagnose\\bn_statistics_shift.py   (after activating venv)
Requires: data/processed/{X_test_raw,y_test}.npy, data/processed/scaler.joblib,
results/baseline_cnn_bn/model.pt, results/baseline_mlp_bn/model.pt,
results/baseline_cnn/model.pt.
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
import snr_degradation as sd  # noqa: E402  (add_noise_floor, preprocess_raw, LOG_EPSILON)
from utils import PROCESSED_DIR, RESULTS_DIR  # noqa: E402
from train_cnn_bn import CNN1DBN  # noqa: E402
from train_mlp_bn import MLPBN  # noqa: E402
from train_cnn import CNN1D  # noqa: E402

OUT_DIR = RESULTS_DIR / "diagnostics"
NOISE_SEED = 0
BATCH_SIZE = 256
CONDITIONS = ["clean", "20", "10", "5", "0"]  # SNR dB, "clean" = no noise
EXTREME_THRESHOLD = 10.0


# ---------------------------------------------------------------------------
# Input construction (mirrors eval/snr_degradation.py's pipeline exactly)
# ---------------------------------------------------------------------------

def build_condition_inputs(X_test_raw, scaler):
    """Returns {condition_str: X_preprocessed} using the exact same
    add_noise_floor + preprocess_raw pipeline as the SNR sweep. One fixed
    seed per condition, so results are reproducible across runs and shared
    identically between the CNN-BN, MLP-BN, and GroupNorm-CNN evaluations
    below (same noisy input feeds all three models at a given SNR)."""
    inputs = {}
    for cond in CONDITIONS:
        if cond == "clean":
            inputs[cond] = sd.preprocess_raw(X_test_raw, scaler)
        else:
            rng = np.random.default_rng(NOISE_SEED)
            x_noisy_raw = sd.add_noise_floor(X_test_raw, int(cond), rng)
            inputs[cond] = sd.preprocess_raw(x_noisy_raw, scaler, snr_db=cond)
    return inputs


# ---------------------------------------------------------------------------
# Model forward helpers
# ---------------------------------------------------------------------------

def load_torch_model(model_class, result_dir_name):
    device = torch.device("cpu")
    model = model_class().to(device)
    model.load_state_dict(torch.load(RESULTS_DIR / result_dir_name / "model.pt", map_location=device))
    model.eval()
    return model


def iter_batches(X, batch_size=BATCH_SIZE):
    for i in range(0, X.shape[0], batch_size):
        yield X[i:i + batch_size]


def to_tensor(Xb, reshape_conv):
    t = torch.from_numpy(Xb).float()
    if reshape_conv:
        t = t.reshape(t.shape[0], 1, -1)
    return t


def forward_full(model, X, reshape_conv, batch_size=BATCH_SIZE):
    """Runs model over X in batches, eval() mode, no_grad. Any hooks
    registered on the model's submodules fire during these calls. Returns
    concatenated logits as a numpy array."""
    model.eval()
    outs = []
    with torch.no_grad():
        for Xb in iter_batches(X, batch_size):
            xb = to_tensor(Xb, reshape_conv)
            logits = model(xb)
            outs.append(logits.numpy())
    return np.concatenate(outs, axis=0)


def softmax(x, axis=1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def get_layers(model, cls):
    """Ordered list of (qualified_name, module) for all submodules of type cls."""
    return [(name, m) for name, m in model.named_modules() if isinstance(m, cls)]


# ---------------------------------------------------------------------------
# Streaming per-channel accumulator (avoids holding full activation tensors
# for the whole test set in memory at once)
# ---------------------------------------------------------------------------

class ChannelAccumulator:
    def __init__(self):
        self.sum = None
        self.sumsq = None
        self.count = 0
        self.extreme = 0
        self.total = 0

    def update(self, x):
        # x: (N, C, L) for conv layers, (N, C) for linear layers
        dims = (0, 2) if x.dim() == 3 else (0,)
        n = x.shape[0] * (x.shape[2] if x.dim() == 3 else 1)
        s = x.sum(dim=dims)
        ss = (x ** 2).sum(dim=dims)
        if self.sum is None:
            self.sum = s.clone()
            self.sumsq = ss.clone()
        else:
            self.sum += s
            self.sumsq += ss
        self.count += n
        self.extreme += int((x.abs() > EXTREME_THRESHOLD).sum().item())
        self.total += x.numel()

    def mean(self):
        return (self.sum / self.count).numpy()

    def var(self):
        # population (biased) variance from accumulated sums
        m = self.sum / self.count
        return (self.sumsq / self.count - m ** 2).numpy()

    def extreme_frac(self):
        return self.extreme / self.total if self.total else float("nan")


def make_pre_hook(acc):
    def hook(module, inp):
        acc.update(inp[0].detach())
    return hook


def make_post_hook(acc):
    def hook(module, inp, out):
        acc.update(out.detach())
    return hook


# ---------------------------------------------------------------------------
# MEASUREMENT 1 -- stored vs actual BN input statistics
# ---------------------------------------------------------------------------

def measurement1_bn_input_stats(model, reshape_conv, conditions_inputs, conditions=CONDITIONS):
    bn_layers = get_layers(model, nn.BatchNorm1d)
    results = {li: {} for li in range(len(bn_layers))}

    for cond in conditions:
        X = conditions_inputs[cond]
        accs = [ChannelAccumulator() for _ in bn_layers]
        hooks = [bn.register_forward_pre_hook(make_pre_hook(acc))
                 for (_, bn), acc in zip(bn_layers, accs)]
        forward_full(model, X, reshape_conv)
        for h in hooks:
            h.remove()

        for li, ((name, bn), acc) in enumerate(zip(bn_layers, accs)):
            actual_mean = acc.mean()
            actual_var = acc.var()
            running_mean = bn.running_mean.detach().numpy()
            running_var = bn.running_var.detach().numpy()
            eps = bn.eps

            mean_abs_diff = float(np.mean(np.abs(actual_mean - running_mean)))
            var_ratio_median = float(np.median(actual_var / running_var))
            norm_divergence = float(np.mean(
                np.abs(actual_mean - running_mean) / np.sqrt(running_var + eps)))

            results[li][cond] = {
                "layer_name": name,
                "num_channels": int(running_mean.shape[0]),
                "mean_abs_diff": mean_abs_diff,
                "var_ratio_median": var_ratio_median,
                "norm_divergence": norm_divergence,
            }
    return results


def print_measurement1_table(model_name, results, conditions=CONDITIONS):
    metrics = [
        ("mean_abs_diff", "mean |actual_mean - running_mean|"),
        ("var_ratio_median", "median(actual_var / running_var)"),
        ("norm_divergence", "mean |actual_mean-running_mean| / sqrt(running_var+eps)  <-- KEY NUMBER"),
    ]
    print(f"\n{'=' * 78}\nMEASUREMENT 1 -- {model_name}: stored vs. actual BN input statistics\n{'=' * 78}")
    for key, label in metrics:
        print(f"\n  {label}")
        header = f"    {'BN layer':<10}" + "".join(f"{c:>12}" for c in conditions)
        print(header)
        for li in sorted(results.keys()):
            row = f"    [{li}] {results[li][conditions[0]]['layer_name']:<12}"[:22].ljust(14)
            for c in conditions:
                row += f"{results[li][c][key]:>12.4f}"
            print(row)


# ---------------------------------------------------------------------------
# MEASUREMENT 2 / 4 -- layer-by-layer output shift under noise (generic:
# works for BatchNorm1d in CNN-BN, or GroupNorm in the GroupNorm-CNN control)
# ---------------------------------------------------------------------------

def measurement_output_shift(model, reshape_conv, conditions_inputs, norm_cls, snr_levels=("5", "0")):
    norm_layers = get_layers(model, norm_cls)
    out = {}

    X_clean = conditions_inputs["clean"]
    for snr in snr_levels:
        X_noisy = conditions_inputs[snr]

        accs_clean = [ChannelAccumulator() for _ in norm_layers]
        hooks = [nl.register_forward_hook(make_post_hook(a))
                 for (_, nl), a in zip(norm_layers, accs_clean)]
        forward_full(model, X_clean, reshape_conv)
        for h in hooks:
            h.remove()

        accs_noisy = [ChannelAccumulator() for _ in norm_layers]
        hooks = [nl.register_forward_hook(make_post_hook(a))
                 for (_, nl), a in zip(norm_layers, accs_noisy)]
        forward_full(model, X_noisy, reshape_conv)
        for h in hooks:
            h.remove()

        layer_rows = []
        for li, ((name, nl), ac, an) in enumerate(zip(norm_layers, accs_clean, accs_noisy)):
            clean_mean_overall = float(np.mean(ac.mean()))
            clean_std_overall = float(np.sqrt(max(np.mean(ac.var()), 0.0)))
            noisy_mean_overall = float(np.mean(an.mean()))
            noisy_std_overall = float(np.sqrt(max(np.mean(an.var()), 0.0)))
            layer_rows.append({
                "layer_index": li,
                "layer_name": name,
                "clean_mean": clean_mean_overall,
                "clean_std": clean_std_overall,
                "noisy_mean": noisy_mean_overall,
                "noisy_std": noisy_std_overall,
                "mean_shift": abs(noisy_mean_overall - clean_mean_overall),
                "std_ratio_noisy_over_clean": (noisy_std_overall / clean_std_overall
                                                if clean_std_overall > 0 else float("nan")),
                "frac_extreme_clean": ac.extreme_frac(),
                "frac_extreme_noisy": an.extreme_frac(),
            })
        out[snr] = layer_rows
    return out


def print_output_shift_table(model_name, results, snr_levels=("5", "0")):
    print(f"\n{'=' * 78}\nMEASUREMENT -- {model_name}: layer-by-layer output shift under noise\n{'=' * 78}")
    for snr in snr_levels:
        print(f"\n  SNR = {snr} dB")
        header = (f"    {'Layer':<8}{'clean_mean':>12}{'clean_std':>12}{'noisy_mean':>12}"
                  f"{'noisy_std':>12}{'mean_shift':>12}{'std_ratio':>11}{'frac_ext_c':>11}{'frac_ext_n':>11}")
        print(header)
        for row in results[snr]:
            print(f"    [{row['layer_index']}] {row['layer_name']:<6}"[:12].ljust(12)
                  + f"{row['clean_mean']:>12.4f}{row['clean_std']:>12.4f}{row['noisy_mean']:>12.4f}"
                    f"{row['noisy_std']:>12.4f}{row['mean_shift']:>12.4f}"
                    f"{row['std_ratio_noisy_over_clean']:>11.4f}"
                    f"{row['frac_extreme_clean'] * 100:>10.2f}%{row['frac_extreme_noisy'] * 100:>10.2f}%")


# ---------------------------------------------------------------------------
# MEASUREMENT 3 -- prediction collapse check
# ---------------------------------------------------------------------------

def measurement3_prediction_collapse(model, reshape_conv, conditions_inputs, snr_levels=("5", "0")):
    out = {}
    for snr in snr_levels:
        X = conditions_inputs[snr]
        logits = forward_full(model, X, reshape_conv)
        probs = softmax(logits)
        preds = probs.argmax(axis=1)
        counts = [int((preds == c).sum()) for c in range(probs.shape[1])]
        mean_prob_per_class = probs.mean(axis=0).tolist()
        out[snr] = {
            "n_samples": int(X.shape[0]),
            "predicted_class_counts": counts,
            "mean_predicted_probability_per_class": mean_prob_per_class,
        }
    return out


def print_prediction_collapse(model_name, results, class_names, snr_levels=("5", "0")):
    print(f"\n{'=' * 78}\nMEASUREMENT 3 -- {model_name}: prediction collapse check\n{'=' * 78}")
    for snr in snr_levels:
        d = results[snr]
        print(f"\n  SNR = {snr} dB (n={d['n_samples']})")
        print(f"    {'class':<12}{'pred_count':>12}{'pred_frac':>12}{'mean_prob':>12}")
        for c, name in enumerate(class_names):
            count = d["predicted_class_counts"][c]
            frac = count / d["n_samples"]
            mean_prob = d["mean_predicted_probability_per_class"][c]
            print(f"    {name:<12}{count:>12d}{frac * 100:>11.2f}%{mean_prob:>12.4f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CLASS_NAMES = ["background", "Bebop", "AR", "Phantom"]

    X_test_raw = np.load(PROCESSED_DIR / "X_test_raw.npy")
    y_test = np.load(PROCESSED_DIR / "y_test.npy")
    scaler = joblib.load(PROCESSED_DIR / "scaler.joblib")
    print(f"Loaded X_test_raw={X_test_raw.shape}, y_test={y_test.shape}")

    print(f"\nBuilding preprocessed inputs for conditions {CONDITIONS} "
          f"(add_noise_floor, seed={NOISE_SEED}, matches eval/snr_degradation.py exactly)...")
    conditions_inputs = build_condition_inputs(X_test_raw, scaler)

    print("\nLoading models (eval mode)...")
    cnn_bn = load_torch_model(CNN1DBN, "baseline_cnn_bn")
    mlp_bn = load_torch_model(MLPBN, "baseline_mlp_bn")
    cnn_gn = load_torch_model(CNN1D, "baseline_cnn")

    all_results = {}

    # --- MEASUREMENT 1: stored vs actual BN input statistics ----------------
    m1_cnn_bn = measurement1_bn_input_stats(cnn_bn, reshape_conv=True, conditions_inputs=conditions_inputs)
    print_measurement1_table("CNN-BN", m1_cnn_bn)
    all_results["measurement1_cnn_bn"] = m1_cnn_bn

    m1_mlp_bn = measurement1_bn_input_stats(mlp_bn, reshape_conv=False, conditions_inputs=conditions_inputs)
    print_measurement1_table("MLP-BN", m1_mlp_bn)
    all_results["measurement1_mlp_bn"] = m1_mlp_bn

    # --- MEASUREMENT 2: layer-by-layer error compounding (CNN-BN only) -----
    m2_cnn_bn = measurement_output_shift(cnn_bn, reshape_conv=True, conditions_inputs=conditions_inputs,
                                          norm_cls=nn.BatchNorm1d, snr_levels=("5", "0"))
    print_output_shift_table("CNN-BN (BatchNorm1d)", m2_cnn_bn)
    all_results["measurement2_cnn_bn"] = m2_cnn_bn

    # --- MEASUREMENT 3: prediction collapse check (CNN-BN only) -------------
    m3_cnn_bn = measurement3_prediction_collapse(cnn_bn, reshape_conv=True, conditions_inputs=conditions_inputs,
                                                  snr_levels=("5", "0"))
    print_prediction_collapse("CNN-BN", m3_cnn_bn, CLASS_NAMES)
    all_results["measurement3_cnn_bn"] = m3_cnn_bn

    # --- MEASUREMENT 4: GroupNorm control -----------------------------------
    m4_cnn_gn = measurement_output_shift(cnn_gn, reshape_conv=True, conditions_inputs=conditions_inputs,
                                          norm_cls=nn.GroupNorm, snr_levels=("5", "0"))
    print_output_shift_table("CNN (GroupNorm) -- control", m4_cnn_gn)
    all_results["measurement4_cnn_groupnorm_control"] = m4_cnn_gn

    metrics_path = OUT_DIR / "bn_shift.json"
    with open(metrics_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved: {metrics_path}")


if __name__ == "__main__":
    main()
