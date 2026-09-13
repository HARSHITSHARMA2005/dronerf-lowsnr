"""CHECK B -- tests whether CNN-BN's stored running_mean/running_var are
actually THE mechanism behind its 5 dB collapse (14.17% acc), as opposed to
some other effect (e.g. the pooling-axis hypothesis already under test in
eval/instancenorm_control.py, which does not depend on staleness at all).

Loads the ALREADY-TRAINED seed-42 CNN-BN from results/baseline_cnn_bn/model.pt
-- no retraining -- and evaluates it at clean/10/5/0 dB in two modes:

  Mode 1 (normal):     model.eval() -- every BatchNorm1d uses its stored
                        running_mean/running_var (estimated on clean training
                        data).
  Mode 2 (batch stats): every BatchNorm1d module is switched to .train() mode
                        so it normalises using batch statistics computed from
                        the CURRENT (noisy) test batch instead of the stored
                        running stats. The whole forward pass runs under
                        torch.no_grad() (no gradient updates -- weights are
                        frozen, this is inference only), and running_mean/
                        running_var/num_batches_tracked are snapshotted before
                        and restored after each batch so the saved model is
                        left byte-identical to how it was loaded (BN.train()
                        still updates the running buffers as a side effect of
                        the forward pass even with no_grad, so an explicit
                        snapshot/restore is used rather than relying on
                        momentum=0, which would not stop the buffers from
                        being touched by the running mean/var accumulation
                        formula on the first pass).
                        A large batch (the full test set, ~3405 samples) is
                        used so the batch statistics are stable, matching the
                        "batches of at least 512" requirement.

Interpretation: if Mode 2 accuracy at 5 dB jumps substantially toward
CNN-IN's ~80% (which uses per-sample instance statistics, structurally
similar to a large-batch BN batch-statistics computation), stored statistics
are (at least part of) the mechanism. If Mode 2 stays near Mode 1's ~14%,
stored statistics are NOT the mechanism, and the pooling-axis explanation
(already supported by eval/instancenorm_control.py) stands uncontradicted.

Run with: python diagnose\\bn_stored_stats_mechanism.py   (after activating venv)
Requires: results/baseline_cnn_bn/model.pt, data/processed/{X_test_raw,y_test}.npy,
data/processed/scaler.joblib.
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
import snr_degradation as sd  # noqa: E402  (add_noise_floor, preprocess_raw, softmax, compute_ece)
from utils import PROCESSED_DIR, RESULTS_DIR  # noqa: E402
from train_cnn_bn import CNN1DBN  # noqa: E402

OUT_DIR = RESULTS_DIR / "diagnostics"
SNR_LEVELS = ["clean", 10, 5, 0]
NOISE_SEEDS = [0, 1, 2, 3, 4]
CLASS_NAMES = ["background", "Bebop", "AR", "Phantom"]


def load_cnn_bn():
    device = torch.device("cpu")
    model = CNN1DBN().to(device)
    model.load_state_dict(torch.load(RESULTS_DIR / "baseline_cnn_bn" / "model.pt", map_location=device))
    model.eval()
    return model


def get_bn_layers(model):
    return [m for m in model.modules() if isinstance(m, nn.BatchNorm1d)]


def snapshot_bn_buffers(bn_layers):
    return [
        (bn.running_mean.clone(), bn.running_var.clone(), bn.num_batches_tracked.clone())
        for bn in bn_layers
    ]


def restore_bn_buffers(bn_layers, snapshot):
    for bn, (rm, rv, nbt) in zip(bn_layers, snapshot):
        bn.running_mean.copy_(rm)
        bn.running_var.copy_(rv)
        bn.num_batches_tracked.copy_(nbt)


def infer_mode1_normal(model, X):
    """Mode 1: standard eval() -- BN uses stored running stats."""
    model.eval()
    X_in = X.reshape(X.shape[0], 1, -1)
    with torch.no_grad():
        logits = model(torch.from_numpy(X_in).float()).numpy()
    return logits


def infer_mode2_batch_stats(model, X):
    """Mode 2: BN layers use batch statistics from the current (noisy) input,
    computed over the full test set as one batch. The model's stored running
    buffers are snapshotted before and restored after, so the saved model is
    unmodified by this diagnostic."""
    bn_layers = get_bn_layers(model)
    snapshot = snapshot_bn_buffers(bn_layers)

    model.eval()
    for bn in bn_layers:
        bn.train()

    X_in = X.reshape(X.shape[0], 1, -1)
    with torch.no_grad():
        logits = model(torch.from_numpy(X_in).float()).numpy()

    for bn in bn_layers:
        bn.eval()
    restore_bn_buffers(bn_layers, snapshot)

    return logits


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


def run_sweep(model, infer_fn, X_test_raw, y_test, scaler):
    results = {}

    X_clean = sd.preprocess_raw(X_test_raw, scaler)
    logits = infer_fn(model, X_clean)
    results["clean"] = evaluate_at_level(logits, y_test)

    for snr_db in SNR_LEVELS:
        if snr_db == "clean":
            continue
        runs = []
        for seed in NOISE_SEEDS:
            rng = np.random.default_rng(seed)
            X_noisy_raw = sd.add_noise_floor(X_test_raw, snr_db, rng)
            X_noisy = sd.preprocess_raw(X_noisy_raw, scaler, snr_db=snr_db)
            logits = infer_fn(model, X_noisy)
            runs.append(evaluate_at_level(logits, y_test))
        results[str(snr_db)] = aggregate(runs)

    return results


def predicted_class_distribution(model, infer_fn, X_test_raw, scaler, snr_db=5, seed=0):
    rng = np.random.default_rng(seed)
    X_noisy_raw = sd.add_noise_floor(X_test_raw, snr_db, rng)
    X_noisy = sd.preprocess_raw(X_noisy_raw, scaler, snr_db=snr_db)
    logits = infer_fn(model, X_noisy)
    probs = sd.softmax(logits, axis=1)
    preds = probs.argmax(axis=1)
    n = X_noisy.shape[0]
    counts = [int((preds == c).sum()) for c in range(probs.shape[1])]
    return {"n_samples": n, "predicted_class_counts": counts,
            "predicted_class_frac": [c / n for c in counts]}


def print_side_by_side(mode1_results, mode2_results):
    print(f"\n{'=' * 90}\nMode 1 (stored running stats) vs Mode 2 (current-batch "
          f"stats), CNN-BN seed 42\n{'=' * 90}")
    header = (f"  {'SNR':>8} | {'Mode1 acc%':>11} {'Mode2 acc%':>11} | "
              f"{'Mode1 ECE':>10} {'Mode2 ECE':>10}")
    print(header)
    print("  " + "-" * (len(header) - 2))
    for level in SNR_LEVELS:
        key = "clean" if level == "clean" else str(level)
        m1 = mode1_results[key]
        m2 = mode2_results[key]
        if key == "clean":
            m1_acc, m1_ece = m1["accuracy"] * 100, m1["ece_uncalibrated"]
            m2_acc, m2_ece = m2["accuracy"] * 100, m2["ece_uncalibrated"]
        else:
            m1_acc, m1_ece = m1["accuracy_mean"] * 100, m1["ece_uncalibrated_mean"]
            m2_acc, m2_ece = m2["accuracy_mean"] * 100, m2["ece_uncalibrated_mean"]
        print(f"  {str(level):>8} | {m1_acc:>11.2f} {m2_acc:>11.2f} | {m1_ece:>10.4f} {m2_ece:>10.4f}")


def print_predicted_class_distribution(dist, mode_name):
    print(f"\n  {mode_name} predicted-class distribution at 5 dB (n={dist['n_samples']})")
    print(f"    {'class':<12}{'pred_count':>12}{'pred_frac':>12}")
    for c, name in enumerate(CLASS_NAMES):
        print(f"    {name:<12}{dist['predicted_class_counts'][c]:>12d}"
              f"{dist['predicted_class_frac'][c] * 100:>11.2f}%")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    X_test_raw = np.load(PROCESSED_DIR / "X_test_raw.npy")
    y_test = np.load(PROCESSED_DIR / "y_test.npy")
    scaler = joblib.load(PROCESSED_DIR / "scaler.joblib")

    print(f"Full test set size: {X_test_raw.shape[0]} samples (used as one "
          f"batch for Mode 2 batch statistics).")

    model = load_cnn_bn()

    print(f"\n{'#' * 60}\nMode 1: stored running stats (normal eval)\n{'#' * 60}")
    mode1_results = run_sweep(model, infer_mode1_normal, X_test_raw, y_test, scaler)

    print(f"\n{'#' * 60}\nMode 2: current-batch stats (BN.train(), no_grad, "
          f"buffers restored)\n{'#' * 60}")
    mode2_results = run_sweep(model, infer_mode2_batch_stats, X_test_raw, y_test, scaler)

    print_side_by_side(mode1_results, mode2_results)

    dist_mode1 = predicted_class_distribution(model, infer_mode1_normal, X_test_raw, scaler, snr_db=5)
    dist_mode2 = predicted_class_distribution(model, infer_mode2_batch_stats, X_test_raw, scaler, snr_db=5)
    print_predicted_class_distribution(dist_mode1, "Mode 1")
    print_predicted_class_distribution(dist_mode2, "Mode 2")

    mode1_5db_acc = mode1_results["5"]["accuracy_mean"] * 100
    mode2_5db_acc = mode2_results["5"]["accuracy_mean"] * 100
    print(f"\n{'=' * 60}\nINTERPRETATION\n{'=' * 60}")
    print(f"  Mode 1 (stored stats) accuracy at 5 dB: {mode1_5db_acc:.2f}%")
    print(f"  Mode 2 (batch stats)  accuracy at 5 dB: {mode2_5db_acc:.2f}%")
    if mode2_5db_acc > mode1_5db_acc + 30:
        print("  -> Mode 2 jumped substantially: stored statistics ARE a "
              "major part of the mechanism.")
    elif mode2_5db_acc < mode1_5db_acc + 10:
        print("  -> Mode 2 stayed near Mode 1: stored statistics are NOT the "
              "mechanism; look elsewhere (e.g. pooling-axis).")
    else:
        print("  -> Mode 2 improved partially: stored statistics may be a "
              "contributing factor but not the sole cause.")

    out = {
        "mode1_stored_stats": mode1_results,
        "mode2_batch_stats": mode2_results,
        "predicted_class_distribution_5db": {
            "mode1_stored_stats": dist_mode1,
            "mode2_batch_stats": dist_mode2,
        },
    }
    out_path = OUT_DIR / "bn_stored_stats_mechanism.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
