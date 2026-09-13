"""CHECK A (steps A3-A4) -- evaluates CNN-BN and CNN-IN across training seeds
{42, 43, 44} at clean/10/5/0 dB, to check whether the BN-collapse / IN-
robustness finding (CNN-BN 14.17%, CNN-IN 79.97% at 5 dB, seed 42) is robust
across independent training runs or a fluke of one seed.

Seed 42 models are the existing results/baseline_cnn_bn/ and
results/baseline_cnn_in/. Seeds 43/44 are results/baseline_cnn_bn_s{43,44}/
and results/baseline_cnn_in_s{43,44}/, produced by diagnose/train_multiseed.py
(must be run first for both models and both seeds -- 4 training runs total).

Uses add_noise_floor + preprocess_raw from eval/snr_degradation.py (the same
noise model and preprocessing as the main SNR sweep), 5 noise seeds per SNR
level, uncalibrated softmax probabilities (no temperature scaling -- these
seed variants were never calibration-fitted, and this check is about
accuracy/ECE robustness across training seeds, not calibration itself).

Run with: python diagnose\\multiseed_eval.py   (after activating venv)
Requires: results/baseline_cnn_bn/model.pt, results/baseline_cnn_in/model.pt
(seed 42, already trained), plus results/baseline_cnn_bn_s43/model.pt,
results/baseline_cnn_bn_s44/model.pt, results/baseline_cnn_in_s43/model.pt,
results/baseline_cnn_in_s44/model.pt (from diagnose/train_multiseed.py),
data/processed/{X_test_raw,y_test}.npy, data/processed/scaler.joblib.
Do NOT run this automatically -- the user runs it manually.
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "eval"))
sys.path.insert(0, str(REPO_ROOT / "train"))
import snr_degradation as sd  # noqa: E402  (add_noise_floor, preprocess_raw, softmax, compute_ece)
from utils import PROCESSED_DIR, RESULTS_DIR  # noqa: E402
from train_cnn_bn import CNN1DBN  # noqa: E402
from train_cnn_in import CNN1DIN  # noqa: E402

OUT_DIR = RESULTS_DIR / "diagnostics"
SNR_LEVELS = ["clean", 10, 5, 0]
NOISE_SEEDS = [0, 1, 2, 3, 4]
TRAIN_SEEDS = [42, 43, 44]

MODEL_DIRS = {
    ("cnn_bn", 42): "baseline_cnn_bn",
    ("cnn_bn", 43): "baseline_cnn_bn_s43",
    ("cnn_bn", 44): "baseline_cnn_bn_s44",
    ("cnn_in", 42): "baseline_cnn_in",
    ("cnn_in", 43): "baseline_cnn_in_s43",
    ("cnn_in", 44): "baseline_cnn_in_s44",
}
MODEL_CLASSES = {"cnn_bn": CNN1DBN, "cnn_in": CNN1DIN}


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


def run_sweep(model, X_test_raw, y_test, scaler):
    results = {}

    X_clean = sd.preprocess_raw(X_test_raw, scaler)
    logits = torch_infer(model, X_clean)
    results["clean"] = evaluate_at_level(logits, y_test)

    for snr_db in SNR_LEVELS:
        if snr_db == "clean":
            continue
        runs = []
        for seed in NOISE_SEEDS:
            rng = np.random.default_rng(seed)
            X_noisy_raw = sd.add_noise_floor(X_test_raw, snr_db, rng)
            X_noisy = sd.preprocess_raw(X_noisy_raw, scaler, snr_db=snr_db)
            logits = torch_infer(model, X_noisy)
            runs.append(evaluate_at_level(logits, y_test))
        results[str(snr_db)] = aggregate(runs)

    return results


def level_key(level):
    return "clean" if level == "clean" else str(level)


def print_seed_table(all_results, model_tag):
    print(f"\n{'=' * 90}\n{model_tag.upper()}: accuracy across training seeds "
          f"{TRAIN_SEEDS}\n{'=' * 90}")
    header = f"  {'SNR':>8} | " + " | ".join(f"seed {s:>3}" for s in TRAIN_SEEDS) + " | mean +/- std"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for level in SNR_LEVELS:
        key = level_key(level)
        accs = []
        for seed in TRAIN_SEEDS:
            r = all_results[(model_tag, seed)]
            if key == "clean":
                acc = r["clean"]["accuracy"] * 100
            else:
                acc = r[key]["accuracy_mean"] * 100
            accs.append(acc)
        accs_arr = np.array(accs)
        row = f"  {str(level):>8} | " + " | ".join(f"{a:>8.2f}" for a in accs)
        row += f" | {accs_arr.mean():>6.2f} +/- {accs_arr.std():>5.2f}"
        print(row)

    print(f"\n  {model_tag.upper()}: uncalibrated ECE across training seeds {TRAIN_SEEDS}")
    header2 = f"  {'SNR':>8} | " + " | ".join(f"seed {s:>3}" for s in TRAIN_SEEDS) + " | mean +/- std"
    print(header2)
    print("  " + "-" * (len(header2) - 2))
    for level in SNR_LEVELS:
        key = level_key(level)
        eces = []
        for seed in TRAIN_SEEDS:
            r = all_results[(model_tag, seed)]
            if key == "clean":
                ece = r["clean"]["ece_uncalibrated"]
            else:
                ece = r[key]["ece_uncalibrated_mean"]
            eces.append(ece)
        eces_arr = np.array(eces)
        row = f"  {str(level):>8} | " + " | ".join(f"{e:>8.4f}" for e in eces)
        row += f" | {eces_arr.mean():>6.4f} +/- {eces_arr.std():>5.4f}"
        print(row)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    X_test_raw = np.load(PROCESSED_DIR / "X_test_raw.npy")
    y_test = np.load(PROCESSED_DIR / "y_test.npy")
    scaler = joblib.load(PROCESSED_DIR / "scaler.joblib")

    all_results = {}
    for (model_tag, seed), dir_name in MODEL_DIRS.items():
        model_path = RESULTS_DIR / dir_name / "model.pt"
        if not model_path.exists():
            print(f"MISSING: {model_path} -- run diagnose/train_multiseed.py "
                  f"--model {model_tag} --seed {seed} first (unless seed==42, "
                  f"which should already exist).")
            continue
        print(f"\n{'#' * 60}\nEvaluating {model_tag} seed={seed} ({dir_name})\n{'#' * 60}")
        model = load_torch_model(MODEL_CLASSES[model_tag], dir_name)
        all_results[(model_tag, seed)] = run_sweep(model, X_test_raw, y_test, scaler)
        for level in SNR_LEVELS:
            key = level_key(level)
            r = all_results[(model_tag, seed)][key]
            if key == "clean":
                print(f"  clean   acc={r['accuracy'] * 100:.2f}% ECE={r['ece_uncalibrated']:.4f}")
            else:
                print(f"  {key:>5} dB acc={r['accuracy_mean'] * 100:.2f}%+/-{r['accuracy_std'] * 100:.2f} "
                      f"ECE={r['ece_uncalibrated_mean']:.4f}+/-{r['ece_uncalibrated_std']:.4f}")

    for model_tag in ["cnn_bn", "cnn_in"]:
        if all((model_tag, s) in all_results for s in TRAIN_SEEDS):
            print_seed_table(all_results, model_tag)
        else:
            missing = [s for s in TRAIN_SEEDS if (model_tag, s) not in all_results]
            print(f"\n(Skipping {model_tag} seed table -- missing seeds {missing}.)")

    serializable = {f"{tag}_s{seed}": res for (tag, seed), res in all_results.items()}
    out_path = OUT_DIR / "multiseed_eval.json"
    with open(out_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
