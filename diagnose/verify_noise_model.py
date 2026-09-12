"""Diagnostic: verify the AWGN noise model in eval/snr_degradation.py actually
produces the SNR levels it claims, given how sparse the DroneRF power spectrum is.

Concern: noise power is computed as sigma^2 = mean(x**2) / 10**(snr/10), i.e. a
single global noise power per sample derived from the mean of ALL 2048 bins.
Because the spectrum is extremely sparse (a few large bins, most bins near the
LOG_EPSILON floor), that mean is dominated by the few large bins. This may mean
the resulting sigma is enormous relative to a "typical" (small) bin, so a
nominal "20 dB SNR" could be swamping the low-power majority of the spectrum
while barely touching the few dominant bins.

This script is DIAGNOSTIC ONLY. It does not modify eval/snr_degradation.py or
any other existing code -- it just reports numbers.

Run with: python diagnose\\verify_noise_model.py   (after activating venv)
Requires: data/processed/X_test_raw.npy
"""
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "train"))
from utils import PROCESSED_DIR  # noqa: E402

LOG_EPSILON = 1e-10
N_SAMPLES = 100
RNG_SEED = 0
SNR_LEVELS_FOR_DISTRIBUTION = [20, 10, 0, -10]
SNR_FOR_LOG_EFFECT = 20
N_SAMPLES_FOR_LOG_EFFECT = 5


def hr(title):
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


# ---------------------------------------------------------------------------
# Noise models
# ---------------------------------------------------------------------------

def add_awgn_global(x_raw, snr_db, rng):
    """Current implementation in eval/snr_degradation.py: single global sigma
    per sample, derived from the mean power over ALL bins."""
    signal_power = np.mean(x_raw ** 2, axis=1, keepdims=True)
    noise_power = signal_power / (10 ** (snr_db / 10))
    sigma = np.sqrt(noise_power)
    noise = rng.standard_normal(x_raw.shape) * sigma
    return x_raw + noise, sigma


def add_awgn_per_bin(x_raw, snr_db, rng):
    """Alternative: noise proportional to each bin's own power.
    sigma_i = x_i / sqrt(10**(snr/10))"""
    sigma = np.abs(x_raw) / np.sqrt(10 ** (snr_db / 10))
    noise = rng.standard_normal(x_raw.shape) * sigma
    return x_raw + noise, sigma


def preprocess_like_pipeline(x_raw, scaler):
    x_clipped = np.maximum(x_raw, LOG_EPSILON)
    x_log = np.log10(x_clipped)
    x_std = scaler.transform(x_log).astype(np.float32)
    clip_frac = np.mean(x_clipped <= LOG_EPSILON)
    return x_std, clip_frac


def main():
    X_test_raw = np.load(PROCESSED_DIR / "X_test_raw.npy")
    scaler_path = PROCESSED_DIR / "scaler.joblib"
    scaler = None
    if scaler_path.exists():
        import joblib
        scaler = joblib.load(scaler_path)

    rng_sample = np.random.default_rng(RNG_SEED)
    idx = rng_sample.choice(X_test_raw.shape[0], size=N_SAMPLES, replace=False)
    X_sample = X_test_raw[idx]

    # =======================================================================
    hr("CHECK 1 — Raw spectrum characterisation (100 random test samples)")
    # =======================================================================
    flat = X_sample.reshape(-1)
    percentiles = [1, 25, 50, 75, 99]
    pvals = np.percentile(flat, percentiles)

    print(f"  n_samples          = {N_SAMPLES}")
    print(f"  n_bins/sample       = {X_sample.shape[1]}")
    print(f"  total bin values    = {flat.size}")
    print(f"  mean                = {flat.mean():.6e}")
    print(f"  median              = {np.median(flat):.6e}")
    print(f"  std                 = {flat.std():.6e}")
    print(f"  min                 = {flat.min():.6e}")
    print(f"  max                 = {flat.max():.6e}")
    print("  percentiles:")
    for p, v in zip(percentiles, pvals):
        print(f"    {p:>2d}th  = {v:.6e}")

    for thresh in [1e-6, 1e-8, 1e-10]:
        frac = np.mean(flat < thresh)
        print(f"  fraction of bins < {thresh:.0e} : {frac * 100:.2f}%")

    # per-sample mean vs median power, to show mean is dominated by few big bins
    per_sample_mean = X_sample.mean(axis=1)
    per_sample_median = np.median(X_sample, axis=1)
    print(f"\n  per-sample mean(bin value):   avg over samples = {per_sample_mean.mean():.6e}")
    print(f"  per-sample median(bin value): avg over samples = {per_sample_median.mean():.6e}")
    ratio = per_sample_mean / np.maximum(per_sample_median, 1e-300)
    print(f"  ratio mean/median per sample: median={np.median(ratio):.3e}, "
          f"mean={ratio.mean():.3e}, max={ratio.max():.3e}")
    print("  (a large mean/median ratio confirms the mean bin power is dominated")
    print("   by a few large bins, not representative of a 'typical' bin)")

    # =======================================================================
    hr("CHECK 2 — Global-sigma noise model: nominal vs. achieved per-bin SNR")
    # =======================================================================
    for snr_db in SNR_LEVELS_FOR_DISTRIBUTION:
        rng = np.random.default_rng(RNG_SEED + 1)
        _, sigma = add_awgn_global(X_sample, snr_db, rng)  # sigma: (N,1)

        # achieved per-bin SNR_i = 10*log10(x_i^2 / sigma^2), sigma constant per sample
        eps = 1e-300
        achieved_db = 10 * np.log10(np.maximum(X_sample ** 2, eps) / (sigma ** 2))

        flat_achieved = achieved_db.reshape(-1)
        med = np.median(flat_achieved)
        p10 = np.percentile(flat_achieved, 10)
        p90 = np.percentile(flat_achieved, 90)
        frac_below_0 = np.mean(flat_achieved < 0)

        print(f"\n  Nominal SNR = {snr_db:>4d} dB")
        print(f"    achieved per-bin SNR:  median={med:8.2f} dB   "
              f"p10={p10:8.2f} dB   p90={p90:8.2f} dB")
        print(f"    fraction of bins with achieved SNR < 0 dB (noise-dominated): "
              f"{frac_below_0 * 100:.2f}%")

    # =======================================================================
    hr(f"CHECK 3 — Effect on log10+standardised representation at {SNR_FOR_LOG_EFFECT} dB "
       f"(global-sigma model, 5 samples)")
    # =======================================================================
    if scaler is None:
        print("  scaler.joblib not found -- skipping standardisation, using raw log10 only")

    sample5_idx = idx[:N_SAMPLES_FOR_LOG_EFFECT]
    X5 = X_test_raw[sample5_idx]

    rng3 = np.random.default_rng(RNG_SEED + 2)
    X5_noisy, sigma5 = add_awgn_global(X5, SNR_FOR_LOG_EFFECT, rng3)

    for i in range(N_SAMPLES_FOR_LOG_EFFECT):
        clean_i = X5[i:i + 1]
        noisy_i = X5_noisy[i:i + 1]

        if scaler is not None:
            clean_proc, clip_frac_clean = preprocess_like_pipeline(clean_i, scaler)
            noisy_proc, clip_frac_noisy = preprocess_like_pipeline(noisy_i, scaler)
        else:
            clean_clipped = np.maximum(clean_i, LOG_EPSILON)
            noisy_clipped = np.maximum(noisy_i, LOG_EPSILON)
            clean_proc = np.log10(clean_clipped)
            noisy_proc = np.log10(noisy_clipped)
            clip_frac_clean = np.mean(clean_clipped <= LOG_EPSILON)
            clip_frac_noisy = np.mean(noisy_clipped <= LOG_EPSILON)

        corr = np.corrcoef(clean_proc.reshape(-1), noisy_proc.reshape(-1))[0, 1]
        print(f"\n  Sample {i} (global-sigma model):")
        print(f"    corr(clean, noisy) after log+standardise = {corr:.4f}")
        print(f"    fraction of bins at LOG_EPSILON floor -- clean: {clip_frac_clean * 100:.2f}%,  "
              f"noisy: {clip_frac_noisy * 100:.2f}%")

    # =======================================================================
    hr(f"CHECK 4 — Alternative per-bin relative noise model at {SNR_FOR_LOG_EFFECT} dB "
       f"(same 5 samples, for comparison)")
    # =======================================================================
    rng4 = np.random.default_rng(RNG_SEED + 3)
    X5_noisy_alt, sigma5_alt = add_awgn_per_bin(X5, SNR_FOR_LOG_EFFECT, rng4)

    corrs_global = []
    corrs_alt = []
    for i in range(N_SAMPLES_FOR_LOG_EFFECT):
        clean_i = X5[i:i + 1]
        noisy_i_alt = X5_noisy_alt[i:i + 1]
        noisy_i_glob = X5_noisy[i:i + 1]

        if scaler is not None:
            clean_proc, clip_frac_clean = preprocess_like_pipeline(clean_i, scaler)
            noisy_proc_alt, clip_frac_noisy_alt = preprocess_like_pipeline(noisy_i_alt, scaler)
            noisy_proc_glob, clip_frac_noisy_glob = preprocess_like_pipeline(noisy_i_glob, scaler)
        else:
            clean_clipped = np.maximum(clean_i, LOG_EPSILON)
            noisy_clipped_alt = np.maximum(noisy_i_alt, LOG_EPSILON)
            noisy_clipped_glob = np.maximum(noisy_i_glob, LOG_EPSILON)
            clean_proc = np.log10(clean_clipped)
            noisy_proc_alt = np.log10(noisy_clipped_alt)
            noisy_proc_glob = np.log10(noisy_clipped_glob)
            clip_frac_clean = np.mean(clean_clipped <= LOG_EPSILON)
            clip_frac_noisy_alt = np.mean(noisy_clipped_alt <= LOG_EPSILON)
            clip_frac_noisy_glob = np.mean(noisy_clipped_glob <= LOG_EPSILON)

        corr_alt = np.corrcoef(clean_proc.reshape(-1), noisy_proc_alt.reshape(-1))[0, 1]
        corr_glob = np.corrcoef(clean_proc.reshape(-1), noisy_proc_glob.reshape(-1))[0, 1]
        corrs_alt.append(corr_alt)
        corrs_global.append(corr_glob)

        print(f"\n  Sample {i} (per-bin relative model):")
        print(f"    corr(clean, noisy) after log+standardise = {corr_alt:.4f}")
        print(f"    fraction of bins at LOG_EPSILON floor -- clean: {clip_frac_clean * 100:.2f}%,  "
              f"noisy: {clip_frac_noisy_alt * 100:.2f}%")

    print(f"\n  Summary over {N_SAMPLES_FOR_LOG_EFFECT} samples at {SNR_FOR_LOG_EFFECT} dB:")
    print(f"    mean corr(clean,noisy) -- global-sigma model:   {np.mean(corrs_global):.4f}")
    print(f"    mean corr(clean,noisy) -- per-bin relative model: {np.mean(corrs_alt):.4f}")

    hr("END OF DIAGNOSTIC — no conclusions drawn, no code modified")


if __name__ == "__main__":
    main()
