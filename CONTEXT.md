# Project Context — DroneRF Calibration Study

## Project Goal
Final-year B.Tech research project. Publishing a paper on the calibration of RF-based drone classifiers on the DroneRF dataset. Target venue: IEEE INDICON 2026. Timeline: 7 days from start (currently on Day 4). Working solo, no GPU, 8GB RAM laptop.

## Core Contribution
First systematic calibration study of DroneRF classifiers. Prior work reports 84-99% accuracy but nobody has measured whether confidence scores are trustworthy (ECE, reliability diagrams). We show classifiers are severely miscalibrated and demonstrate temperature scaling fixes it without accuracy loss. Secondary contribution: calibration degrades faster than accuracy under noise.

## Dataset
- DroneRF preprocessed (Al-Sa'd et al., 2019) — 22,700 samples, 4-class task
- 2048-bin power spectrum per sample (already extracted, we don't have raw IQ)
- Classes: 0=background (4100), 1=Bebop (8400), 2=AR (8100), 3=Phantom (2100)
- File: data/dronerf_preprocessed/RF_Data.csv (482 MB, gitignored)
- Stratified 70/15/15 train/val/test split, seed=42
- Preprocessing: log10(X + 1e-10) then StandardScaler fit on train
- Processed splits: data/processed/{X,y}_{train,val,test}.npy and scaler.joblib

## Baselines (5 models total — 3 main + 2 BatchNorm ablations, do not add more)
- SVM (RBF, C=10, full data) → results/baseline_svm/ → done, 93.89% test accuracy
- CNN-GroupNorm (1D, 3 conv blocks, GroupNorm/LayerNorm-equivalent) → results/baseline_cnn/ → done, 91.22% test accuracy — well-calibrated variant
- MLP-LayerNorm-free (Linear 2048→512→128→4, ReLU + Dropout 0.2, no norm layers) → results/baseline_mlp/ → done, 90.28% test accuracy — well-calibrated variant
- CNN-BatchNorm (same architecture as CNN-GroupNorm, BatchNorm1d swapped in) → results/baseline_cnn_bn/ → pending manual training (train/train_cnn_bn.py)
- MLP-BatchNorm (same architecture as MLP-LayerNorm-free, BatchNorm1d added after each hidden Linear) → results/baseline_mlp_bn/ → pending manual training (train/train_mlp_bn.py)
- ResNet1D — DO NOT USE (redundant with CNN, skipped)

Each baseline saves: test_logits.npy (or test_scores.npy for SVM), test_labels.npy, metrics.json

## Ablation study
Added CNN-BatchNorm and MLP-BatchNorm variants to strengthen the calibration
story. Prior DroneRF work (Al-Sa'd 2019, Al-Emadi 2020) uses BatchNorm-style
architectures; our main CNN/MLP baselines deliberately avoid BatchNorm
(GroupNorm/no-norm) because BatchNorm was unstable on the sparse
log-spectrum input during development. Hypothesis: the BatchNorm variants
will show noticeably worse (higher ECE) uncalibrated confidence than our
main baselines, i.e. the normalization choice itself — not just model
family — is a driver of miscalibration, and temperature scaling should
still fix it post-hoc without hurting accuracy. This turns the calibration
story from "our 3 models are miscalibrated" into a controlled architecture
comparison.

## Known Findings from Diagnostics
- No data leakage in this dataset (verified via shuffled-label control and nearest-neighbor checks)
- Natural accuracy ceiling on 2048-bin PSD is ~94% (SVM confirmed, 1-NN got 90.8%)
- BatchNorm unstable on sparse log-spectrum → use LayerNorm for deep models
- Bebop↔AR is the main confusion pair (both Parrot drones with similar RF signatures)

## Environment
- Windows, PowerShell, VS Code
- Python venv at venv/ — activate with venv\Scripts\Activate.ps1
- Execution policy fix if needed: Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
- CPU only, no GPU
- All dependencies in requirements.txt

## Repo Layout
- data/                  (gitignored, dataset lives here)
- preprocess/            (load_and_split.py — log+standardize pipeline)
- models/                (empty, model classes inline in train scripts)
- train/                 (train_svm.py, train_cnn.py, train_mlp.py, train_cnn_bn.py, train_mlp_bn.py, utils.py)
- eval/                  (calibration.py — ECE/MCE/Brier/NLL + temperature scaling for all 5 models)
- diagnose/              (diagnostic scripts — reference only)
- results/               (gitignored, all outputs)
- paper/references/REFERENCES.md   (13 key citations)
- notebooks/
- CONTEXT.md             (THIS FILE)
- requirements.txt

## Key References (see paper/references/REFERENCES.md for full list)
- Al-Sa'd et al. 2019 — DroneRF dataset paper
- Guo et al. 2017 (ICML) — Temperature scaling and ECE, our primary methodology
- Glüge et al. 2024 (arXiv:2406.18624) — Low-SNR robustness gap on DroneRF
- Shulman 2025 (arXiv:2607.01025) — DroneRF benchmark data leakage (we verified we don't have this issue)

## What Not to Do
- Do NOT redesign architectures beyond small fixes — baselines are frozen
- Do NOT chase 99% accuracy — 94% ceiling is understood and documented
- Do NOT add ResNet or new baselines
- Do NOT commit anything under data/ or results/
- Do NOT run scripts without activating the venv first

## Current State (update this section as we progress)
- Day 5: Fixing calibration bug, adding BatchNorm ablation baselines
- Original 3 baselines complete: SVM 93.89%, CNN 91.22%, MLP 90.28%
- Bebop-AR confusion consistent across all three model families (data-inherent)
- Found and fixed a real bug in eval/calibration.py's fit_temperature(): the
  softplus reparameterization dampened LBFGS's gradient and left it
  unconverged at max_iter=50, so temperature scaling was making ECE/NLL/Brier
  *worse* for all 3 models instead of better. Fixed by optimizing T directly
  (no reparameterization) with line_search_fn="strong_wolfe", max_iter=100,
  and 3 repeated .step(closure) calls, matching the reference Guo et al. 2017
  implementation. Verified against manual grid search (MLP: true optimal
  T~1.2, now returns T~1.18; ECE now drops 0.0177→0.0119 instead of rising).
- Next: run train_cnn_bn.py and train_mlp_bn.py manually, then re-run
  eval/calibration.py across all 5 models, then calibration-under-noise
  experiments, then paper writing
