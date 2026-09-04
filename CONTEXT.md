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

## Baselines (finalized, do not add more)
- SVM (RBF, C=10, full data) → results/baseline_svm/ → 93.89% test accuracy
- CNN (1D, LayerNorm-based, 3 conv blocks) → results/baseline_cnn/ → target 88-92%
- MLP (Linear 2048→512→128→4, ReLU + Dropout 0.2) → results/baseline_mlp/ → ~87%
- ResNet1D — DO NOT USE (redundant with CNN, skipped)

Each baseline saves: test_logits.npy (or test_scores.npy for SVM), test_labels.npy, metrics.json

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
- train/                 (train_svm.py, train_cnn.py, train_mlp.py, utils.py)
- eval/                  (calibration.py to be built)
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
- All three baselines complete: SVM 93.89%, CNN 91.22%, MLP 90.28%
- Bebop-AR confusion consistent across all three model families (data-inherent)
- Next: calibration analysis (ECE, reliability diagrams, temperature scaling)
