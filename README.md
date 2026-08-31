# DroneRF Low-SNR Robustness Study
Final-year B.Tech research project: multi-class drone identification from RF signals using a CNN-BiLSTM architecture, with a focus on evaluating classification robustness across the full SNR range (a gap identified in prior DroneRF literature, e.g. Glüge et al. 2024, arXiv:2406.18624).

## Setup
1. Create venv: python -m venv venv
2. Activate: venv\Scripts\activate (Windows) or source venv/bin/activate (Mac/Linux)
3. Install: pip install -r requirements.txt

## Project structure
- data/ — dataset (not tracked in git, see Dataset section)
- preprocess/ — signal preprocessing scripts (STFT, WPT features, SNR injection)
- models/ — model architecture definitions (SVM baseline, CNN, ResNet-1D, CNN-BiLSTM)
- train/ — training scripts
- eval/ — evaluation and plotting scripts
- results/ — output CSVs, plots, confusion matrices
- paper/ — LaTeX manuscript
- notebooks/ — exploratory Jupyter notebooks

## Dataset
DroneRF dataset (Al-Sa'd et al., 2019). Instructions for obtaining it are in preprocess/README.md (to be added).
