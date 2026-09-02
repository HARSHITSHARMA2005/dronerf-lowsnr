# Key References for DroneRF Calibration Study

## Primary Dataset Paper
1. **Al-Sa'd, M. F., Al-Ali, A., Mohamed, A., Khattab, T., & Erbad, A. (2019).** RF-based drone detection and identification using deep learning approaches: An initiative towards a large open source drone database. *Future Generation Computer Systems*, 100, 86-97. — Original DroneRF paper, reported ~99.7% (2-class), 84.5% (4-class), 46.8% (10-class).
2. **Allahham, M. S., Al-Sa'd, M. F., Al-Ali, A., Mohamed, A., Khattab, T., & Erbad, A. (2019).** DroneRF dataset: A dataset of drones for RF-based detection, classification and identification. *Data in Brief*, 26, 104313. — DroneRF dataset description paper.

## DroneRF State-of-the-Art (for our accuracy baselines)
3. **Al-Emadi, N., & Al-Senaid, F. (2020).** Drone detection approach based on radio-frequency using convolutional neural network. *IEEE International Conference on Informatics, IoT, and Enabling Technologies*. — CNN baseline (99.8% / 85.8% / 59.2%).
4. **Medaiyese, O. O., Ezuma, M., Lauf, A. P., & Adeniran, A. A. (2021).** Semi-supervised learning framework for UAV detection. — XGBoost approach (99.96% / 90.73% / 70.09%).
5. **Inani, K. N., Sangwan, K. S., & Dhiraj (2023).** Machine learning based framework for drone detection and identification using RF signals. — SOTA on DroneRF (100% / 99.82% / 99.51%).

## Low-SNR / Robustness on DroneRF (supports our Gap 2 secondary contribution)
6. **Glüge, S., Nyfeler, M., Aghaebrahimian, A., Ramagnano, N., & Schüpbach, C. (2024).** Robust Low-Cost Drone Detection and Classification in Low SNR Environments. *arXiv:2406.18624*. — Explicitly identifies low-SNR as an open problem; supports our secondary contribution.
7. **Nguyen, P., Ravindranath, L., Nguyen, D., Han, R., & Vu, T. (2020).** Investigating cost-effective RF-based detection of drones. *arXiv:2009.05519*. — CNN classifier for drones at multiple SNR levels.

## Calibration Foundations (methodology we will apply)
8. **Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017).** On calibration of modern neural networks. *ICML 2017*. — Foundational paper for temperature scaling and Expected Calibration Error (ECE); the primary calibration method we will use.
9. **Naeini, M. P., Cooper, G., & Hauskrecht, M. (2015).** Obtaining well calibrated probabilities using bayesian binning. *AAAI 2015*. — Original ECE metric definition.
10. **Kumar, A., Liang, P., & Ma, T. (2019).** Verified uncertainty calibration. *NeurIPS 2019*. — Scaling-binning calibrator; provides theoretical grounding.

## Calibration Applied to Adjacent RF/Signal Domains (proves methodology carries over)
11. **Kopp, F., Käding, C., & Denzler, J. (2021).** Improving Uncertainty of Deep Learning-based Object Classification on Radar Spectra using Label Smoothing. *arXiv:2109.12851*. — Applies calibration methods to radar spectra; closest adjacent domain to ours.
12. **Ye, T., Si, S., Wang, J., Cheng, N., & Xiao, J. (2022).** Uncertainty Calibration for Deep Audio Classifiers. *arXiv:2206.13071*. — Benchmarks calibration methods for deep audio classifiers.

## Data Leakage Concern (recent, mention as related work)
13. **Shulman, D. (2025).** How Much Do RF Drone Benchmarks Overstate? A Controlled Study and Theory of Data Leakage in UAV Signal Identification. *arXiv:2607.01025*. — Very recent paper arguing DroneRF benchmark accuracies are inflated; complements our finding that even reported accuracies come with unreliable confidence.

---

**Note:** Download the actual PDFs of at least papers 1, 2, 5, 6, 8, and 13 into paper/references/pdfs/ for offline reference during writing. Others can be cited from their abstracts.
