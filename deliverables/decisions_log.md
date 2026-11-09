# Decisions, deviations and open questions (running log)

## Objective 1
- N-BaIoT was not downloaded (my decision). The guide lists "N-BaIoT or TON_IoT" as required for anomaly work.
- The CICIoT2023 CSV files used have 39 feature columns; the paper reports 46. Features were extracted by the dataset authors with their own DPKT-based scripts (not CICFlowMeter or nfstream).
- Duplicates were removed BEFORE splitting: 37% of sampled CICIoT2023 rows and 11% of WSN-DS rows.
- Imbalance handling: class weights were chosen. In the Objective 1 study, undersampling cut every class above the cap (not only benign/DDoS), and the comparison used one Random Forest, one seed, validation split only.
- Label conflicts: in the random split, 26,706 CICIoT2023 test rows (5.6%) have exactly the same features as a train row; 21,299 (4.4%) have a different category; only 53 differ at binary level. Near-duplicates add more overlap (DoS vs DDoS, Benign vs Recon).
- Main CICIoT2023 feature set is the top 30 (k=30 matched "all 39": 0.7157 vs 0.7164 macro-F1 in the study).
- Temporal split: CICIoT2023 assumes capture-file numbers follow capture order (no timestamps); WSN-DS uses its Time column. Not yet scored in Objective 2.

## Objective 2
- KNN trained on a 200k-row subsample and scored on 50k validation rows; linear SVM trained on a 300k subsample.
- Optuna tuning used a 150k-row stratified train subsample, at category level only; parameters were reused for binary and attack type. Random Forest got 30 trials and CatBoost 20 on CICIoT2023 (others 50).
- LightGBM at 34 classes diverged with default settings (macro-F1 0.12-0.16, validation loss rose after ~20 rounds). Fixed with learning_rate 0.05, min_child_samples 100, reg_lambda 5 and sqrt class weights.
- Final test evaluation: seeds 42-46 for dt, lgbm, xgb, mlp; seeds 42-44 for rf, catboost, cnn1d. Seeds change model randomness only; the data split is fixed, so the std values understate uncertainty. RF and CatBoost were not run at attack-type level.
- Deep models: MLP ~50k parameters, 1D-CNN 142k-269k; sqrt class weights at 34 classes. The CICIoT binary CNN checkpoint came from a one-epoch validation spike (poor test false-alarm rate 0.18); reported as is.
- Binary macro-F1 is misleading on CICIoT2023 (benign is 3.4% of rows); use balanced accuracy, false-alarm rate and ROC-AUC.
- Soft-voting ensemble over the top 3 (chosen from VALIDATION scores) gave no reliable gain over its best member and costs the sum of its members.
- Friedman/Nemenyi has low power (4-5 blocks, blocks are not independent datasets).
- Sequence models (LSTM/GRU, CNN-LSTM) dropped: CICIoT2023 has no flow identifiers; in WSN-DS 77% of rows share (id, Time, attack) with another row, and only Normal has groups of 10+ rows.
- Tree/boosting models and PyTorch models must run in separate processes (segmentation fault on macOS otherwise).
- All latency numbers are from the Mac CPU, not the Raspberry Pi.
- Provisional recommendation: LightGBM (main), MLP (tiny edge candidate), shallow Decision Tree (explainable baseline / distillation target). Random Forest (3 GB), the ensemble and the 1D-CNN not recommended.

## Objective 3 (planned)
- "Never trained on" = leave-one-category-out (CICIoT2023) / leave-one-attack-out (WSN-DS); detectors see benign rows only; thresholds from benign validation percentiles (95, 99, 99.5).
- LSTM autoencoder and per-node sequences are not possible (no valid sequence identifiers); planned substitutes: dense autoencoder, per-row forwarded/received ratio on WSN-DS.
- Disclose: the top-30 feature ranking was computed with all categories present (or re-rank per fold).
- First probe: Isolation Forest and dense autoencoder, detection rate per category at 1%/5% benign false alarms.

## Objective 4 (risks)
- Models use the 39 CSV features; nfstream on the Raspberry Pi computes different features. Options: own DPKT/scapy extractor following the paper's definitions, a deployable feature subset, or retraining from raw PCAPs.
- Energy: USB-C inline power meter on the Pi (best); otherwise estimate from CPU time x published Pi 4 power, plus a Mac powermetrics proxy; label estimates clearly.
- One Raspberry Pi 4 available. Replay test traffic from the Mac over a direct cable or isolated switch only.

## Questions for the supervisor (not yet asked)
1. Is leave-one-category-out an acceptable definition of "attacks never trained on"?
2. May the LSTM autoencoder and per-node sequences be replaced by a dense autoencoder and per-row ratio features?
3. Is N-BaIoT/TON_IoT required, or are CICIoT2023 + WSN-DS enough for Objective 3?

## Objective 3 - probe results (test split, 3 seeds)
- Benign-only detectors, thresholds from benign validation percentiles; test benign false alarms matched nominal (about 5%, 1%, 0.5%).
- CICIoT2023 autoencoder: DDoS, DoS, Mirai detected ~100% even at the 99.5th percentile (AUC 1.000); Recon AUC 0.83 (31% at 99th pct); BruteForce 0.81 (13%); Web 0.77 (5.5%); Spoofing 0.74 (6.5%). Isolation Forest is weaker on floods.
- WSN-DS: Isolation Forest collapses at the 99th percentile (6-16% detected) while the autoencoder keeps 79-99.8%; AE chosen.
- The autoencoder hit the 100-epoch cap in 5 of 6 runs and its best benign validation error varies across seeds: not fully converged.
- The probe was run on the TEST split; operating points (percentile, confidence threshold) are chosen on validation in hybrid.py.
- Feature ranking (top 30) was computed with all categories present; disclose as a limitation.

## Objective 3 - hybrid results (test split, guide defaults: 99th percentile, tau 0.6)
- Gated hybrid as specified performs worse than its parts: on WSN-DS the classifier (trained without the held-out attack) calls flagged rows "Normal" with confidence (detected 35% vs 87% anomaly-only); on CICIoT2023 known-class macro-F1 falls from 0.71 to 0.475.
- hybrid-or (classifier decides; unknown if confidence < tau or if the detector flags a benign verdict): WSN-DS detects 87% of unseen attacks at 1.1% benign false alarms with known-class macro-F1 0.971 (supervised 0.972).
- CICIoT2023: hybrid-or detects 94.4% of unseen attacks but gives 35% benign false alarms (supervised alone 18%, anomaly-only 1%) and 16.6% of known attacks reported unknown; it is within 0.2 points of supervised+reject, i.e. the benign-only detector adds almost nothing beyond confidence rejection. Cause: Benign vs Recon overlap in the flow features.
- Unseen floods (DDoS/DoS/Mirai) are alarmed 100% but almost never labelled unknown (confidently assigned to another flood family).
- "detected" = alarm raised (labelled anything but benign); "unknown" and "misassigned" are reported separately.
- Validation and test agree within ~0.003; operating points are chosen on validation.

## Objective 3 - hybrid and detector comparison (update)
- CICIoT2023 test, 99th pct, tau 0.6 (unseen detected / benign false alarms / known macro-F1): gate 50.7% / 0.7% / 0.475; hybrid-guard 68.2% / 5.7% / 0.705; supervised 87.5% / 18.3% / 0.713; hybrid-or 94.4% / 34.9% / 0.719. No variant dominates: detection trades against false alarms because benign and attack flows overlap in the features.
- Operating point chosen on validation (false-alarm budget <= 5%): 99th pct, tau 0.7. Test: hybrid-guard 62.5% detected / 3.3% false alarms / known macro-F1 0.691 (validation predicted 62.2% / 3.2%); hybrid-or 96.1% / 43.4%; 19.8% of known attacks reported unknown by guard.
- WSN-DS: guard = hybrid-or, 87% detected at 1.1% false alarms, known macro-F1 0.971 (gate: 35% detected).
- Mean per-category detection at the 99th percentile (CICIoT2023 / WSN-DS): One-Class SVM 0.53 / 0.81, autoencoder 0.51 / 0.87, LOF 0.46 / 0.42, Isolation Forest 0.36 / 0.11. All four miss Spoofing and Web (5-12%). One-Class SVM and LOF: one seed, 20k benign subsample.
- N-BaIoT (115 features, 9 devices, 35 Mirai + 45 BASHLITE files, no labels inside). One benign-only detector per device; benign rows split in time order 70/15/15; attack files used only as test data. Devices 1-2, autoencoder: Mirai 100%, gafgyt combo/junk/scan 100%, gafgyt tcp/udp missed on device 2. Device 2 benign false alarms 5.3% at the 99th pct (nominal 1%): only ~2,000 benign validation rows.
- N-BaIoT full run (9 devices, 3 seeds, one benign-only detector per device). Autoencoder: Mirai and BASHLITE combo/junk/scan detected 100% at all thresholds; BASHLITE tcp/udp 81.5% / 29.7% / 18.6% at the 95th / 99th / 99.5th percentile. Isolation Forest weaker (26-75% per attack type at the 99th). Devices 3 and 7 have no Mirai.
- N-BaIoT benign false alarms at the 99th percentile (nominal 1%): autoencoder 0.4-1.4% on 5 devices but 2.6% (3), 4.7% (2), 7.3% (9), 8.4% (7); Isolation Forest up to 17.7% (device 8). Cause: time-ordered benign split (drift) and small benign sets.
- gafgyt.tcp and gafgyt.udp give identical detection numbers for both detectors: needs checking for overlapping flows before reporting.
- N-BaIoT: gafgyt.tcp/udp contain ~100 distinct rows each (>=98% one repeated vector); reported separately. Deduplicated benign run is the main result: autoencoder false alarms at or below nominal on 8 of 9 devices (device 2: 4.7% at the 99th pct).
- WSN-DS: all Blackhole, Flooding and Grayhole rows in the training split are cluster heads. A detector trained on all Normal rows flagged Normal cluster heads at 11.2% vs 0.8% for other Normal rows (99th pct). Cluster-head detector (trained on Normal cluster heads only): Blackhole 100% by rule fwd_ratio==0 at 0.3% false alarms; TDMA 91.6% (autoencoder); Grayhole and Flooding hard (19% and 10% at the 99th pct).
