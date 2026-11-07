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
