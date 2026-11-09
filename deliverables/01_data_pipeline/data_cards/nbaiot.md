# Data card: N-BaIoT

## Overview
- **Source:** N-BaIoT (Meidan et al., 2018): traffic from 9 commercial IoT devices, benign and infected with BASHLITE (gafgyt) and Mirai. Add the download URL you used here.
- **Task here:** semi-supervised anomaly detection, one autoencoder / detector per device, trained on that device's benign traffic only, tested on its Mirai and BASHLITE traffic.
- **Features:** 115 damped incremental statistics per row (names like `MI_dir_L5_weight`), computed by the dataset authors. No identifiers, timestamps or label column inside the files.
- **Labels:** from the file name: `<device>.benign.csv`, `<device>.<mirai|gafgyt>.<attack>.csv`.
  BASHLITE attacks: combo, junk, scan, tcp, udp. Mirai attacks: ack, scan, syn, udp, udpplain.
- **Files:** 89 CSV files, about 8 GB. Devices 3 and 7 have BASHLITE traffic only.

## Devices (rows; attack files / attack rows)
| ID | Device | Benign rows | BASHLITE files / rows | Mirai files / rows | Non-finite values in benign | Duplicate benign rows |
|---|---|---|---|---|---|---|
| 1 | Danmini_Doorbell | 49,548 | 5 / 316,650 | 5 / 652,100 | 0 | 18.5% |
| 2 | Ecobee_Thermostat | 13,113 | 5 / 310,630 | 5 / 512,133 | 0 | 0.0% |
| 3 | Ennio_Doorbell | 39,100 | 5 / 316,400 | 0 / 0 | 0 | 9.4% |
| 4 | Philips_B120N10_Baby_Monitor | 175,240 | 5 / 312,723 | 5 / 610,714 | 0 | 5.8% |
| 5 | Provision_PT_737E_Security_Camera | 62,154 | 5 / 330,096 | 5 / 436,010 | 0 | 11.2% |
| 6 | Provision_PT_838_Security_Camera | 98,514 | 5 / 309,040 | 5 / 429,337 | 0 | 4.6% |
| 7 | Samsung_SNH_1011_N_Webcam | 52,150 | 5 / 323,072 | 0 / 0 | 0 | 5.0% |
| 8 | SimpleHome_XCS7_1002_WHT_Security_Camera | 46,585 | 5 / 303,223 | 5 / 513,248 | 0 | 8.2% |
| 9 | SimpleHome_XCS7_1003_WHT_Security_Camera | 19,528 | 5 / 316,438 | 5 / 514,860 | 0 | 8.2% |

## Splits
- Benign rows of each device are split **in file order** (assumed to be time order; the files carry no timestamps): first 70% train, next 15% validation (anomaly thresholds), last 15% test. No shuffling, because neighbouring rows are highly correlated.
- Attack files are used only as test data and never seen in training.
- No scaling is applied beyond the autoencoder's own standardisation, fitted on benign training rows only.

## Cleaning
- None applied: the files are used as distributed. The benign files were checked for non-finite values and duplicate rows (table above). The attack files were NOT checked.

## Known limitations
- Row order is assumed to be capture order; this was not verified.
- Benign false-alarm rates on the later test rows exceed the nominal threshold rate on several devices (benign drift), notably devices 2, 3, 7 and 9.
- Devices 2 and 9 have little benign data, so their thresholds are noisy.
- BASHLITE tcp and udp gave identical detection scores; see docs/decisions_log.md for the follow-up.

## Reproduce
`python -m src.train.nbaiot --detector ae --seeds 42,43,44` and `python -m src.train.nbaiot --detector iforest --seeds 42,43,44`

## Findings after the first runs
- `gafgyt.tcp` and `gafgyt.udp` are degenerate: only ~100 distinct rows each (device 2: 96 of 95,021 and 110 of 104,791), at least 98% of each file is one repeated feature vector. All other attack files have every row distinct. Their detection rates are a yes/no on one pattern per device; headline results use the other 8 attack types.
- Benign duplicate rows (0-18.5%) were removed before the time split in the reported runs (`--dedup`); without removal, false alarms on devices 7 and 9 were 8.4% and 7.3% at the 99th percentile and fell to 0.3% after.
