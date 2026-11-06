# Data card: WSN-DS

## Overview
- **Source:** WSN-DS, a wireless sensor network dataset simulated with LEACH clustering (NS-2), with per-node features.
- **Task:** Detection of WSN attacks: binary (Normal vs attack) and 5-way type (Normal, Blackhole, Grayhole, Flooding, TDMA).
- **Labels:** `binary` (0 = Normal, 1 = Attack); `category` and `attack_type` are identical (the dataset has one level of attack naming).
- **Sampling:** Full dataset used, no sampling.

## Cleaning (identical rows removed before splitting, so duplicates cannot leak across splits)
- Rows before cleaning: 374,661
- Dropped NaN/inf rows: 0
- Dropped duplicate rows: 41,140
- Rows after cleaning: 333,521
- Dropped tiny classes (<10 rows): none
- Dropped constant columns (detected on train only): none
- Dropped identifier columns: ['id', 'Time', 'who CH']
- Features kept: 15

## Splits
- **Random stratified** (seed 42): train 233,464 / val 50,028 / test 50,029  (`data/processed/wsn_ds/`)
- **Temporal:** train 233,467 / val 50,029 / test 50,025  (`data/processed/wsn_ds_temporal/`)
- Time note: The raw `Time` column (simulation round) is used only to order rows for the temporal split, then dropped.
- Scaling and class-imbalance handling are fit on the train split only (`src/data/prepare.py`).

## Class counts (random split)
| attack_type | train | val | test |
|---|---|---|---|
| Normal | 216,675 | 46,431 | 46,431 |
| Grayhole | 7,591 | 1,626 | 1,627 |
| TDMA | 4,033 | 864 | 865 |
| Blackhole | 2,958 | 634 | 633 |
| Flooding | 2,207 | 473 | 473 |

## Top 10 features (mean of MI, Random Forest and RFE ranks, train split only)
1. `ADV_S` (mean rank 1.0)
2. `Expended_Energy` (mean rank 4.0)
3. `Is_CH` (mean rank 4.7)
4. `SCH_S` (mean rank 5.3)
5. `Rank` (mean rank 5.3)
6. `Data_Sent_To_BS` (mean rank 6.7)
7. `ADV_R` (mean rank 9.3)
8. `JOIN_S` (mean rank 9.3)
9. `DATA_R` (mean rank 9.7)
10. `dist_CH_To_BS` (mean rank 10.0)

Full ranking: `results/features/wsn_ds_feature_ranking.csv`

## Known limitations
- Simulated data, not real sensor traffic; results may not transfer to real deployments.
- Very imbalanced (Normal is 93% of rows, Flooding about 1%).
- Label conflicts: in the random split, 20 test rows share their features with a train row of a different attack type, none at the `binary` level.
- Identifier columns (`id`, `Time`, `who CH`) were dropped so models cannot memorise nodes or rounds.
- Duplicate rows were removed before splitting.

## Reproduce
`python -m src.data.prep_wsn` then `python -m src.data.features wsn_ds` then `python -m src.data.temporal_split wsn_ds`
