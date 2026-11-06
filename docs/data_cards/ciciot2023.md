# Data card: CICIoT2023

## Overview
- **Source:** Canadian Institute for Cybersecurity, IoT attack dataset (flow-level features extracted from pcap captures of an IoT testbed).
- **Task:** Detection of IoT network attacks: binary (benign vs attack), 8-way category, and 34-way attack type.
- **Labels:** `binary` (0 = Benign, 1 = Attack); `category` (Benign, DDoS, DoS, Recon, Mirai, Spoofing, BruteForce, Web); `attack_type` (folder name of the capture, Benign merged into one class).
- **Sampling:** Attack folders under 30 MB keep 100% of rows (rare classes); larger folders are sampled at 10%.

## Cleaning (identical rows removed before splitting, so duplicates cannot leak across splits)
- Rows before cleaning: 5,108,909
- Dropped NaN/inf rows: 110
- Dropped duplicate rows: 1,904,190
- Rows after cleaning: 3,204,609
- Dropped tiny classes (<10 rows): none
- Dropped constant columns (detected on train only): none
- Features kept: 39

## Splits
- **Random stratified** (seed 42): train 2,243,226 / val 480,691 / test 480,692  (`data/processed/ciciot2023/`)
- **Temporal:** train 2,243,242 / val 480,694 / test 480,673  (`data/processed/ciciot2023_temporal/`)
- Time note: No timestamps. The temporal split orders rows by capture-file number within each attack. This ASSUMES file numbers follow capture order, which the dataset does not guarantee.
- Scaling and class-imbalance handling are fit on the train split only (`src/data/prepare.py`).

## Class counts (random split)
| attack_type | train | val | test |
|---|---|---|---|
| DDoS-ICMP_Flood | 209,295 | 44,849 | 44,849 |
| DDoS-UDP_Flood | 192,425 | 41,234 | 41,234 |
| DDoS-SYN_Flood | 168,906 | 36,194 | 36,195 |
| DDoS-PSHACK_FLOOD | 164,518 | 35,254 | 35,254 |
| DDoS-TCP_Flood | 157,055 | 33,654 | 33,655 |
| DoS-UDP_Flood | 150,284 | 32,203 | 32,204 |
| DDoS-RSTFINFLOOD | 137,751 | 29,518 | 29,518 |
| DDoS-SynonymousIP_Flood | 129,037 | 27,650 | 27,651 |
| DoS-TCP_Flood | 110,564 | 23,692 | 23,693 |
| DoS-SYN_Flood | 102,561 | 21,977 | 21,978 |
| Recon-HostDiscovery | 93,467 | 20,028 | 20,029 |
| Benign | 76,755 | 16,447 | 16,448 |
| Mirai-greeth_flood | 68,725 | 14,727 | 14,727 |
| Recon-OSScan | 67,546 | 14,474 | 14,475 |
| Mirai-udpplain | 61,067 | 13,085 | 13,086 |
| Recon-PortScan | 56,169 | 12,037 | 12,036 |
| Mirai-greip_flood | 52,307 | 11,209 | 11,209 |
| DoS-HTTP_Flood | 50,040 | 10,723 | 10,722 |
| DDoS-ICMP_Fragmentation | 31,658 | 6,784 | 6,784 |
| VulnerabilityScan | 26,099 | 5,593 | 5,592 |
| DDoS-HTTP_Flood | 20,153 | 4,319 | 4,318 |
| MITM-ArpSpoofing | 20,104 | 4,308 | 4,308 |
| DDoS-UDP_Fragmentation | 20,071 | 4,301 | 4,301 |
| DDoS-ACK_Fragmentation | 19,939 | 4,273 | 4,272 |
| DDoS-SlowLoris | 16,398 | 3,514 | 3,513 |
| DNS_Spoofing | 12,292 | 2,634 | 2,634 |
| DictionaryBruteForce | 9,143 | 1,960 | 1,959 |
| BrowserHijacking | 4,052 | 868 | 868 |
| CommandInjection | 3,772 | 808 | 808 |
| SqlInjection | 3,671 | 787 | 786 |
| XSS | 2,692 | 577 | 577 |
| Backdoor_Malware | 2,251 | 482 | 482 |
| Recon-PingSweep | 1,583 | 340 | 339 |
| Uploading_Attack | 876 | 188 | 188 |

## Top 10 features (mean of MI, Random Forest and RFE ranks, train split only)
1. `Tot sum` (mean rank 1.0)
2. `Tot size` (mean rank 3.7)
3. `AVG` (mean rank 4.0)
4. `Max` (mean rank 4.3)
5. `Number` (mean rank 5.7)
6. `IAT` (mean rank 6.3)
7. `Header_Length` (mean rank 6.7)
8. `Rate` (mean rank 6.7)
9. `Time_To_Live` (mean rank 9.7)
10. `Variance` (mean rank 10.7)

Full ranking: `results/features/ciciot2023_feature_ranking.csv`

## Known limitations
- Label conflicts: in the random split, 26,706 test rows (5.6%) have exactly the same features as a train row; 21,299 of them (4.4% of test) carry a different `category`, but only 53 differ at the `binary` level. Category/attack-type accuracy therefore has a ceiling of roughly 96%, while binary detection is essentially unaffected.
- Heavy class imbalance (DDoS is 56% of rows, BruteForce 0.4%).
- Flow-level features only; no payload content. Results come from a single lab testbed and may not transfer to other networks.
- Cleaning removed a large share of duplicate rows, so the processed set is smaller than the published dataset.

## Reproduce
`python -m src.data.prep_ciciot` then `python -m src.data.features ciciot2023` then `python -m src.data.temporal_split ciciot2023`
