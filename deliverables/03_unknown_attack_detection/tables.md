# Objective 3: detecting attacks the classifier was never trained on

## ciciot2023: leave-one-category-out (TEST; anomaly detector = autoencoder; percentile 99, confidence 0.7, chosen on validation)

Each row: the held-out category was removed from classifier training; detector trained on benign rows only.

### Share of the held-out category that raised an alarm (labelled anything but benign)

| heldout | supervised | supervised+reject | anomaly-only | hybrid (gate) | hybrid-or | hybrid-guard |
|---|---|---|---|---|---|---|
| BruteForce | 0.765 | 0.928 | 0.132 | 0.131 | 0.929 | 0.329 |
| DDoS | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| DoS | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| Mirai | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| Recon | 0.742 | 0.878 | 0.313 | 0.307 | 0.884 | 0.509 |
| Spoofing | 0.786 | 0.935 | 0.065 | 0.064 | 0.936 | 0.266 |
| Web | 0.830 | 0.978 | 0.055 | 0.054 | 0.979 | 0.269 |

### Share labelled "unknown"

| heldout | supervised | supervised+reject | anomaly-only | hybrid (gate) | hybrid-or | hybrid-guard |
|---|---|---|---|---|---|---|
| BruteForce | 0.000 | 0.633 |  | 0.034 | 0.635 | 0.035 |
| DDoS | 0.000 | 0.002 |  | 0.002 | 0.002 | 0.002 |
| DoS | 0.000 | 0.067 |  | 0.067 | 0.067 | 0.067 |
| Mirai | 0.000 | 0.002 |  | 0.002 | 0.002 | 0.002 |
| Recon | 0.000 | 0.438 |  | 0.063 | 0.444 | 0.069 |
| Spoofing | 0.000 | 0.703 |  | 0.033 | 0.704 | 0.034 |
| Web | 0.000 | 0.730 |  | 0.021 | 0.731 | 0.022 |

### Three-system comparison (mean over held-out categories)

`known_*` columns measure the cost on the categories that were NOT held out.

| system | heldout_detected | heldout_unknown | heldout_missed | known_benign_false_alarm | known_attack_detected | known_attack_unknown | known_macro_f1 |
|---|---|---|---|---|---|---|---|
| supervised | 0.875 | 0.000 | 0.125 | 0.183 | 0.978 | 0.000 | 0.713 |
| supervised+reject | 0.960 | 0.368 | 0.040 | 0.432 | 0.992 | 0.263 | 0.703 |
| anomaly-only | 0.509 |  |  | 0.011 | 0.885 |  |  |
| hybrid (gate) | 0.508 | 0.032 | 0.492 | 0.008 | 0.885 | 0.198 | 0.467 |
| hybrid-or | 0.961 | 0.369 | 0.039 | 0.434 | 0.992 | 0.263 | 0.703 |
| hybrid-guard | 0.625 | 0.033 | 0.375 | 0.033 | 0.927 | 0.198 | 0.691 |

## wsn_ds: leave-one-category-out (TEST; anomaly detector = autoencoder; percentile 99, confidence 0.6, chosen on validation)

Each row: the held-out category was removed from classifier training; detector trained on benign rows only.

### Share of the held-out category that raised an alarm (labelled anything but benign)

| heldout | supervised | supervised+reject | anomaly-only | hybrid (gate) | hybrid-or | hybrid-guard |
|---|---|---|---|---|---|---|
| Blackhole | 0.897 | 0.918 | 0.998 | 0.916 | 1.000 | 1.000 |
| Flooding | 0.004 | 0.006 | 0.839 | 0.006 | 0.839 | 0.839 |
| Grayhole | 0.231 | 0.232 | 0.789 | 0.230 | 0.790 | 0.789 |
| TDMA | 0.220 | 0.239 | 0.852 | 0.239 | 0.853 | 0.853 |

### Share labelled "unknown"

| heldout | supervised | supervised+reject | anomaly-only | hybrid (gate) | hybrid-or | hybrid-guard |
|---|---|---|---|---|---|---|
| Blackhole | 0.000 | 0.044 |  | 0.044 | 0.126 | 0.126 |
| Flooding | 0.000 | 0.002 |  | 0.002 | 0.835 | 0.835 |
| Grayhole | 0.000 | 0.001 |  | 0.000 | 0.560 | 0.559 |
| TDMA | 0.000 | 0.040 |  | 0.040 | 0.654 | 0.654 |

### Three-system comparison (mean over held-out categories)

`known_*` columns measure the cost on the categories that were NOT held out.

| system | heldout_detected | heldout_unknown | heldout_missed | known_benign_false_alarm | known_attack_detected | known_attack_unknown | known_macro_f1 |
|---|---|---|---|---|---|---|---|
| supervised | 0.338 | 0.000 | 0.662 | 0.001 | 0.972 | 0.000 | 0.972 |
| supervised+reject | 0.349 | 0.022 | 0.651 | 0.002 | 0.973 | 0.003 | 0.972 |
| anomaly-only | 0.870 |  |  | 0.010 | 0.851 |  |  |
| hybrid (gate) | 0.348 | 0.022 | 0.652 | 0.001 | 0.848 | 0.002 | 0.932 |
| hybrid-or | 0.871 | 0.544 | 0.129 | 0.011 | 0.976 | 0.006 | 0.971 |
| hybrid-guard | 0.870 | 0.544 | 0.130 | 0.011 | 0.975 | 0.005 | 0.971 |

## ciciot2023: benign-only detectors per attack category (TEST)

The benign row shows the false-alarm rate. Isolation Forest and autoencoder: 3 seeds; One-Class SVM and LOF: 1 seed, 20,000-row benign subsample.

### ROC-AUC vs benign

| group | iforest | ae | ocsvm | lof |
|---|---|---|---|---|
| Benign |  |  |  |  |
| BruteForce | 0.743 | 0.805 | 0.762 | 0.711 |
| DDoS | 0.963 | 1.000 | 1.000 | 0.996 |
| DoS | 0.963 | 1.000 | 1.000 | 0.997 |
| Mirai | 0.995 | 1.000 | 1.000 | 0.999 |
| Recon | 0.824 | 0.827 | 0.761 | 0.728 |
| Spoofing | 0.684 | 0.740 | 0.700 | 0.755 |
| Web | 0.751 | 0.774 | 0.691 | 0.729 |

### Detection rate at the 99th-percentile threshold

| group | iforest | ae | ocsvm | lof |
|---|---|---|---|---|
| Benign | 0.009 | 0.011 | 0.012 | 0.012 |
| BruteForce | 0.074 | 0.132 | 0.207 | 0.020 |
| DDoS | 0.561 | 1.000 | 1.000 | 1.000 |
| DoS | 0.525 | 1.000 | 1.000 | 1.000 |
| Mirai | 0.925 | 1.000 | 1.000 | 1.000 |
| Recon | 0.292 | 0.313 | 0.363 | 0.028 |
| Spoofing | 0.096 | 0.065 | 0.047 | 0.118 |
| Web | 0.061 | 0.055 | 0.069 | 0.063 |

### Detection rate at the 95th-percentile threshold

| group | iforest | ae | ocsvm | lof |
|---|---|---|---|---|
| Benign | 0.048 | 0.050 | 0.048 | 0.054 |
| BruteForce | 0.248 | 0.363 | 0.457 | 0.160 |
| DDoS | 0.779 | 1.000 | 1.000 | 1.000 |
| DoS | 0.763 | 1.000 | 1.000 | 1.000 |
| Mirai | 0.997 | 1.000 | 1.000 | 1.000 |
| Recon | 0.464 | 0.477 | 0.492 | 0.303 |
| Spoofing | 0.245 | 0.209 | 0.255 | 0.321 |
| Web | 0.222 | 0.211 | 0.265 | 0.193 |

## wsn_ds: benign-only detectors per attack category (TEST)

The benign row shows the false-alarm rate. Isolation Forest and autoencoder: 3 seeds; One-Class SVM and LOF: 1 seed, 20,000-row benign subsample.

### ROC-AUC vs benign

| group | iforest | ae | ocsvm | lof |
|---|---|---|---|---|
| Blackhole | 0.981 | 0.999 | 0.999 | 0.994 |
| Flooding | 0.985 | 0.995 | 0.998 | 0.525 |
| Grayhole | 0.979 | 0.992 | 0.984 | 0.952 |
| Normal |  |  |  |  |
| TDMA | 0.942 | 0.954 | 0.951 | 0.941 |

### Detection rate at the 99th-percentile threshold

| group | iforest | ae | ocsvm | lof |
|---|---|---|---|---|
| Blackhole | 0.117 | 0.998 | 1.000 | 0.776 |
| Flooding | 0.160 | 0.839 | 1.000 | 0.175 |
| Grayhole | 0.064 | 0.789 | 0.375 | 0.347 |
| Normal | 0.011 | 0.010 | 0.011 | 0.009 |
| TDMA | 0.116 | 0.852 | 0.881 | 0.384 |

### Detection rate at the 95th-percentile threshold

| group | iforest | ae | ocsvm | lof |
|---|---|---|---|---|
| Blackhole | 0.993 | 1.000 | 1.000 | 1.000 |
| Flooding | 1.000 | 1.000 | 1.000 | 0.279 |
| Grayhole | 0.998 | 0.994 | 0.961 | 0.881 |
| Normal | 0.053 | 0.053 | 0.054 | 0.052 |
| TDMA | 0.923 | 0.904 | 0.921 | 0.860 |

## N-BaIoT: one benign-only detector per device (deduplicated benign data, 3 seeds)

Benign rows split in time order 70/15/15; thresholds from benign validation percentiles; attack files are test-only.

**Caution:** `gafgyt.tcp` and `gafgyt.udp` contain only ~100 distinct rows each (at least 98% of each file is one repeated feature vector), so their rates are a yes/no on one pattern per device, not a measure over ~100,000 flows.

### Per attack type (mean over devices and seeds)

| group | AE auc | AE rate_p95 | AE rate_p99 | AE rate_p99_5 | IF auc | IF rate_p95 | IF rate_p99 | IF rate_p99_5 |
|---|---|---|---|---|---|---|---|---|
| gafgyt.combo | 1.000 | 1.000 | 1.000 | 1.000 | 0.983 | 0.827 | 0.584 | 0.348 |
| gafgyt.junk | 1.000 | 1.000 | 1.000 | 1.000 | 0.985 | 0.884 | 0.602 | 0.368 |
| gafgyt.scan | 1.000 | 1.000 | 1.000 | 0.999 | 0.970 | 0.832 | 0.567 | 0.318 |
| gafgyt.tcp | 0.976 | 0.889 | 0.260 | 0.260 | 0.912 | 0.556 | 0.260 | 0.222 |
| gafgyt.udp | 0.976 | 0.889 | 0.260 | 0.260 | 0.912 | 0.556 | 0.260 | 0.222 |
| mirai.ack | 1.000 | 1.000 | 1.000 | 1.000 | 0.994 | 0.997 | 0.655 | 0.461 |
| mirai.scan | 1.000 | 1.000 | 1.000 | 0.999 | 0.964 | 0.742 | 0.381 | 0.268 |
| mirai.syn | 1.000 | 1.000 | 1.000 | 1.000 | 0.975 | 0.758 | 0.465 | 0.309 |
| mirai.udp | 1.000 | 1.000 | 1.000 | 1.000 | 0.995 | 1.000 | 0.661 | 0.461 |
| mirai.udpplain | 1.000 | 1.000 | 1.000 | 1.000 | 0.995 | 1.000 | 0.708 | 0.544 |

### Autoencoder, the 8 non-degenerate attack types only (mean over all devices)

| detector | auc | rate_p95 | rate_p99 | rate_p99_5 |
|---|---|---|---|---|
| AE | 1.000 | 1.000 | 1.000 | 1.000 |

### Benign false-alarm rate per device (nominal 5% / 1% / 0.5%)

| device | AE rate_p95 | AE rate_p99 | AE rate_p99_5 | IF rate_p95 | IF rate_p99 | IF rate_p99_5 |
|---|---|---|---|---|---|---|
| 1.000 | 0.010 | 0.000 | 0.000 | 0.038 | 0.005 | 0.002 |
| 2.000 | 0.119 | 0.047 | 0.038 | 0.081 | 0.011 | 0.009 |
| 3.000 | 0.034 | 0.008 | 0.008 | 0.039 | 0.008 | 0.007 |
| 4.000 | 0.033 | 0.003 | 0.001 | 0.026 | 0.003 | 0.001 |
| 5.000 | 0.020 | 0.003 | 0.002 | 0.048 | 0.011 | 0.005 |
| 6.000 | 0.047 | 0.007 | 0.003 | 0.039 | 0.007 | 0.004 |
| 7.000 | 0.039 | 0.003 | 0.001 | 0.048 | 0.010 | 0.005 |
| 8.000 | 0.099 | 0.008 | 0.002 | 0.107 | 0.068 | 0.047 |
| 9.000 | 0.054 | 0.003 | 0.001 | 0.057 | 0.008 | 0.004 |

## WSN-DS: node behaviour, cluster-head rows only (TEST)

Detectors trained on Normal cluster-head rows only; features include forwarded/received ratio. The Normal row is the false-alarm rate on Normal cluster heads. The rule flags a cluster head that forwarded nothing to the base station.

| index | iforest auc | iforest rate_p95 | iforest rate_p99 | ae auc | ae rate_p95 | ae rate_p99 | rule (forwarded==0) |
|---|---|---|---|---|---|---|---|
| Blackhole | 0.973 | 0.875 | 0.045 | 0.987 | 0.991 | 0.317 | 1.000 |
| Flooding | 0.947 | 0.600 | 0.166 | 0.914 | 0.409 | 0.096 | 0.019 |
| Grayhole | 0.878 | 0.368 | 0.003 | 0.947 | 0.685 | 0.192 | 0.230 |
| Normal |  | 0.035 | 0.010 |  | 0.040 | 0.007 | 0.003 |
| TDMA | 0.948 | 0.798 | 0.062 | 0.987 | 0.920 | 0.916 | 0.791 |
