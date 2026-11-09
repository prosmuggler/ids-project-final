# Objective 2: classifier benchmark (TEST split)

Cells are mean +- std over seeds (test split, fixed data split, so std understates uncertainty). Seeds: 5 for dt, lgbm, xgb, mlp; 3 for rf, catboost, cnn1d. Latency is from the development Mac CPU, not the Raspberry Pi. `ensemble` = soft vote of the top 3 models chosen on validation.

## Macro-F1

| model | ciciot2023/attack_type | ciciot2023/binary | ciciot2023/category | wsn_ds/binary | wsn_ds/category |
|---|---|---|---|---|---|
| catboost |  | 0.778 ± 0.001 | 0.680 ± 0.001 | 0.984 ± 0.001 | 0.955 ± 0.000 |
| cnn1d | 0.627 ± 0.001 | 0.742 ± 0.009 | 0.641 ± 0.001 | 0.984 ± 0.001 | 0.947 ± 0.003 |
| dt | 0.592 ± 0.000 | 0.779 ± 0.000 | 0.658 ± 0.000 | 0.975 ± 0.000 | 0.951 ± 0.001 |
| ensemble | 0.647 ± 0.001 | 0.772 ± 0.000 | 0.693 ± 0.001 | 0.986 ± 0.000 | 0.968 ± 0.000 |
| lgbm | 0.647 ± 0.001 | 0.754 ± 0.000 | 0.679 ± 0.001 | 0.987 ± 0.000 | 0.968 ± 0.001 |
| mlp | 0.613 ± 0.002 | 0.714 ± 0.004 | 0.634 ± 0.001 | 0.982 ± 0.001 | 0.943 ± 0.001 |
| rf |  | 0.795 ± 0.000 | 0.720 ± 0.000 | 0.985 ± 0.000 | 0.962 ± 0.000 |
| xgb | 0.633 ± 0.001 | 0.775 ± 0.001 | 0.698 ± 0.000 | 0.986 ± 0.000 | 0.967 ± 0.001 |

## Balanced accuracy

| model | ciciot2023/attack_type | ciciot2023/binary | ciciot2023/category | wsn_ds/binary | wsn_ds/category |
|---|---|---|---|---|---|
| catboost |  | 0.955 ± 0.001 | 0.762 ± 0.001 | 0.988 ± 0.000 | 0.976 ± 0.000 |
| cnn1d | 0.629 ± 0.001 | 0.934 ± 0.041 | 0.738 ± 0.004 | 0.988 ± 0.001 | 0.965 ± 0.001 |
| dt | 0.617 ± 0.001 | 0.929 ± 0.000 | 0.703 ± 0.000 | 0.973 ± 0.000 | 0.946 ± 0.001 |
| ensemble | 0.656 ± 0.000 | 0.959 ± 0.000 | 0.773 ± 0.001 | 0.988 ± 0.000 | 0.974 ± 0.000 |
| lgbm | 0.653 ± 0.000 | 0.962 ± 0.000 | 0.775 ± 0.001 | 0.986 ± 0.000 | 0.971 ± 0.001 |
| mlp | 0.617 ± 0.002 | 0.956 ± 0.001 | 0.731 ± 0.004 | 0.988 ± 0.001 | 0.964 ± 0.001 |
| rf |  | 0.941 ± 0.000 | 0.737 ± 0.001 | 0.988 ± 0.000 | 0.972 ± 0.000 |
| xgb | 0.660 ± 0.001 | 0.957 ± 0.000 | 0.768 ± 0.001 | 0.988 ± 0.000 | 0.975 ± 0.001 |

## False-alarm rate (benign flows flagged)

| model | ciciot2023/attack_type | ciciot2023/binary | ciciot2023/category | wsn_ds/binary | wsn_ds/category |
|---|---|---|---|---|---|
| catboost |  | 0.043 ± 0.002 | 0.207 ± 0.002 | 0.003 ± 0.000 | 0.005 ± 0.000 |
| cnn1d | 0.254 ± 0.005 | 0.072 ± 0.095 | 0.283 ± 0.011 | 0.003 ± 0.000 | 0.004 ± 0.001 |
| dt | 0.406 ± 0.001 | 0.100 ± 0.000 | 0.310 ± 0.001 | 0.003 ± 0.000 | 0.003 ± 0.000 |
| ensemble | 0.235 ± 0.001 | 0.030 ± 0.001 | 0.189 ± 0.001 | 0.002 ± 0.000 | 0.002 ± 0.000 |
| lgbm | 0.232 ± 0.002 | 0.017 ± 0.000 | 0.204 ± 0.004 | 0.002 ± 0.000 | 0.002 ± 0.000 |
| mlp | 0.249 ± 0.009 | 0.008 ± 0.001 | 0.286 ± 0.014 | 0.004 ± 0.000 | 0.005 ± 0.000 |
| rf |  | 0.079 ± 0.000 | 0.210 ± 0.001 | 0.003 ± 0.000 | 0.003 ± 0.000 |
| xgb | 0.284 ± 0.001 | 0.037 ± 0.000 | 0.185 ± 0.001 | 0.002 ± 0.000 | 0.003 ± 0.000 |

## ROC-AUC

| model | ciciot2023/attack_type | ciciot2023/binary | ciciot2023/category | wsn_ds/binary | wsn_ds/category |
|---|---|---|---|---|---|
| catboost |  | 0.989 ± 0.000 | 0.978 ± 0.000 | 0.995 ± 0.000 | 0.995 ± 0.000 |
| cnn1d | 0.987 ± 0.000 | 0.984 ± 0.005 | 0.976 ± 0.000 | 0.994 ± 0.000 | 0.995 ± 0.000 |
| dt | 0.877 ± 0.000 | 0.939 ± 0.000 | 0.879 ± 0.000 | 0.973 ± 0.000 | 0.968 ± 0.001 |
| ensemble | 0.988 ± 0.000 | 0.989 ± 0.000 | 0.980 ± 0.000 | 0.995 ± 0.000 | 0.996 ± 0.000 |
| lgbm | 0.988 ± 0.000 | 0.989 ± 0.000 | 0.980 ± 0.000 | 0.995 ± 0.000 | 0.995 ± 0.000 |
| mlp | 0.986 ± 0.000 | 0.985 ± 0.001 | 0.974 ± 0.000 | 0.995 ± 0.000 | 0.995 ± 0.000 |
| rf |  | 0.987 ± 0.000 | 0.978 ± 0.000 | 0.994 ± 0.000 | 0.992 ± 0.000 |
| xgb | 0.988 ± 0.000 | 0.989 ± 0.000 | 0.979 ± 0.000 | 0.995 ± 0.000 | 0.996 ± 0.000 |

## Seeds per model

| model | ciciot2023/attack_type | ciciot2023/binary | ciciot2023/category | wsn_ds/binary | wsn_ds/category |
|---|---|---|---|---|---|
| catboost |  | 3 | 3 | 3 | 3 |
| cnn1d | 3 | 3 | 3 | 3 | 3 |
| dt | 5 | 5 | 5 | 5 | 5 |
| ensemble | 3 | 3 | 3 | 3 | 3 |
| lgbm | 5 | 5 | 5 | 5 | 5 |
| mlp | 5 | 5 | 5 | 5 | 5 |
| rf |  | 3 | 3 | 3 | 3 |
| xgb | 5 | 5 | 5 | 5 | 5 |

## Cost (mean over seeds)

| block | model | size MB | latency ms (batch 1) | flows/s | train s | peak RAM MB |
|---|---|---|---|---|---|---|
| ciciot2023/attack_type | cnn1d | 1.1 | 0.16 | 9,959 | 1037 | 116 |
| ciciot2023/attack_type | dt | 77.9 | 0.26 | 3,122,391 | 21 | 251 |
| ciciot2023/attack_type | ensemble | 168.0 | 9.57 | 1,711 | 2481 | 1655 |
| ciciot2023/attack_type | lgbm | 84.5 | 3.38 | 2,360 | 741 | 1362 |
| ciciot2023/attack_type | mlp | 0.2 | 0.06 | 1,982,365 | 103 | 252 |
| ciciot2023/attack_type | xgb | 82.4 | 5.47 | 27,623 | 673 | 1089 |
| ciciot2023/binary | catboost | 6.7 | 0.77 | 581,221 | 194 | 486 |
| ciciot2023/binary | cnn1d | 1.1 | 0.15 | 9,900 | 1074 | 150 |
| ciciot2023/binary | dt | 1.8 | 0.26 | 3,650,256 | 7 | 90 |
| ciciot2023/binary | ensemble | 13.6 | 2.41 | 159,281 | 228 | 486 |
| ciciot2023/binary | lgbm | 2.5 | 0.58 | 345,710 | 19 | 230 |
| ciciot2023/binary | mlp | 0.2 | 0.06 | 2,146,364 | 79 | 165 |
| ciciot2023/binary | rf | 382.5 | 14.46 | 47,850 | 381 | 875 |
| ciciot2023/binary | xgb | 4.4 | 1.16 | 564,045 | 16 | 66 |
| ciciot2023/category | catboost | 29.8 | 0.67 | 573,343 | 1357 | 2477 |
| ciciot2023/category | cnn1d | 1.1 | 0.18 | 10,075 | 475 | 125 |
| ciciot2023/category | dt | 20.8 | 0.24 | 3,490,912 | 19 | 97 |
| ciciot2023/category | ensemble | 82.2 | 3.55 | 25,288 | 1625 | 2477 |
| ciciot2023/category | lgbm | 19.8 | 0.81 | 35,229 | 139 | 435 |
| ciciot2023/category | mlp | 0.2 | 0.05 | 2,463,597 | 79 | 206 |
| ciciot2023/category | rf | 3037.8 | 15.28 | 35,232 | 528 | 2708 |
| ciciot2023/category | xgb | 32.6 | 2.05 | 120,428 | 130 | 168 |
| wsn_ds/binary | catboost | 7.9 | 0.19 | 1,362,607 | 8 | 193 |
| wsn_ds/binary | cnn1d | 0.6 | 0.18 | 26,810 | 24 | 119 |
| wsn_ds/binary | dt | 0.3 | 0.20 | 4,874,000 | 1 | 5 |
| wsn_ds/binary | ensemble | 67.7 | 13.95 | 14,968 | 34 | 162 |
| wsn_ds/binary | lgbm | 2.7 | 0.29 | 304,225 | 4 | 35 |
| wsn_ds/binary | mlp | 0.2 | 0.04 | 2,660,191 | 6 | 122 |
| wsn_ds/binary | rf | 64.4 | 13.47 | 38,955 | 7 | 152 |
| wsn_ds/binary | xgb | 2.2 | 0.45 | 1,117,717 | 1 | 33 |
| wsn_ds/category | catboost | 23.5 | 0.30 | 741,101 | 49 | 385 |
| wsn_ds/category | cnn1d | 0.6 | 0.21 | 29,556 | 29 | 122 |
| wsn_ds/category | dt | 0.4 | 0.21 | 4,467,173 | 1 | 6 |
| wsn_ds/category | ensemble | 93.0 | 16.60 | 24,258 | 25 | 170 |
| wsn_ds/category | lgbm | 8.0 | 0.53 | 98,202 | 15 | 63 |
| wsn_ds/category | mlp | 0.2 | 0.04 | 2,410,862 | 11 | 127 |
| wsn_ds/category | rf | 77.7 | 15.19 | 34,688 | 7 | 162 |
| wsn_ds/category | xgb | 7.2 | 0.96 | 393,221 | 4 | 38 |
