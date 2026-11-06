"""Threshold-free comparison on CICIoT2023 binary (validation). Attack = positive class.
Usage: python -m src.models.binary_ops 30"""
import sys

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from xgboost import XGBClassifier

from src.data.common import load_cfg
from src.data.prepare import build

seed = load_cfg()["seed"]
k = sys.argv[1] if len(sys.argv) > 1 else "30"


def recall_at_fpr(y, p, fpr_max):
    fpr, tpr, _ = roc_curve(y, p)
    return float(tpr[fpr <= fpr_max].max())


rows = []
for mode in ["none", "class_weight"]:
    d = build("ciciot2023", k=k, level="binary", imbalance=mode, save=False)
    cw = d["class_weight"]
    ytr, yv = d["y_train"].astype(int), d["y_val"].astype(int)
    models = {
        "rf": RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=seed, class_weight=cw),
        "xgb": XGBClassifier(n_estimators=200, max_depth=8, learning_rate=0.1, tree_method="hist",
                             n_jobs=-1, random_state=seed),
        "lgbm": LGBMClassifier(n_estimators=200, num_leaves=63, n_jobs=-1, random_state=seed,
                               class_weight=cw, verbose=-1),
    }
    for name, m in models.items():
        if name == "xgb" and cw:
            m.fit(d["X_train"], ytr, sample_weight=np.array([cw[c] for c in ytr]))
        else:
            m.fit(d["X_train"], ytr)
        p = m.predict_proba(d["X_val"])[:, 1]
        rows.append({"weights": mode, "model": name,
                     "roc_auc": roc_auc_score(yv, p),
                     "pr_auc_attack": average_precision_score(yv, p),
                     "pr_auc_benign": average_precision_score(1 - yv, 1 - p),
                     "attack_recall@1%FA": recall_at_fpr(yv, p, 0.01),
                     "attack_recall@5%FA": recall_at_fpr(yv, p, 0.05)})
        print(rows[-1], flush=True)

res = pd.DataFrame(rows).set_index(["weights", "model"])
res.to_csv(f"results/baseline/ciciot2023_binary_k{k}_threshold_free.csv")
print("\n", res.round(4).to_string())
