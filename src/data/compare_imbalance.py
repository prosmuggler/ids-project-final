"""Compare imbalance strategies with one fixed Random Forest, scored on VALIDATION.
Usage: python -m src.data.compare_imbalance ciciot2023 20 100000
       python -m src.data.compare_imbalance wsn_ds all 10000
args: dataset, k (10|20|30|all), cap (SMOTE target / undersample ceiling)
"""
import sys
import time
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, f1_score, recall_score

from src.data.common import load_cfg
from src.data.prepare import build

METHODS = ["none", "class_weight", "smote", "undersample"]


def main(ds, k, cap, level="category"):
    seed = load_cfg()["seed"]
    rows, recalls = [], {}
    for m in METHODS:
        d = build(ds, k=k, level=level, imbalance=m, cap=cap, save=False)
        rf = RandomForestClassifier(n_estimators=60, n_jobs=-1, random_state=seed,
                                    class_weight=d["class_weight"])
        t0 = time.time()
        rf.fit(d["X_train"], d["y_train"])
        fit_s = time.time() - t0
        pred = rf.predict(d["X_val"])
        labels = sorted(set(d["y_val"]))
        recalls[m] = pd.Series(recall_score(d["y_val"], pred, labels=labels, average=None), index=labels)
        rows.append({"method": m, "train_rows": len(d["X_train"]),
                     "macro_f1": f1_score(d["y_val"], pred, average="macro"),
                     "balanced_acc": balanced_accuracy_score(d["y_val"], pred),
                     "weighted_f1": f1_score(d["y_val"], pred, average="weighted"),
                     "fit_seconds": fit_s})
        print(f"done {m}: macro_f1={rows[-1]['macro_f1']:.4f}  ({fit_s:.0f}s)", flush=True)

    summary = pd.DataFrame(rows).set_index("method")
    per_class = pd.DataFrame(recalls)
    out = Path("results/imbalance")
    summary.to_csv(out / f"{ds}_k{k}_summary.csv")
    per_class.to_csv(out / f"{ds}_k{k}_per_class_recall.csv")
    print("\n=== summary (validation) ===")
    print(summary.round(4).to_string())
    print("\n=== per-class recall (validation) ===")
    print(per_class.round(3).to_string())


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]))
