"""Baseline tree models, scored on VALIDATION.
Usage: python -m src.models.baseline <dataset> <level> <k> <imbalance> [models]
e.g.   python -m src.models.baseline wsn_ds category all class_weight
       python -m src.models.baseline ciciot2023 category 30 class_weight rf,xgb,lgbm
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

from src.data.common import load_cfg
from src.data.prepare import build
from src.models.evaluate import evaluate


def make(name, cw, seed):
    if name == "rf":
        return RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=seed, class_weight=cw)
    if name == "xgb":
        return XGBClassifier(n_estimators=200, max_depth=8, learning_rate=0.1, tree_method="hist",
                             n_jobs=-1, random_state=seed)
    if name == "lgbm":
        return LGBMClassifier(n_estimators=200, learning_rate=0.1, num_leaves=63, n_jobs=-1,
                              random_state=seed, class_weight=cw, verbose=-1)
    raise ValueError(name)


def main(ds, level, k, imbalance, models="rf,xgb,lgbm"):
    seed = load_cfg()["seed"]
    d = build(ds, k=k, level=level, imbalance=imbalance, save=False)
    le = LabelEncoder().fit(d["y_train"])
    ytr = le.transform(d["y_train"])
    cw = d["class_weight"]
    # models are trained on encoded integers, so the weight dict must use the same integer keys
    cw_enc = {int(le.transform([c])[0]): w for c, w in cw.items()} if cw else None
    out = Path("results/baseline")
    out.mkdir(parents=True, exist_ok=True)

    for name in models.split(","):
        model = make(name, cw_enc, seed)
        t0 = time.time()
        if name == "xgb":
            sw = np.array([cw[c] for c in d["y_train"]]) if cw else None
            model.fit(d["X_train"], ytr, sample_weight=sw)
        else:
            model.fit(d["X_train"], ytr)
        fit_s = time.time() - t0
        pred = le.inverse_transform(model.predict(d["X_val"]))
        m, per_class, cm = evaluate(d["y_val"], pred, level)
        m.update(fit_seconds=fit_s, model=name, dataset=ds, level=level, k=k, imbalance=imbalance)

        tag = f"{ds}_{level}_k{k}_{imbalance}_{name}"
        json.dump({kk: (float(v) if isinstance(v, (np.floating, float)) else v) for kk, v in m.items()},
                  open(out / f"{tag}.json", "w"), indent=2)
        cm.to_csv(out / f"{tag}_confusion.csv")
        per_class.to_csv(out / f"{tag}_recall.csv")
        print(f"\n[{name}] macro_f1={m['macro_f1']:.4f} balanced_acc={m['balanced_acc']:.4f} "
              f"false_alarm={m.get('false_alarm_rate', float('nan')):.4f} ({fit_s:.0f}s)")
        print(per_class.round(3).to_string())


if __name__ == "__main__":
    main(*sys.argv[1:])
