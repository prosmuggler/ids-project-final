"""Top-k features + scaler Pipeline + imbalance handling. Everything is FIT ON TRAIN ONLY.

    from src.data.prepare import build
    d = build("ciciot2023", k="20", level="category", imbalance="class_weight")
    d["X_train"], d["y_train"], d["X_val"], ..., d["class_weight"], d["pipeline"]

k: "10" | "20" | "30" | "all"      level: binary | category | attack_type
scaler: standard | minmax | none   imbalance: none | class_weight | smote | undersample
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.utils.class_weight import compute_class_weight

from src.data.common import load_cfg

CAP = 100_000  # SMOTE raises small classes up to CAP; undersampling cuts big classes down to CAP


def build(ds, k="all", level="category", scaler="standard", imbalance="none",
          cap=CAP, save=True):
    seed = load_cfg()["seed"]
    d = Path(f"data/processed/{ds}")

    topk = json.load(open(f"results/features/{ds}_topk.json"))
    if str(k) not in topk:
        raise ValueError(f"k={k} not available for {ds}; choose from {list(topk)}")
    feats = topk[str(k)]

    parts = {n: pd.read_parquet(d / f"{n}.parquet") for n in ("train", "val", "test")}
    X = {n: p[feats].astype("float32") for n, p in parts.items()}
    y = {n: p[level].to_numpy() for n, p in parts.items()}

    # 1) scaler: fit on TRAIN, then only transform val/test
    pipe = None
    if scaler != "none":
        pipe = Pipeline([("scaler", {"standard": StandardScaler,
                                     "minmax": MinMaxScaler}[scaler]())])
        pipe.fit(X["train"])
        X = {n: pd.DataFrame(pipe.transform(v), columns=feats) for n, v in X.items()}
        if save:
            Path("results/pipelines").mkdir(parents=True, exist_ok=True)
            joblib.dump(pipe, f"results/pipelines/{ds}_k{k}_{scaler}.joblib")

    # 2) imbalance: TRAIN only, after scaling (SMOTE's neighbour search needs scaled data)
    Xtr, ytr = X["train"], y["train"]
    counts = pd.Series(ytr).value_counts()
    class_weight = None

    if imbalance == "class_weight":
        classes = np.unique(ytr)
        w = compute_class_weight("balanced", classes=classes, y=ytr)
        class_weight = dict(zip(classes.tolist(), w.tolist()))
    elif imbalance == "smote":
        strategy = {c: cap for c, n in counts.items() if n < cap}
        if strategy:
            Xtr, ytr = SMOTE(sampling_strategy=strategy, k_neighbors=5,
                             random_state=seed).fit_resample(Xtr, ytr)
    elif imbalance == "undersample":
        strategy = {c: cap for c, n in counts.items() if n > cap}
        if strategy:
            Xtr, ytr = RandomUnderSampler(sampling_strategy=strategy,
                                          random_state=seed).fit_resample(Xtr, ytr)
    elif imbalance != "none":
        raise ValueError(f"unknown imbalance method: {imbalance}")

    return {"X_train": Xtr, "y_train": ytr, "X_val": X["val"], "y_val": y["val"],
            "X_test": X["test"], "y_test": y["test"], "features": feats,
            "class_weight": class_weight, "pipeline": pipe}
