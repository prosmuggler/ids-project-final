"""Rank features (MI, Random Forest, RFE) on the TRAIN split only.
Usage: python -m src.data.features ciciot2023
"""
import json
import sys
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE, mutual_info_classif
from sklearn.model_selection import train_test_split

from src.data.common import load_cfg

LABEL = "category"


def sample(df, n, seed):
    if len(df) <= n:
        return df
    s, _ = train_test_split(df, train_size=n, stratify=df[LABEL], random_state=seed)
    return s


def main(ds):
    seed = load_cfg()["seed"]
    d = Path(f"data/processed/{ds}")
    feats = json.load(open(d / "cleaning_log.json"))["feature_columns"]
    train = pd.read_parquet(d / "train.parquet")  # train only: no val/test info used

    # mutual information (slow, so smaller sample)
    s = sample(train, 30_000, seed)
    y = pd.factorize(s[LABEL])[0]
    mi = pd.Series(mutual_info_classif(s[feats], y, random_state=seed), index=feats)

    # random forest importance
    s = sample(train, 100_000, seed)
    rf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=seed,
                                class_weight="balanced_subsample")
    rf.fit(s[feats], s[LABEL])
    rf_imp = pd.Series(rf.feature_importances_, index=feats)

    # recursive feature elimination -> full ranking (1 = best)
    s = sample(train, 30_000, seed)
    est = RandomForestClassifier(n_estimators=30, max_depth=10, n_jobs=-1, random_state=seed)
    rfe = RFE(est, n_features_to_select=1, step=1).fit(s[feats], s[LABEL])
    rfe_rank = pd.Series(rfe.ranking_, index=feats)

    out = pd.DataFrame({"mi": mi, "rf_importance": rf_imp, "rfe_rank": rfe_rank})
    out["mi_rank"] = out["mi"].rank(ascending=False)
    out["rf_rank"] = out["rf_importance"].rank(ascending=False)
    out["mean_rank"] = out[["mi_rank", "rf_rank", "rfe_rank"]].mean(axis=1)
    out["final_rank"] = out["mean_rank"].rank(method="first").astype(int)
    out = out.sort_values("final_rank")

    res = Path("results/features")
    res.mkdir(parents=True, exist_ok=True)
    out.index.name = "feature"
    out.to_csv(res / f"{ds}_feature_ranking.csv")

    ranked = out.index.tolist()
    topk = {str(k): ranked[:k] for k in (10, 20, 30) if k < len(ranked)}
    topk["all"] = ranked
    json.dump(topk, open(res / f"{ds}_topk.json", "w"), indent=2)

    print(out.round(4).to_string())
    print(f"\nSaved to {res}/")


if __name__ == "__main__":
    main(sys.argv[1])
