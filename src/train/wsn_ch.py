"""WSN-DS cluster-head behaviour. In this data every Blackhole, Flooding and Grayhole row is a cluster-head row
(Is_CH == 1), so a detector trained on ALL Normal rows can flag cluster-head status itself. Here the detectors are
trained on NORMAL CLUSTER-HEAD rows only and evaluated on cluster-head rows only, with forwarded/received features.
  python -m src.train.wsn_ch --detector iforest
  python -m src.train.wsn_ch --detector ae        (separate command)
"""
import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
import sys

if "--detector" in sys.argv and sys.argv[sys.argv.index("--detector") + 1] == "ae":
    import torch  # noqa: F401

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from src.train.anomaly_probe import ae_scores, evaluate, iforest_scores

warnings.filterwarnings("ignore")
OUT = Path("results/anomaly")


def add_ratio(df):
    df = df.copy()
    df["fwd_ratio"] = df["Data_Sent_To_BS"] / (df["DATA_R"] + 1)       # forwarded to base station / received
    df["fwd_minus_rcv"] = df["Data_Sent_To_BS"] - df["DATA_R"]
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--detector", required=True, choices=["iforest", "ae"])
    ap.add_argument("--seeds", default="42,43,44")
    a = ap.parse_args()

    base = json.load(open("results/features/wsn_ds_topk.json"))["all"]
    cols = base + ["category", "attack_type"]
    parts = {}
    for s in ("train", "val", "test"):
        d = add_ratio(pd.read_parquet(f"data/processed/wsn_ds/{s}.parquet", columns=cols))
        parts[s] = d[d.Is_CH == 1].reset_index(drop=True)             # cluster-head rows only
    feats = [c for c in base if c != "Is_CH"] + ["fwd_ratio", "fwd_minus_rcv"]
    ben = lambda d: d.loc[d.category == "Normal", feats].to_numpy(np.float32)
    Xb_tr, Xb_val = ben(parts["train"]), ben(parts["val"])
    test = parts["test"]
    print(f"cluster-head rows: train Normal {len(Xb_tr):,}; val Normal {len(Xb_val):,}; test {len(test):,} "
          f"({(test.category == 'Normal').sum():,} Normal); features {len(feats)}")

    fn = iforest_scores if a.detector == "iforest" else ae_scores
    rows = []
    for seed in [int(s) for s in a.seeds.split(",")]:
        s_val, s_test, _ = fn(Xb_tr, Xb_val, test[feats].to_numpy(np.float32), seed)
        for r in evaluate(s_val, s_test, test, "wsn_ds"):
            rows.append({"detector": a.detector, "seed": seed, **r})
    res = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    res.to_csv(OUT / f"wsn_ch_{a.detector}.csv", index=False)
    cat = res[res.level == "category"].groupby("group")[["n_rows", "auc", "rate_p95", "rate_p99", "rate_p99_5"]].mean()
    print(f"\n== WSN-DS cluster-head rows / {a.detector}: mean over seeds")
    print("   Normal row = false-alarm rate on Normal cluster heads; attack rows = detection rate")
    print(cat.round(3).to_string())

    rule = (test["fwd_ratio"] == 0)
    print("\n== simple rule 'cluster head forwarded nothing to the base station' (fwd_ratio == 0), test rows flagged:")
    print(rule.groupby(test["category"]).mean().round(3).to_string())
    print("\nsaved", OUT / f"wsn_ch_{a.detector}.csv")


if __name__ == "__main__":
    main()
