"""Time-ordered 70/15/15 split. Usage: python -m src.data.temporal_split ciciot2023|wsn_ds
Output: data/processed/<ds>_temporal/   (random split in data/processed/<ds>/ is untouched)
"""
import glob
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.common import (load_cfg, clean_rows, drop_tiny_classes,
                             drop_constant_cols, save_all)
from src.data.prep_ciciot import attack_name, to_category, KEEP_ALL_MB


def temporal_split(df, label_col, cfg):
    """Within each class, first 70% of rows (in time order) -> train, next 15% -> val, rest -> test."""
    tr, va = cfg["split"]["train"], cfg["split"]["val"]
    frac = df.groupby(label_col).cumcount() / df.groupby(label_col)[label_col].transform("size")
    parts = (df[frac < tr], df[(frac >= tr) & (frac < tr + va)], df[frac >= tr + va])
    return tuple(p.reset_index(drop=True) for p in parts)


def file_order(p):
    m = re.findall(r"(\d+)\.pcap", p.name)
    return int(m[-1]) if m else 0          # "Flood.pcap.csv" (no number) comes first


def ciciot():
    cfg = load_cfg()
    seed, frac = cfg["seed"], cfg["sample_fraction"]
    groups = defaultdict(list)
    for f in glob.glob("data/raw/ciciot2023/**/*.csv", recursive=True):
        groups[attack_name(Path(f))].append(Path(f))

    frames = []
    for attack, fl in groups.items():
        fl = sorted(fl, key=file_order)                    # capture order
        size_mb = sum(f.stat().st_size for f in fl) / 1e6
        f_use = 1.0 if size_mb < KEEP_ALL_MB else frac
        cat = to_category(attack)
        atk = "Benign" if cat == "Benign" else attack
        for f in fl:
            df = pd.read_csv(f)
            df.columns = df.columns.str.strip()
            if f_use < 1.0:
                df = df.sample(frac=f_use, random_state=seed).sort_index()   # keep row order
            num = df.select_dtypes(include=[np.number]).columns
            df[num] = df[num].astype("float32")
            df["category"], df["attack_type"] = cat, atk
            frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    del frames
    df["binary"] = (df["category"] != "Benign").astype("int8")
    return df, "CICIoT2023", "data/processed/ciciot2023_temporal", cfg


def wsn():
    cfg = load_cfg()
    df = pd.read_csv("data/raw/wsn_ds/WSN-DS.csv")
    df.columns = df.columns.str.strip()
    df = df.rename(columns={"Expaned Energy": "Expended_Energy", "Attack type": "attack_type"})
    df["attack_type"] = df["attack_type"].str.strip()
    # stable sort by simulation round inside each attack, then drop identifiers like prep_wsn.py
    df = df.sort_values(["attack_type", "Time"], kind="stable")
    df = df.drop(columns=["id", "Time", "who CH"]).reset_index(drop=True)
    df["category"] = df["attack_type"]
    df["binary"] = (df["attack_type"] != "Normal").astype("int8")
    return df, "WSN-DS", "data/processed/wsn_ds_temporal", cfg


def main(ds):
    df, name, out, cfg = ciciot() if ds == "ciciot2023" else wsn()
    label_cols = ["binary", "category", "attack_type"]
    feature_cols = [c for c in df.columns if c not in label_cols]
    log = {"dataset": name, "split": "temporal (capture/round order within each attack)"}
    df = clean_rows(df, log)                  # keeps row order
    df = drop_tiny_classes(df, "attack_type", log)
    train, val, test = temporal_split(df, "attack_type", cfg)
    train, val, test, feature_cols = drop_constant_cols(train, val, test, feature_cols, log)
    for k, v in log.items():
        if k != "feature_columns":
            print(f"  {k}: {v}")
    save_all(out, train, val, test, feature_cols, log, label_cols)


if __name__ == "__main__":
    main(sys.argv[1])
