"""N-BaIoT: one benign-only detector PER DEVICE, tested on that device's Mirai and BASHLITE (gafgyt) traffic.
Files are named <device>.benign.csv and <device>.<mirai|gafgyt>.<attack>.csv; there are no labels inside.
Benign rows are split in TIME ORDER: first 70% train, next 15% validation (thresholds), last 15% test.
Attack files are used only as test data. Thresholds = percentiles of benign VALIDATION scores.
  python -m src.train.nbaiot --detector iforest --devices 1,2     # quick test
  python -m src.train.nbaiot --detector ae
Run the iforest and ae detectors in SEPARATE commands (PyTorch must be loaded first).
"""
import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
import sys

if "--detector" in sys.argv and sys.argv[sys.argv.index("--detector") + 1] == "ae":
    import torch  # noqa: F401

import argparse
import re
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.train.anomaly_probe import ae_scores, iforest_scores

warnings.filterwarnings("ignore")
DATA = Path("data/raw/nbaiot")
OUT = Path("results/anomaly")
PCTS = [95, 99, 99.5]


def read(path):
    return pd.read_csv(path, dtype=np.float32).to_numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--detector", required=True, choices=["iforest", "ae"])
    ap.add_argument("--seeds", default="42")
    ap.add_argument("--devices", default="")
    a = ap.parse_args()

    found = sorted({int(m.group(1)) for f in DATA.glob("*.csv") if (m := re.match(r"(\d+)\.", f.name))})
    devices = [int(d) for d in a.devices.split(",")] if a.devices else found
    fn = iforest_scores if a.detector == "iforest" else ae_scores
    rows = []
    for dev in devices:
        t0 = time.time()
        ben = read(DATA / f"{dev}.benign.csv")
        n = len(ben)
        i1, i2 = int(0.70 * n), int(0.85 * n)
        tr, va, te = ben[:i1], ben[i1:i2], ben[i2:]                  # time order, no shuffling
        att = {}
        for f in sorted(DATA.glob(f"{dev}.*.csv")):
            name = f.name.split(".", 1)[1].rsplit(".", 1)[0]          # e.g. "mirai.scan"
            if name != "benign":
                att[name] = read(f)
        names = list(att)
        X_test = np.vstack([te] + [att[k] for k in names])
        bounds = np.cumsum([0, len(te)] + [len(att[k]) for k in names])
        print(f"device {dev}: benign train/val/test = {len(tr):,}/{len(va):,}/{len(te):,}; "
              f"attack files: {', '.join(f'{k} ({len(att[k]):,})' for k in names)}", flush=True)
        for seed in [int(s) for s in a.seeds.split(",")]:
            s_val, s_test, _ = fn(tr, va, X_test, seed)
            thr = {p: np.percentile(s_val, p) for p in PCTS}
            sb = s_test[:len(te)]
            rate = lambda s: {f"rate_p{str(p).replace('.', '_')}": float((s > thr[p]).mean()) for p in PCTS}
            base = {"device": dev, "detector": a.detector, "seed": seed}
            rows.append({**base, "group": "benign_test", "family": "benign", "n_rows": len(sb),
                         "auc": np.nan, **rate(sb)})
            for j, k in enumerate(names):
                s = s_test[bounds[j + 1]:bounds[j + 2]]
                auc = float(roc_auc_score(np.r_[np.zeros(len(sb)), np.ones(len(s))], np.r_[sb, s]))
                rows.append({**base, "group": k, "family": k.split(".")[0], "n_rows": len(s),
                             "auc": auc, **rate(s)})
        print(f"  done in {time.time() - t0:.0f}s", flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    res = pd.DataFrame(rows)
    res.to_csv(OUT / f"nbaiot_{a.detector}.csv", index=False)
    cols = ["auc", "rate_p95", "rate_p99", "rate_p99_5"]
    benign = res[res.family == "benign"].groupby("device")[["rate_p95", "rate_p99", "rate_p99_5"]].mean()
    fam = res[res.family != "benign"].groupby(["device", "family"])[cols].mean()
    print(f"\n== N-BaIoT / {a.detector}: benign-test false-alarm rate per device (mean over seeds)")
    print(benign.round(3).to_string())
    print("\n== detection of each attack family per device (mean over attack files and seeds)")
    print(fam.round(3).to_string())
    print("\n== per attack type, mean over devices")
    print(res[res.family != "benign"].groupby("group")[cols].mean().round(3).to_string())
    print("\nsaved", OUT / f"nbaiot_{a.detector}.csv")


if __name__ == "__main__":
    main()
