"""Probe: how well do detectors trained on BENIGN data only flag each attack category?
No classifier is involved. Thresholds come from percentiles of benign VALIDATION scores.
  python -m src.train.anomaly_probe --dataset ciciot2023 --detector iforest
  python -m src.train.anomaly_probe --dataset ciciot2023 --detector ae
  python -m src.train.anomaly_probe --dataset wsn_ds --detector iforest   (then --detector ae)
Run the two detectors in SEPARATE commands: PyTorch has crashed when mixed with other native libraries on macOS.
"""
import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
import sys

if "--detector" in sys.argv and sys.argv[sys.argv.index("--detector") + 1] == "ae":
    import torch  # noqa: F401  loaded first, before any other native library

import argparse
import copy
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
BENIGN = {"ciciot2023": "Benign", "wsn_ds": "Normal"}
K = {"ciciot2023": "30", "wsn_ds": "all"}
PCTS = [95, 99, 99.5]
OUT = Path("results/anomaly")


def load(ds):
    feats = json.load(open(f"results/features/{ds}_topk.json"))[K[ds]]
    cols = feats + ["category", "attack_type"]
    parts = {s: pd.read_parquet(f"data/processed/{ds}/{s}.parquet", columns=cols)
             for s in ("train", "val", "test")}
    return feats, parts


def iforest_scores(Xb_tr, Xb_val, X_test, seed):
    from sklearn.ensemble import IsolationForest
    m = IsolationForest(n_estimators=200, max_samples=256, random_state=seed, n_jobs=-1).fit(Xb_tr)
    return -m.score_samples(Xb_val), -m.score_samples(X_test)       # higher = more anomalous


def ae_scores(Xb_tr, Xb_val, X_test, seed, epochs=100, patience=8, bs=512):
    import torch
    import torch.nn as nn
    sc = StandardScaler().fit(Xb_tr)                                 # fit on BENIGN train only
    prep = lambda X: torch.from_numpy(np.clip(sc.transform(X), -10, 10).astype(np.float32))
    tr, va, te = prep(Xb_tr), prep(Xb_val), prep(X_test)
    n_in = tr.shape[1]
    torch.manual_seed(seed)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    net = nn.Sequential(nn.Linear(n_in, 32), nn.ReLU(), nn.Linear(32, 16), nn.ReLU(),
                        nn.Linear(16, 8), nn.ReLU(), nn.Linear(8, 16), nn.ReLU(),
                        nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, n_in)).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    g = torch.Generator().manual_seed(seed)

    @torch.inference_mode()
    def err(X):
        net.eval()
        out = [((net(X[i:i + 65536].to(device)) - X[i:i + 65536].to(device)) ** 2).mean(1).cpu()
               for i in range(0, len(X), 65536)]
        return torch.cat(out).numpy()

    best, best_state, bad = np.inf, None, 0
    for ep in range(1, epochs + 1):
        net.train()
        perm = torch.randperm(len(tr), generator=g)
        for i in range(0, len(tr), bs):
            xb = tr[perm[i:i + bs]].to(device)
            opt.zero_grad(set_to_none=True)
            loss = ((net(xb) - xb) ** 2).mean()
            loss.backward()
            opt.step()
        v = float(err(va).mean())                                    # benign validation reconstruction error
        if v < best - 1e-6:
            best, bad = v, 0
            best_state = copy.deepcopy({k: x.detach().cpu() for k, x in net.state_dict().items()})
        else:
            bad += 1
            if bad >= patience:
                break
    net.load_state_dict(best_state)
    print(f"    autoencoder: {ep} epochs, best benign-val MSE {best:.4f}", flush=True)
    return err(va), err(te)


def evaluate(s_val, s_test, test, ds):
    thr = {p: np.percentile(s_val, p) for p in PCTS}
    ben_name = BENIGN[ds]
    is_ben = (test["category"] == ben_name).to_numpy()
    rows = []
    for level in ("category", "attack_type"):
        for grp, idx in test.groupby(level).indices.items():
            s = s_test[idx]
            r = {"level": level, "group": grp, "n_rows": len(idx)}
            for p in PCTS:
                r[f"rate_p{str(p).replace('.', '_')}"] = float((s > thr[p]).mean())
            if grp == ben_name:
                r["auc"] = np.nan                                    # rate columns = false-alarm rate here
            else:
                y = np.r_[np.zeros(is_ben.sum()), np.ones(len(idx))]
                r["auc"] = float(roc_auc_score(y, np.r_[s_test[is_ben], s]))
            rows.append(r)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(BENIGN))
    ap.add_argument("--detector", required=True, choices=["iforest", "ae"])
    ap.add_argument("--seeds", default="42,43,44")
    a = ap.parse_args()

    feats, parts = load(a.dataset)
    ben = BENIGN[a.dataset]
    Xb_tr = parts["train"].loc[parts["train"].category == ben, feats].to_numpy(np.float32)
    Xb_val = parts["val"].loc[parts["val"].category == ben, feats].to_numpy(np.float32)
    test = parts["test"].reset_index(drop=True)
    X_test = test[feats].to_numpy(np.float32)
    print(f"{a.dataset}/{a.detector}: benign train={len(Xb_tr):,} benign val={len(Xb_val):,} "
          f"test rows={len(test):,} features={len(feats)}")

    rows = []
    for seed in [int(s) for s in a.seeds.split(",")]:
        t0 = time.time()
        fn = iforest_scores if a.detector == "iforest" else ae_scores
        s_val, s_test = fn(Xb_tr, Xb_val, X_test, seed)
        for r in evaluate(s_val, s_test, test, a.dataset):
            rows.append({"dataset": a.dataset, "detector": a.detector, "seed": seed, **r})
        print(f"  seed {seed} done in {time.time() - t0:.0f}s", flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    res = pd.DataFrame(rows)
    res.to_csv(OUT / f"probe_{a.dataset}_{a.detector}.csv", index=False)
    cat = res[res.level == "category"].groupby("group")[["n_rows", "auc", "rate_p95", "rate_p99", "rate_p99_5"]].mean()
    print(f"\n== {a.dataset} / {a.detector}: category level, mean over seeds")
    print("   rate_pXX = share of rows flagged at the benign-validation XXth-percentile threshold")
    print("   (for the benign row this is the false-alarm rate; for attacks it is the detection rate)")
    print(cat.round(3).to_string())


if __name__ == "__main__":
    main()
