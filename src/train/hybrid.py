"""Hybrid detector + leave-one-category-out, using SAVED anomaly scores (no PyTorch in this process).
First:  python -m src.train.anomaly_probe --dataset DS --detector ae --save-scores
Then:   python -m src.train.hybrid --dataset ciciot2023 --detector ae

For each held-out category h the classifier (LightGBM, tuned params, class weights) is retrained WITHOUT h.
Systems compared:
  supervised          classifier only; it can never say "unknown"
  supervised+reject   classifier; label "unknown" when max probability < tau
  anomaly-only        flag when the anomaly score exceeds a percentile of benign VALIDATION scores
  hybrid              flagged -> classifier (max prob < tau -> "unknown"); not flagged -> benign
Choose (percentile, tau) on the VALIDATION rows (split == "val"); then read TEST at that point.
"""
import argparse
import glob
import json
import re
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight

warnings.filterwarnings("ignore")
BENIGN = {"ciciot2023": "Benign", "wsn_ds": "Normal"}
K = {"ciciot2023": "30", "wsn_ds": "all"}
PCTS = [95, 99, 99.5]
TAUS = [0.5, 0.6, 0.7, 0.8]
OUT = Path("results/anomaly")


def held_metrics(lab, held, bidx, unk):
    l = lab[held]
    missed, unknown = float((l == bidx).mean()), float((l == unk).mean())
    return {"heldout_detected": 1 - missed, "heldout_unknown": unknown,
            "heldout_misassigned": 1 - missed - unknown, "heldout_missed": missed}


def known_metrics(lab, yk, known, bidx, C):
    l = lab[known]
    att = yk != bidx
    return {"known_macro_f1": f1_score(yk, l, labels=list(range(C)), average="macro", zero_division=0),
            "known_benign_false_alarm": float((l[~att] != bidx).mean()),
            "known_attack_detected": float((l[att] != bidx).mean()),
            "known_attack_unknown": float((l[att] == C).mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(BENIGN))
    ap.add_argument("--detector", default="ae", choices=["ae", "iforest"])
    ap.add_argument("--pctl", type=float, default=99, help="percentile shown in the test table")
    ap.add_argument("--tau", type=float, default=0.6, help="confidence threshold shown in the test table")
    ap.add_argument("--clf-seed", type=int, default=42)
    a = ap.parse_args()

    ds, ben = a.dataset, BENIGN[a.dataset]
    feats = json.load(open(f"results/features/{ds}_topk.json"))[K[ds]]
    params = json.load(open("results/tuning/best_params.json"))[f"{ds}/category/lgbm"]["params"]
    cols = feats + ["category"]
    tr, va, te = (pd.read_parquet(f"data/processed/{ds}/{s}.parquet", columns=cols)
                  for s in ("train", "val", "test"))
    seeds = sorted(int(re.search(r"_s(\d+)_val", f).group(1))
                   for f in glob.glob(str(OUT / f"scores_{ds}_{a.detector}_s*_val.npy")))
    if not seeds:
        raise SystemExit("no saved scores: run anomaly_probe.py with --save-scores first")
    scores = {s: (np.load(OUT / f"scores_{ds}_{a.detector}_s{s}_val.npy"),
                  np.load(OUT / f"scores_{ds}_{a.detector}_s{s}_test.npy")) for s in seeds}
    ben_val = (va.category == ben).to_numpy()
    heldouts = sorted(c for c in tr.category.unique() if c != ben)
    print(f"{ds}/{a.detector}: anomaly seeds {seeds}; held-out categories {heldouts}")

    rows = []
    for h in heldouts:
        t0 = time.time()
        trn = tr[tr.category != h]
        le = LabelEncoder().fit(trn.category)
        C, unk = len(le.classes_), len(le.classes_)
        bidx = int(le.transform([ben])[0])
        y = le.transform(trn.category)
        cw = dict(enumerate(compute_class_weight("balanced", classes=np.arange(C), y=y)))
        clf = LGBMClassifier(class_weight=cw, subsample_freq=1, n_jobs=-1, random_state=a.clf_seed,
                             verbose=-1, **params).fit(trn[feats], y)
        print(f"  held out {h}: classifier trained on {len(trn):,} rows in {time.time() - t0:.0f}s", flush=True)
        for split, df in (("val", va), ("test", te)):
            proba = clf.predict_proba(df[feats])
            pmax, pidx = proba.max(1), proba.argmax(1)
            cat = df.category.to_numpy()
            held, known = cat == h, cat != h
            yk = le.transform(cat[known])
            base = {"dataset": ds, "detector": a.detector, "split": split, "heldout": h}

            def emit(system, lab, seed=-1, pctl=np.nan, tau=np.nan, flagged=None):
                if flagged is None:
                    r = {**held_metrics(lab, held, bidx, unk), **known_metrics(lab, yk, known, bidx, C)}
                else:   # anomaly-only: no attack names, so only detection / false-alarm rates
                    att = yk != bidx
                    r = {"heldout_detected": float(flagged[held].mean()),
                         "known_benign_false_alarm": float(flagged[known][~att].mean()),
                         "known_attack_detected": float(flagged[known][att].mean())}
                rows.append({**base, "system": system, "seed": seed, "pctl": pctl, "tau": tau, **r})

            emit("supervised", pidx)
            for tau in TAUS:
                emit("supervised+reject", np.where(pmax >= tau, pidx, unk), tau=tau)
            for seed in seeds:
                sv, st = scores[seed]
                s = sv if split == "val" else st
                for p in PCTS:
                    flagged = s > np.percentile(sv[ben_val], p)      # threshold from benign VALIDATION scores
                    emit("anomaly-only", None, seed, p, flagged=flagged)
                    for tau in TAUS:
                        lab = np.where(flagged, np.where(pmax >= tau, pidx, unk), bidx)
                        emit("hybrid", lab, seed, p, tau)

    res = pd.DataFrame(rows)
    res.to_csv(OUT / f"hybrid_{ds}_{a.detector}.csv", index=False)

    print("\n== VALIDATION grid: hybrid, mean over held-out categories and anomaly seeds "
          "(use this to choose percentile and tau)")
    g = res[(res.split == "val") & (res.system == "hybrid")].groupby(["pctl", "tau"])[
        ["heldout_detected", "heldout_unknown", "heldout_missed", "known_benign_false_alarm", "known_macro_f1"]].mean()
    print(g.round(3).to_string())

    t = res[(res.split == "test") & (res.heldout.notna())]
    def pick(system, **kw):
        m = t[t.system == system]
        for k_, v in kw.items():
            m = m[np.isclose(m[k_], v)]
        return m.groupby("heldout").mean(numeric_only=True)
    sup, anom = pick("supervised"), pick("anomaly-only", pctl=a.pctl)
    hyb, rej = pick("hybrid", pctl=a.pctl, tau=a.tau), pick("supervised+reject", tau=a.tau)
    tab = pd.DataFrame({
        "sup_detected": sup.heldout_detected, "rej_detected": rej.heldout_detected,
        "rej_unknown": rej.heldout_unknown, "anom_detected": anom.heldout_detected,
        "hyb_detected": hyb.heldout_detected, "hyb_unknown": hyb.heldout_unknown,
        "hyb_misassigned": hyb.heldout_misassigned, "hyb_missed": hyb.heldout_missed,
        "hyb_benignFA": hyb.known_benign_false_alarm, "hyb_knownF1": hyb.known_macro_f1,
        "sup_knownF1": sup.known_macro_f1})
    print(f"\n== TEST, per held-out category (percentile {a.pctl}, tau {a.tau}; mean over anomaly seeds)")
    print("   detected = labelled anything but benign; unknown = labelled 'unknown'; "
          "misassigned = confidently given a wrong known attack name; missed = called benign")
    print(tab.round(3).to_string())
    print("\nsaved", OUT / f"hybrid_{ds}_{a.detector}.csv")


if __name__ == "__main__":
    main()
