"""Soft-voting ensemble over the top 3 models, built from SAVED test probabilities (no retraining).
Members are chosen per (dataset, level) from VALIDATION results only:
score = mean(macro-F1, balanced accuracy) in results/benchmark/results.csv and results/deep/results.csv
(default-setting validation runs), restricted to models that have final test probabilities.
  python -m src.train.ensemble
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from src.models.evaluate import evaluate, per_class_f1, score_metrics

OUT = Path("results/final")
TOP = 3


def pick_members(val, avail, ds, level):
    v = val[(val.dataset == ds) & (val.level == level) & (val.model.isin(avail))]
    s = v.groupby("model").score.mean().sort_values(ascending=False)
    return s.index[:TOP].tolist(), s


def load_probs(ds, level, model, seed):
    p = OUT / "probs" / f"{ds}_{level}_{model}_s{seed}.npy"
    if not p.exists():
        return None
    x = np.load(p).astype(np.float32)
    return x / x.sum(axis=1, keepdims=True)          # undo float16 rounding


def main():
    res = pd.read_csv(OUT / "results_test.csv")
    res = res[res.model != "ensemble"]
    val = pd.concat([pd.read_csv("results/benchmark/results.csv"),
                     pd.read_csv("results/deep/results.csv")], ignore_index=True)
    val["score"] = (val.macro_f1 + val.balanced_acc) / 2

    rows = []
    for (ds, level), g in res.groupby(["dataset", "level"]):
        members, s = pick_members(val, set(g.model), ds, level)
        print(f"\n{ds}/{level}: members {members}  (validation score "
              f"{', '.join(f'{m}={s[m]:.3f}' for m in members)})")
        tr = pd.read_parquet(f"data/processed/{ds}/train.parquet", columns=[level])[level]
        te = pd.read_parquet(f"data/processed/{ds}/test.parquet", columns=[level])[level]
        le = LabelEncoder().fit(tr)
        yt = le.transform(te)
        seeds = sorted(set.intersection(*[set(g[g.model == m].seed) for m in members]))
        for seed in seeds:
            ps = [load_probs(ds, level, m, seed) for m in members]
            if any(p is None for p in ps):
                print(f"  seed {seed}: missing probabilities, skipped")
                continue
            probs = np.mean(ps, axis=0)
            probs /= probs.sum(axis=1, keepdims=True)
            pred = probs.argmax(1)
            m, recall, cm = evaluate(le.inverse_transform(yt), le.inverse_transform(pred), level)
            f1c = per_class_f1(le.inverse_transform(yt), le.inverse_transform(pred))
            m.update(score_metrics(yt, probs, level == "binary"))
            mem = g[(g.seed == seed) & g.model.isin(members)]
            rows.append({"dataset": ds, "level": level, "k": mem.k.iloc[0], "model": "ensemble",
                         "seed": seed, "split": "test", "test_rows": len(yt), **m,
                         "fit_seconds": mem.fit_seconds.sum(),
                         "latency_ms_batch1": mem.latency_ms_batch1.sum(),
                         "throughput_flows_s": 1.0 / (1.0 / mem.throughput_flows_s).sum(),
                         "model_size_mb": mem.model_size_mb.sum(), "peak_ram_mb": mem.peak_ram_mb.max(),
                         "members": "+".join(members)})
            tag = f"{ds}_{level}_ensemble_s{seed}"
            cm.to_csv(OUT / "details" / f"{tag}_confusion.csv")
            pd.DataFrame({"recall": recall, "f1": f1c}).to_csv(OUT / "details" / f"{tag}_per_class.csv")
            print(f"  seed {seed}: TEST macro_f1={m['macro_f1']:.4f} bal_acc={m['balanced_acc']:.4f} "
                  f"FA={m['false_alarm_rate']:.4f}")
    pd.DataFrame(rows).to_csv(OUT / "results_ensemble.csv", index=False)
    print("\nsaved", OUT / "results_ensemble.csv", f"({len(rows)} rows)")


if __name__ == "__main__":
    main()
