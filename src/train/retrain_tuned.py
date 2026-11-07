"""Retrain Optuna-tuned models on the FULL train split, score on VALIDATION, compare with defaults.
  python -m src.train.retrain_tuned --dataset wsn_ds
  python -m src.train.retrain_tuned --dataset ciciot2023 --models dt,lgbm,xgb
Resumable: finished (dataset, model) pairs in results/tuning/retrained_val.csv are skipped.
"""
import argparse
import json
import time
import warnings
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from src.data.common import load_cfg
from src.data.prepare import build
from src.models.evaluate import evaluate, score_metrics
from src.train.benchmark import PeakRSS, latency, model_size_mb
from src.train.tune import K, make_model

warnings.filterwarnings("ignore")
LEVEL = "category"
CSV = Path("results/tuning/retrained_val.csv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--models", default="dt,lgbm,xgb,rf,catboost")
    a = ap.parse_args()

    seed = load_cfg()["seed"]
    best = json.load(open("results/tuning/best_params.json"))
    bench = pd.read_csv("results/benchmark/results.csv")
    rows = pd.read_csv(CSV).to_dict("records") if CSV.exists() else []
    done = {(r["dataset"], r["model"]) for r in rows}

    d = build(a.dataset, k=K[a.dataset], level=LEVEL, imbalance="class_weight", save=False)
    le = LabelEncoder().fit(d["y_train"])
    ytr, yv = le.transform(d["y_train"]), le.transform(d["y_val"])
    cw_enc = {int(le.transform([c])[0]): w for c, w in d["class_weight"].items()}
    mlflow.set_experiment("ids-tuned-retrain")

    for name in a.models.split(","):
        key = f"{a.dataset}/{LEVEL}/{name}"
        if key not in best or (a.dataset, name) in done:
            print(f"[skip] {key}")
            continue
        p = best[key]["params"]
        print(f"[start] {name} {p} at {time.strftime('%H:%M:%S')}", flush=True)
        mdl = make_model(name, p, cw_enc, seed)
        with PeakRSS() as ram:
            t0 = time.time()
            if name == "xgb":
                mdl.fit(d["X_train"], ytr, sample_weight=pd.Series(ytr).map(cw_enc).to_numpy())
            else:
                mdl.fit(d["X_train"], ytr)
            fit_s = time.time() - t0
            scores = mdl.predict_proba(d["X_val"])
        pred = np.asarray(mdl.classes_)[scores.argmax(axis=1)].astype(int)

        m, recall, cm = evaluate(le.inverse_transform(yv), le.inverse_transform(pred), LEVEL)
        m.update(score_metrics(yv, scores, False))
        l1, lb, thr = latency(mdl, d["X_val"])
        row = {"dataset": a.dataset, "level": LEVEL, "model": name, "seed": seed, **m,
               "fit_seconds": fit_s, "latency_ms_batch1": l1, "throughput_flows_s": thr,
               "model_size_mb": model_size_mb(mdl), "peak_ram_mb": ram.mb,
               "cv_macro_f1": best[key]["cv_macro_f1"], "params": json.dumps(p)}
        rows.append(row)
        pd.DataFrame(rows).to_csv(CSV, index=False)
        cm.to_csv(f"results/tuning/{a.dataset}_{LEVEL}_{name}_tuned_confusion.csv")
        with mlflow.start_run(run_name=f"retrain_{a.dataset}_{LEVEL}_{name}"):
            mlflow.log_params({"dataset": a.dataset, "level": LEVEL, "model": name, **p})
            mlflow.log_metrics({x: float(v) for x, v in row.items()
                                if isinstance(v, (int, float, np.floating)) and np.isfinite(v)
                                and x not in ("seed",)})
        print(f"[{name}] macro_f1={row['macro_f1']:.4f} bal_acc={row['balanced_acc']:.4f} "
              f"FA={row['false_alarm_rate']:.4f} size={row['model_size_mb']:.1f}MB "
              f"fit={fit_s:.0f}s", flush=True)

    res = pd.DataFrame(rows)
    res = res[res.dataset == a.dataset]
    base = bench[(bench.dataset == a.dataset) & (bench.level == LEVEL)].set_index("model")
    cmp = pd.DataFrame({
        "default_f1": base.macro_f1.reindex(res.model).values,
        "tuned_f1": res.macro_f1.values,
        "default_FA": base.false_alarm_rate.reindex(res.model).values,
        "tuned_FA": res.false_alarm_rate.values,
        "default_MB": base.model_size_mb.reindex(res.model).values,
        "tuned_MB": res.model_size_mb.values}, index=res.model.values)
    cmp["d_f1"] = cmp.tuned_f1 - cmp.default_f1
    print("\n== tuned vs default (validation, full train set)")
    print(cmp.round(4).to_string())


if __name__ == "__main__":
    main()
