"""FINAL evaluation on the untouched TEST split. Do not tune anything based on these numbers.
  python -m src.train.final_eval --dataset wsn_ds --level binary --models dt,mlp --out /tmp/t.csv   # smoke test
  python -m src.train.final_eval --dataset ciciot2023 --level category --seeds 42
Resumable: finished (dataset, level, model, seed) rows in the output CSV are skipped.
"""
import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
import sys

_DEEP = {"mlp", "cnn1d"}
_arg = next((sys.argv[i + 1] for i, x in enumerate(sys.argv[:-1]) if x == "--models"),
            "dt,lgbm,xgb,rf,catboost,mlp,cnn1d")
_req = set(_arg.split(","))
if _req & _DEEP and _req - _DEEP:
    raise SystemExit("Run tree models and deep models in SEPARATE commands "
                     "(they crash when loaded together on macOS).")
if _req & _DEEP:
    import torch  # noqa: F401  must be imported BEFORE the tree libraries
import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from src.data.prepare import build
from src.models.evaluate import evaluate, per_class_f1, score_metrics
from src.train.benchmark import PeakRSS, latency, model_size_mb
from src.train.tune import K, make_model

warnings.filterwarnings("ignore")
OUT = Path("results/final")
DEEP = {"mlp", "cnn1d"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--level", required=True)
    ap.add_argument("--models", default="dt,lgbm,xgb,rf,catboost,mlp,cnn1d")
    ap.add_argument("--seeds", default="42")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--out", default=str(OUT / "results_test.csv"))
    a = ap.parse_args()

    (OUT / "details").mkdir(parents=True, exist_ok=True)
    (OUT / "probs").mkdir(parents=True, exist_ok=True)
    out = Path(a.out)
    rows = pd.read_csv(out).to_dict("records") if out.exists() else []
    done = {(r["dataset"], r["level"], r["model"], int(r["seed"])) for r in rows}
    best = json.load(open("results/tuning/best_params.json"))
    models = a.models.split(",")
    seeds = [int(s) for s in a.seeds.split(",")]

    k = K[a.dataset]
    d = build(a.dataset, k=k, level=a.level, imbalance="class_weight", save=False)
    le = LabelEncoder().fit(d["y_train"])
    ytr, yv, yt = (le.transform(d[f"y_{s}"]) for s in ("train", "val", "test"))
    C, n_in = len(le.classes_), d["X_train"].shape[1]
    cw = {int(le.transform([c])[0]): w for c, w in d["class_weight"].items()}
    binary = a.level == "binary"
    print(f"{a.dataset}/{a.level}: classes={C} train={len(ytr):,} test={len(yt):,}")

    if DEEP & set(models):
        import torch
        from src.train.deep import CNN1D, MLP, cpu_latency, logits_of, n_params, size_mb, train
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        soft = C > 20
        wt = torch.tensor([cw[i] ** 0.5 if soft else cw[i] for i in range(C)],
                          dtype=torch.float32).to(device)
        Xtr_t = torch.from_numpy(d["X_train"].to_numpy(dtype=np.float32)).to(device)
        ytr_t = torch.from_numpy(ytr).long().to(device)
        Xv_t = torch.from_numpy(d["X_val"].to_numpy(dtype=np.float32))
        Xte_np = d["X_test"].to_numpy(dtype=np.float32)
        Xte_t = torch.from_numpy(Xte_np)

    for seed in seeds:
        for name in models:
            if (a.dataset, a.level, name, seed) in done:
                print(f"[skip] {name} seed {seed}")
                continue
            print(f"\n[start] {name} seed {seed} at {time.strftime('%H:%M:%S')}", flush=True)
            extra = {}
            try:
                with PeakRSS() as ram:
                    t0 = time.time()
                    if name in DEEP:
                        torch.manual_seed(seed)
                        np.random.seed(seed)
                        mdl = (MLP if name == "mlp" else CNN1D)(n_in, C).to(device)
                        assert n_params(mdl) < 500_000
                        be, le_ep = train(mdl, Xtr_t, ytr_t, Xv_t, yv, wt, device, a.epochs,
                                          2048, 1e-3, a.patience, seed)
                        fit_s = time.time() - t0
                        probs = torch.softmax(logits_of(mdl, Xte_t, device), 1).numpy()
                        pred = probs.argmax(1)
                        extra = {"params": n_params(mdl), "best_epoch": be, "epochs_run": le_ep}
                    else:
                        p = best[f"{a.dataset}/category/{name}"]["params"]
                        w = {c: v ** 0.5 for c, v in cw.items()} if (name == "lgbm" and C > 20) else cw
                        mdl = make_model(name, p, w, seed)
                        if name == "xgb":
                            mdl.fit(d["X_train"], ytr, sample_weight=pd.Series(ytr).map(cw).to_numpy())
                        else:
                            mdl.fit(d["X_train"], ytr)
                        fit_s = time.time() - t0
                        probs = mdl.predict_proba(d["X_test"])
                        pred = np.asarray(mdl.classes_)[probs.argmax(1)].astype(int)
                if name in DEEP:
                    l1, lb, thr = cpu_latency(mdl, Xte_np)
                    msize = size_mb(mdl)
                else:
                    l1, lb, thr = latency(mdl, d["X_test"])
                    msize = model_size_mb(mdl)
            except Exception as e:
                print(f"[FAILED] {name} seed {seed}: {type(e).__name__}: {e}")
                continue

            m, recall, cm = evaluate(le.inverse_transform(yt), le.inverse_transform(pred), a.level)
            f1c = per_class_f1(le.inverse_transform(yt), le.inverse_transform(pred))
            m.update(score_metrics(yt, probs, binary))
            row = {"dataset": a.dataset, "level": a.level, "k": k, "model": name, "seed": seed,
                   "split": "test", "train_rows": len(ytr), "test_rows": len(yt), **m,
                   "fit_seconds": fit_s, "latency_ms_batch1": l1, "latency_ms_per_flow_batch1024": lb,
                   "throughput_flows_s": thr, "model_size_mb": msize, "peak_ram_mb": ram.mb, **extra}
            rows.append(row)
            pd.DataFrame(rows).to_csv(out, index=False)
            tag = f"{a.dataset}_{a.level}_{name}_s{seed}"
            cm.to_csv(OUT / "details" / f"{tag}_confusion.csv")
            pd.DataFrame({"recall": recall, "f1": f1c}).to_csv(OUT / "details" / f"{tag}_per_class.csv")
            np.save(OUT / "probs" / f"{tag}.npy", probs.astype(np.float16))
            print(f"[{name}] TEST macro_f1={row['macro_f1']:.4f} bal_acc={row['balanced_acc']:.4f} "
                  f"FA={row['false_alarm_rate']:.4f} AUC={row.get('roc_auc', float('nan')):.4f} "
                  f"fit={fit_s:.0f}s size={msize:.1f}MB", flush=True)
            del mdl


if __name__ == "__main__":
    main()
