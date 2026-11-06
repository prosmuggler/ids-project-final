"""Benchmark classical + boosting models on VALIDATION, with cost metrics, logged to MLflow.
  python -m src.train.benchmark --dataset wsn_ds --level binary --models dt,rf,xgb   # quick test
  python -m src.train.benchmark --dataset wsn_ds                                     # one dataset
  python -m src.train.benchmark                                                      # all runs in the config
Finished (dataset, level, k, model, seed) rows are skipped, so you can stop and resume.
"""
import argparse
import tempfile
import threading
import time
import warnings
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd
import psutil
import yaml
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from src.data.common import load_cfg
from src.data.prepare import build
from src.models.evaluate import evaluate, per_class_f1, score_metrics

warnings.filterwarnings("ignore")
OUT = Path("results/benchmark")
CSV = OUT / "results.csv"


def make(name, seed, cw):
    if name == "lr":
        return LogisticRegression(max_iter=100, class_weight=cw, random_state=seed)
    if name == "nb":
        return GaussianNB()
    if name == "knn":
        return KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
    if name == "dt":
        return DecisionTreeClassifier(class_weight=cw, random_state=seed)
    if name == "rf":
        return RandomForestClassifier(n_estimators=200, n_jobs=-1, class_weight=cw, random_state=seed)
    if name == "svm_linear":
        return LinearSVC(dual=False, class_weight=cw, max_iter=2000, random_state=seed)
    if name == "xgb":
        return XGBClassifier(n_estimators=200, max_depth=8, learning_rate=0.1, tree_method="hist",
                             n_jobs=-1, random_state=seed)
    if name == "lgbm":
        # with 34 classes the full balanced weights (up to ~100x) made LightGBM diverge (macro-F1 0.16),
        # so for many-class problems use the square root of the balanced weights
        w = {k: float(v) ** 0.5 for k, v in cw.items()} if (cw and len(cw) > 20) else cw
        return LGBMClassifier(n_estimators=200, num_leaves=63, learning_rate=0.1, n_jobs=-1,
                              class_weight=w, random_state=seed, verbose=-1)
    if name == "catboost":
        return CatBoostClassifier(iterations=300, depth=8, learning_rate=0.1, thread_count=-1,
                                  auto_class_weights="Balanced", random_seed=seed, verbose=0)
    raise ValueError(name)


def subsample(X, y, cap, seed, min_per_class=200):
    """Proportional subsample that keeps at least min_per_class rows of every class."""
    y = np.asarray(y)
    if not cap or len(y) <= cap:
        return X, y
    rng, idx = np.random.default_rng(seed), []
    frac = cap / len(y)
    for c in np.unique(y):
        ii = np.flatnonzero(y == c)
        idx.append(rng.choice(ii, min(len(ii), max(int(round(len(ii) * frac)), min_per_class)),
                              replace=False))
    idx = np.sort(np.concatenate(idx))
    return X.iloc[idx].reset_index(drop=True), y[idx]


class PeakRSS:
    """Peak resident memory of this process above its starting level (approximate, in MB)."""
    def __enter__(self):
        self.p = psutil.Process()
        self.base = self.peak = self.p.memory_info().rss
        self._stop = False
        self.t = threading.Thread(target=self._run, daemon=True)
        self.t.start()
        return self

    def _run(self):
        while not self._stop:
            self.peak = max(self.peak, self.p.memory_info().rss)
            time.sleep(0.05)

    def __exit__(self, *a):
        self._stop = True
        self.t.join()
        self.peak = max(self.peak, self.p.memory_info().rss)

    @property
    def mb(self):
        return (self.peak - self.base) / 1e6


def get_scores(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)
    if hasattr(model, "decision_function"):
        return model.decision_function(X)
    return None


def latency(model, X, n_single=200, batch=1024, reps=10):
    one = []
    for i in range(n_single):
        x = X.iloc[[i]]
        t0 = time.perf_counter()
        model.predict(x)
        one.append(time.perf_counter() - t0)
    xb = X.iloc[:batch]
    model.predict(xb)                                   # warm-up
    t0 = time.perf_counter()
    for _ in range(reps):
        model.predict(xb)
    dt = (time.perf_counter() - t0) / reps
    return float(np.median(one) * 1000), dt / len(xb) * 1000, len(xb) / dt


def model_size_mb(model):
    with tempfile.NamedTemporaryFile(suffix=".joblib") as f:
        joblib.dump(model, f.name)
        return Path(f.name).stat().st_size / 1e6


def run_group(run, models, seed, force, cfg, df_done):
    ds, level, k = run["dataset"], run["level"], str(run["k"])
    done = set() if force else set(zip(df_done.dataset, df_done.level, df_done.k.astype(str),
                                       df_done.model, df_done.seed))
    pending = [m for m in models if (ds, level, k, m, seed) not in done]
    if not pending:
        print(f"[skip] {ds}/{level}/k={k}: nothing to do")
        return df_done

    d = build(ds, k=k, level=level, imbalance=cfg["imbalance"], save=False)
    le = LabelEncoder().fit(d["y_train"])
    ytr_full, yv_full = le.transform(d["y_train"]), le.transform(d["y_val"])
    cw = d["class_weight"]
    cw_enc = {int(le.transform([c])[0]): w for c, w in cw.items()} if cw else None
    binary = level == "binary"
    print(f"\n===== {ds} | {level} | k={k} | train {len(ytr_full):,} | val {len(yv_full):,} =====")

    for name in pending:
        caps = cfg.get("caps", {}).get(name, {})
        Xtr, ytr = subsample(d["X_train"], ytr_full, caps.get("train"), seed)
        Xv, yv = subsample(d["X_val"], yv_full, caps.get("val"), seed, min_per_class=50)
        tag = f"{ds}_{level}_k{k}_{name}_s{seed}"
        try:
            model = make(name, seed, cw_enc)
            with PeakRSS() as ram:
                t0 = time.time()
                if name == "xgb" and cw_enc:
                    model.fit(Xtr, ytr, sample_weight=pd.Series(ytr).map(cw_enc).to_numpy())
                else:
                    model.fit(Xtr, ytr)
                fit_s = time.time() - t0
                scores = get_scores(model, Xv)
                if scores is not None and hasattr(model, "predict_proba"):
                    pred = np.asarray(model.classes_)[scores.argmax(axis=1)]
                else:
                    pred = np.asarray(model.predict(Xv)).ravel()
            pred = np.asarray(pred).astype(int)

            m, recall, cm = evaluate(le.inverse_transform(yv), le.inverse_transform(pred), level)
            f1c = per_class_f1(le.inverse_transform(yv), le.inverse_transform(pred))
            m.update(score_metrics(yv, scores, binary))
            l1, lb, thr = latency(model, Xv)
            row = {"dataset": ds, "level": level, "k": k, "model": name, "seed": seed,
                   "imbalance": cfg["imbalance"], "train_rows": len(ytr), "val_rows": len(yv),
                   **m, "fit_seconds": fit_s, "latency_ms_batch1": l1,
                   "latency_ms_per_flow_batch1024": lb, "throughput_flows_s": thr,
                   "model_size_mb": model_size_mb(model), "peak_ram_mb": ram.mb}
        except Exception as e:
            print(f"[FAILED] {tag}: {type(e).__name__}: {e}")
            continue

        cm.to_csv(OUT / "details" / f"{tag}_confusion.csv")
        pd.DataFrame({"recall": recall, "f1": f1c}).to_csv(OUT / "details" / f"{tag}_per_class.csv")
        with mlflow.start_run(run_name=tag):
            mlflow.log_params({x: row[x] for x in ["dataset", "level", "k", "model", "seed",
                                                   "imbalance", "train_rows", "val_rows"]})
            mlflow.log_metrics({x: float(v) for x, v in row.items()
                                if isinstance(v, (int, float, np.floating)) and np.isfinite(v)
                                and x not in ("seed", "train_rows", "val_rows")})
            mlflow.log_artifact(str(OUT / "details" / f"{tag}_confusion.csv"))
            mlflow.log_artifact(str(OUT / "details" / f"{tag}_per_class.csv"))
        df_done = pd.concat([df_done, pd.DataFrame([row])], ignore_index=True)
        df_done.to_csv(CSV, index=False)
        print(f"[{name:10}] macro_f1={row['macro_f1']:.4f} bal_acc={row['balanced_acc']:.4f} "
              f"FA={row['false_alarm_rate']:.4f} miss={row['attack_miss_rate']:.4f} "
              f"AUC={row.get('roc_auc', float('nan')):.4f} fit={fit_s:.0f}s "
              f"lat1={l1:.2f}ms thr={thr:,.0f}/s size={row['model_size_mb']:.1f}MB "
              f"ram={ram.mb:.0f}MB", flush=True)
    return df_done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset")
    ap.add_argument("--level")
    ap.add_argument("--models")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    cfg = yaml.safe_load(open("configs/benchmark.yaml"))
    seed = a.seed if a.seed is not None else load_cfg()["seed"]
    models = a.models.split(",") if a.models else cfg["models"]
    runs = [r for r in cfg["runs"] if (not a.dataset or r["dataset"] == a.dataset)
            and (not a.level or r["level"] == a.level)]
    (OUT / "details").mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(CSV) if CSV.exists() else pd.DataFrame(
        columns=["dataset", "level", "k", "model", "seed"])
    mlflow.set_experiment(cfg["experiment"])
    for r in runs:
        df = run_group(r, models, seed, a.force, cfg, df)
    print("\nAll results in", CSV)


if __name__ == "__main__":
    main()
