"""Optuna tuning: 5-fold stratified CV, macro-F1, on a stratified TRAIN subsample.
  python -m src.train.tune --dataset wsn_ds --level category --models dt,rf,lgbm,xgb,catboost --trials 50
Resumable: trials are stored in results/tuning/optuna.db. Test data is never used.
"""
import argparse
import json
import time
import warnings
from pathlib import Path

import mlflow
import numpy as np
import optuna
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from src.data.common import load_cfg
from src.data.prepare import build
from src.train.benchmark import subsample

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)
K = {"ciciot2023": "30", "wsn_ds": "all"}
OUT = Path("results/tuning")


def space(name, t):
    if name == "lgbm":
        return dict(n_estimators=t.suggest_int("n_estimators", 100, 500),
                    learning_rate=t.suggest_float("learning_rate", 0.02, 0.3, log=True),
                    num_leaves=t.suggest_int("num_leaves", 15, 255, log=True),
                    min_child_samples=t.suggest_int("min_child_samples", 20, 200),
                    reg_lambda=t.suggest_float("reg_lambda", 1e-3, 10, log=True),
                    subsample=t.suggest_float("subsample", 0.6, 1.0),
                    colsample_bytree=t.suggest_float("colsample_bytree", 0.5, 1.0))
    if name == "xgb":
        return dict(n_estimators=t.suggest_int("n_estimators", 100, 500),
                    max_depth=t.suggest_int("max_depth", 3, 12),
                    learning_rate=t.suggest_float("learning_rate", 0.02, 0.3, log=True),
                    subsample=t.suggest_float("subsample", 0.6, 1.0),
                    colsample_bytree=t.suggest_float("colsample_bytree", 0.5, 1.0),
                    min_child_weight=t.suggest_int("min_child_weight", 1, 20),
                    reg_lambda=t.suggest_float("reg_lambda", 1e-3, 10, log=True))
    if name == "catboost":
        return dict(iterations=t.suggest_int("iterations", 100, 500),
                    depth=t.suggest_int("depth", 4, 10),
                    learning_rate=t.suggest_float("learning_rate", 0.03, 0.3, log=True),
                    l2_leaf_reg=t.suggest_float("l2_leaf_reg", 1, 10, log=True))
    if name == "rf":      # depth and leaf size bounded so the model stays small
        return dict(n_estimators=t.suggest_int("n_estimators", 100, 300),
                    max_depth=t.suggest_int("max_depth", 6, 30),
                    min_samples_leaf=t.suggest_int("min_samples_leaf", 1, 20),
                    max_features=t.suggest_categorical("max_features", ["sqrt", "log2", 0.5]))
    if name == "dt":
        return dict(max_depth=t.suggest_int("max_depth", 3, 30),
                    min_samples_leaf=t.suggest_int("min_samples_leaf", 1, 50),
                    criterion=t.suggest_categorical("criterion", ["gini", "entropy"]))
    raise ValueError(name)


def make_model(name, p, cw, seed):
    if name == "lgbm":
        return LGBMClassifier(class_weight=cw, subsample_freq=1, n_jobs=-1, random_state=seed,
                              verbose=-1, **p)
    if name == "xgb":
        return XGBClassifier(tree_method="hist", n_jobs=-1, random_state=seed, **p)
    if name == "catboost":
        return CatBoostClassifier(auto_class_weights="Balanced", thread_count=-1, random_seed=seed,
                                  verbose=0, allow_writing_files=False, **p)
    if name == "rf":
        return RandomForestClassifier(class_weight=cw, n_jobs=-1, random_state=seed, **p)
    if name == "dt":
        return DecisionTreeClassifier(class_weight=cw, random_state=seed, **p)
    raise ValueError(name)


def make_objective(name, X, y, cw, folds, seed):
    splits = list(StratifiedKFold(folds, shuffle=True, random_state=seed).split(X, y))

    def objective(trial):
        p = space(name, trial)
        scores = []
        for i, (tr, va) in enumerate(splits):
            m = make_model(name, p, cw, seed)
            Xtr, ytr = X.iloc[tr], y[tr]
            if name == "xgb":
                sw = pd.Series(ytr).map(cw).to_numpy() if cw else None
                m.fit(Xtr, ytr, sample_weight=sw)
            else:
                m.fit(Xtr, ytr)
            pred = np.asarray(m.predict(X.iloc[va])).ravel().astype(int)
            scores.append(f1_score(y[va], pred, average="macro"))
            trial.report(float(np.mean(scores)), i)
            if trial.should_prune():
                raise optuna.TrialPruned()
        return float(np.mean(scores))
    return objective


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--level", default="category")
    ap.add_argument("--models", default="dt,rf,lgbm,xgb,catboost")
    ap.add_argument("--trials", type=int, default=50)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--subsample", type=int, default=150_000)
    a = ap.parse_args()

    seed = load_cfg()["seed"]
    d = build(a.dataset, k=K[a.dataset], level=a.level, imbalance="class_weight", save=False)
    le = LabelEncoder().fit(d["y_train"])
    X, y = subsample(d["X_train"], le.transform(d["y_train"]), a.subsample, seed)
    cw = {int(le.transform([c])[0]): w for c, w in d["class_weight"].items()}
    print(f"tuning on {len(y):,} stratified train rows, {a.folds}-fold CV, {len(cw)} classes")

    OUT.mkdir(parents=True, exist_ok=True)
    best_path = OUT / "best_params.json"
    best = json.load(open(best_path)) if best_path.exists() else {}
    mlflow.set_experiment("ids-tuning")

    for name in a.models.split(","):
        c = cw
        if name == "lgbm" and len(cw) > 20:        # many-class weights diverged in the benchmark
            c = {k: v ** 0.5 for k, v in cw.items()}
        study = optuna.create_study(
            study_name=f"{a.dataset}_{a.level}_{name}", storage=f"sqlite:///{OUT / 'optuna.db'}",
            load_if_exists=True, direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=seed),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=8))
        done = sum(t.state.name in ("COMPLETE", "PRUNED") for t in study.trials)
        remaining = max(a.trials - done, 0)
        print(f"\n[{name}] {done} trials done, {remaining} to run", flush=True)
        t0 = time.time()
        if remaining:
            study.optimize(make_objective(name, X, y, c, a.folds, seed), n_trials=remaining,
                           callbacks=[lambda s, t: print(f"  trial {t.number}: {t.state.name} "
                                                         f"{t.value if t.value is not None else ''}",
                                                         flush=True)])
        if not any(t.state.name == "COMPLETE" for t in study.trials):
            print(f"[{name}] no completed trials")
            continue
        study.trials_dataframe().to_csv(OUT / f"trials_{a.dataset}_{a.level}_{name}.csv", index=False)
        best[f"{a.dataset}/{a.level}/{name}"] = {
            "params": study.best_params, "cv_macro_f1": study.best_value,
            "tune_rows": len(y), "folds": a.folds, "n_trials": len(study.trials)}
        json.dump(best, open(best_path, "w"), indent=2)
        if remaining:
            with mlflow.start_run(run_name=f"tune_{a.dataset}_{a.level}_{name}"):
                mlflow.log_params({**study.best_params, "dataset": a.dataset, "level": a.level,
                                   "model": name, "tune_rows": len(y), "folds": a.folds})
                mlflow.log_metric("cv_macro_f1", study.best_value)
        print(f"[{name}] best CV macro-F1 = {study.best_value:.4f}  ({(time.time() - t0) / 60:.1f} min)")
        print("   ", study.best_params, flush=True)


if __name__ == "__main__":
    main()
