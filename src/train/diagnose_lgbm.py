"""Why does LightGBM fail on 34 classes? Try three variants. Usage: python -m src.train.diagnose_lgbm"""
import lightgbm as lgb
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder

from src.data.common import load_cfg
from src.data.prepare import build

seed = load_cfg()["seed"]
d = build("ciciot2023", k="30", level="attack_type", imbalance="class_weight", save=False)
le = LabelEncoder().fit(d["y_train"])
ytr, yv = le.transform(d["y_train"]), le.transform(d["y_val"])
cw = {int(le.transform([c])[0]): w for c, w in d["class_weight"].items()}
soft = {k: v ** 0.5 for k, v in cw.items()}

variants = {
    "A: no weights, lr=0.1 (original settings)": dict(class_weight=None, learning_rate=0.1),
    "B: sqrt weights, lr=0.05, min_child=100, lambda=5": dict(class_weight=soft, learning_rate=0.05,
                                                              min_child_samples=100, reg_lambda=5),
    "C: no weights, lr=0.05, min_child=100, lambda=5": dict(class_weight=None, learning_rate=0.05,
                                                            min_child_samples=100, reg_lambda=5),
}
for name, kw in variants.items():
    m = LGBMClassifier(n_estimators=200, num_leaves=63, n_jobs=-1, random_state=seed, verbose=-1, **kw)
    m.fit(d["X_train"], ytr, eval_set=[(d["X_val"], yv)],
          callbacks=[lgb.early_stopping(20, verbose=False), lgb.log_evaluation(0)])
    pred = m.predict(d["X_val"])
    loss = m.evals_result_["valid_0"]["multi_logloss"]
    print(f"{name}\n   macro_f1={f1_score(yv, pred, average='macro'):.4f} "
          f"bal_acc={balanced_accuracy_score(yv, pred):.4f} best_iter={m.best_iteration_} "
          f"val_logloss: first={loss[0]:.3f} min={min(loss):.3f} last={loss[-1]:.3f}", flush=True)
