"""Confusion matrix + k comparison + binary score on CICIoT2023 (validation)."""
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, f1_score

from src.data.common import load_cfg
from src.data.prepare import build

seed = load_cfg()["seed"]


def fit_eval(k, level):
    d = build("ciciot2023", k=k, level=level, imbalance="class_weight", save=False)
    rf = RandomForestClassifier(n_estimators=60, n_jobs=-1, random_state=seed,
                                class_weight=d["class_weight"]).fit(d["X_train"], d["y_train"])
    pred = rf.predict(d["X_val"])
    return d["y_val"], pred, f1_score(d["y_val"], pred, average="macro")

for k in ["20", "all"]:
    y, p, f = fit_eval(k, "category")
    print(f"category, k={k}: macro_f1={f:.4f}")
    if k == "20":
        labels = sorted(set(y))
        cm = pd.DataFrame(confusion_matrix(y, p, labels=labels, normalize="true"),
                          index=labels, columns=labels)
        print("\nrow = true class, column = predicted (share of the row)")
        print(cm.round(2).to_string())

y, p, f = fit_eval("20", "binary")
print(f"\nbinary, k=20: macro_f1={f:.4f}")
