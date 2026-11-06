"""Shared evaluation: every model is scored the same way."""
import pandas as pd
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix,
                             f1_score, precision_score, recall_score)

BENIGN = {"binary": 0, "category": "Benign", "attack_type": "Benign"}


def evaluate(y_true, y_pred, level):
    labels = sorted(set(y_true) | set(y_pred))
    per_class = pd.Series(recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0),
                          index=labels, name="recall")
    cm = pd.DataFrame(confusion_matrix(y_true, y_pred, labels=labels, normalize="true"),
                      index=labels, columns=labels)
    m = {"accuracy": accuracy_score(y_true, y_pred),
         "balanced_acc": balanced_accuracy_score(y_true, y_pred),
         "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
         "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
         "macro_precision": precision_score(y_true, y_pred, average="macro", zero_division=0)}
    b = BENIGN[level]
    if b in per_class.index:
        m["false_alarm_rate"] = 1 - per_class[b]   # share of benign flows NOT recognised as benign
    return m, per_class, cm
