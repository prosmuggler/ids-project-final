"""Shared evaluation: every model is scored the same way."""
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
                             confusion_matrix, f1_score, precision_score, recall_score,
                             roc_auc_score, roc_curve)
from sklearn.preprocessing import label_binarize

BENIGN_NAMES = {"Benign", "Normal", 0}   # benign label in CICIoT2023 / WSN-DS / binary level


def evaluate(y_true, y_pred, level=None):
    """Returns (metrics dict, per-class recall, row-normalised confusion matrix)."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    labels = sorted(set(y_true.tolist()) | set(y_pred.tolist()))
    recall = pd.Series(recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0),
                       index=labels, name="recall")
    cm = pd.DataFrame(confusion_matrix(y_true, y_pred, labels=labels, normalize="true"),
                      index=labels, columns=labels)
    tb = pd.Series(y_true).isin(BENIGN_NAMES).to_numpy()   # truly benign
    pb = pd.Series(y_pred).isin(BENIGN_NAMES).to_numpy()   # predicted benign
    m = {"accuracy": accuracy_score(y_true, y_pred),
         "balanced_acc": balanced_accuracy_score(y_true, y_pred),
         "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
         "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
         "macro_precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
         # false-positive rate: benign flows flagged as any attack
         "false_alarm_rate": float((~pb[tb]).mean()) if tb.any() else float("nan"),
         # attacks that slipped through as benign
         "attack_miss_rate": float(pb[~tb].mean()) if (~tb).any() else float("nan")}
    return m, recall, cm


def per_class_f1(y_true, y_pred):
    labels = sorted(set(np.asarray(y_true).tolist()) | set(np.asarray(y_pred).tolist()))
    return pd.Series(f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0),
                     index=labels, name="f1")


def _recall_at_fpr(y, s, fpr_max):
    fpr, tpr, _ = roc_curve(y, s)
    return float(tpr[fpr <= fpr_max].max())


def score_metrics(y, s, binary, fa_rates=(0.01, 0.05)):
    """Threshold-free metrics from probabilities (or decision scores). y = encoded labels.
    Binary: class 1 = attack. Returns {} when scores are unavailable."""
    out = {}
    if s is None:
        return out
    s = np.asarray(s)
    try:
        if binary:
            s1 = s[:, 1] if s.ndim == 2 else s
            out["roc_auc"] = roc_auc_score(y, s1)
            out["pr_auc"] = average_precision_score(y, s1)
            out["pr_auc_benign"] = average_precision_score(1 - y, -s1)
            for r in fa_rates:
                out[f"attack_recall_at_{int(r * 100)}pct_fa"] = _recall_at_fpr(y, s1, r)
        elif s.ndim == 2 and np.allclose(s[:1000].sum(axis=1), 1.0, atol=1e-3):  # real probabilities only
            out["roc_auc"] = roc_auc_score(y, s, multi_class="ovr", average="macro")
            out["pr_auc"] = average_precision_score(label_binarize(y, classes=range(s.shape[1])),
                                                    s, average="macro")
    except Exception as e:           # e.g. a class missing from a validation subsample
        print("  (score metrics skipped:", type(e).__name__, e, ")")
    return out
