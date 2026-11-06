"""Read-only checks + summary of results/benchmark/results.csv"""
import pandas as pd

d = pd.read_csv("results/benchmark/results.csv")
MODELS = ["lr", "nb", "knn", "dt", "rf", "svm_linear", "xgb", "lgbm", "catboost"]
RUNS = [("wsn_ds", "binary"), ("wsn_ds", "category"), ("ciciot2023", "binary"),
        ("ciciot2023", "category"), ("ciciot2023", "attack_type")]
d["run"] = d.dataset + "/" + d.level

print("rows:", len(d), "(expected 45)")
print("duplicate rows:", int(d.duplicated(["dataset", "level", "k", "model", "seed"]).sum()))
for ds, lv in RUNS:
    g = d[(d.dataset == ds) & (d.level == lv)]
    print(f"{ds:11} {lv:12} models={len(g)}  missing={sorted(set(MODELS) - set(g.model))}")
print("rows with NaN core metrics:",
      int(d[["macro_f1", "balanced_acc", "false_alarm_rate"]].isna().any(axis=1).sum()))
print("capped runs (train_rows < full):")
print(d.groupby(["run", "model"]).train_rows.first().unstack().loc[:, ["lr", "knn", "svm_linear", "catboost"]].to_string())

med = d.groupby("run").macro_f1.transform("median")
sus = d[d.macro_f1 < 0.5 * med]
print("\nsuspiciously low macro-F1 (< half the median of its run):")
print(sus[["run", "model", "macro_f1", "roc_auc"]].to_string(index=False) if len(sus) else "  none")

def piv(col):
    return d.pivot_table(index="model", columns="run", values=col).reindex(MODELS)

for col in ["macro_f1", "balanced_acc", "false_alarm_rate", "model_size_mb", "fit_seconds"]:
    t = piv(col)
    t.to_csv(f"results/benchmark/summary_{col}.csv")
    print(f"\n== {col}")
    print(t.round(3 if "size" not in col and "fit" not in col else 1).to_string())

rank = (piv("macro_f1").rank(ascending=False).mean(axis=1) +
        piv("balanced_acc").rank(ascending=False).mean(axis=1)) / 2
rank = rank.sort_values()
rank.to_csv("results/benchmark/summary_mean_rank.csv", header=["mean_rank"])
print("\n== mean rank over macro-F1 and balanced accuracy across all runs (lower = better)")
print(rank.round(2).to_string())
