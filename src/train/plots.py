"""Confusion-matrix plots for the best 3 models per block (by mean TEST macro-F1, seed 42) and a list of
the most confused class pairs.   python -m src.train.plots
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = Path("results/final")
SEED = 42
BIN = {"0": "Benign/Normal", "1": "Attack"}


def main():
    d = pd.read_csv(OUT / "results_test.csv")
    ens = OUT / "results_ensemble.csv"
    if ens.exists():
        d = pd.concat([d, pd.read_csv(ens)], ignore_index=True)
    (OUT / "figures").mkdir(exist_ok=True)
    pairs = []
    for (ds, level), g in d.groupby(["dataset", "level"]):
        best = g.groupby("model").macro_f1.mean().sort_values(ascending=False).index[:3].tolist()
        fig, axes = plt.subplots(1, len(best), figsize=(5.2 * len(best), 5), squeeze=False)
        for ax, name in zip(axes[0], best):
            f = OUT / "details" / f"{ds}_{level}_{name}_s{SEED}_confusion.csv"
            if not f.exists():
                ax.set_title(f"{name}: no file")
                ax.axis("off")
                continue
            cm = pd.read_csv(f, index_col=0)
            cm.index, cm.columns = cm.index.astype(str), cm.columns.astype(str)
            if level == "binary":
                cm = cm.rename(index=BIN, columns=BIN)
            ax.imshow(cm.values, cmap="Blues", vmin=0, vmax=1)
            n = len(cm)
            ax.set_xticks(range(n))
            ax.set_xticklabels(cm.columns, rotation=90, fontsize=7 if n > 12 else 8)
            ax.set_yticks(range(n))
            ax.set_yticklabels(cm.index, fontsize=7 if n > 12 else 8)
            if n <= 8:
                for i in range(n):
                    for j in range(n):
                        ax.text(j, i, f"{cm.values[i, j]:.2f}", ha="center", va="center", fontsize=7,
                                color="white" if cm.values[i, j] > 0.5 else "black")
            ax.set_xlabel("predicted")
            ax.set_ylabel("true")
            ax.set_title(f"{name}  (macro-F1 {g[g.model == name].macro_f1.mean():.3f})", fontsize=9)
            for i in range(n):
                for j in range(n):
                    if i != j and cm.values[i, j] >= 0.05:
                        pairs.append({"dataset": ds, "level": level, "model": name, "true": cm.index[i],
                                      "predicted_as": cm.columns[j], "share_of_true": cm.values[i, j]})
        fig.suptitle(f"{ds} / {level}: row-normalised confusion, test split, seed {SEED}", fontsize=10)
        fig.tight_layout()
        fig.savefig(OUT / "figures" / f"confusion_{ds}_{level}.png", dpi=150)
        plt.close(fig)
        print("saved", OUT / "figures" / f"confusion_{ds}_{level}.png", "models:", best)
    p = pd.DataFrame(pairs)
    if len(p):
        p = p.sort_values(["dataset", "level", "model", "share_of_true"], ascending=[True, True, True, False])
        p.to_csv(OUT / "top_confusions.csv", index=False)
        print("\nmost confused class pairs (share of the true class sent to another class), best model per block:")
        for (ds, level), g in p.groupby(["dataset", "level"]):
            print(f"  {ds}/{level}: ", g.head(4)[["model", "true", "predicted_as", "share_of_true"]]
                  .round(2).to_dict("records"))


if __name__ == "__main__":
    main()
