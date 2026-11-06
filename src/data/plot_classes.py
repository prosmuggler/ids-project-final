"""Log-scale class distribution per dataset. Usage: python -m src.data.plot_classes"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

out = Path("results/figures")
out.mkdir(parents=True, exist_ok=True)

for key, title in [("ciciot2023", "CICIoT2023"), ("wsn_ds", "WSN-DS")]:
    c = pd.read_csv(f"data/processed/{key}/class_counts.csv", index_col=0).sum(axis=1)
    c = c.sort_values()
    fig, ax = plt.subplots(figsize=(8, max(3, 0.28 * len(c))))
    ax.barh(c.index, c.values)
    ax.set_xscale("log")
    ax.set_xlabel("rows (log scale)")
    ax.set_title(f"{title}: class distribution after cleaning")
    fig.tight_layout()
    fig.savefig(out / f"{key}_class_distribution.png", dpi=150)
    print("saved", out / f"{key}_class_distribution.png")
