"""Summary tables (mean +- std over seeds), Friedman test, Nemenyi post-hoc, critical-difference diagrams.
Reads results/final/results_test.csv (+ results_ensemble.csv if present).   python -m src.train.stats
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare

OUT = Path("results/final")
# Nemenyi critical values q_alpha at alpha = 0.05 for k = 2..10 models (Demsar 2006)
Q05 = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164}
QUALITY = ["macro_f1", "balanced_acc", "false_alarm_rate", "roc_auc"]
COST = ["model_size_mb", "latency_ms_batch1", "throughput_flows_s", "fit_seconds", "peak_ram_mb"]


def load():
    d = pd.read_csv(OUT / "results_test.csv")
    ens = OUT / "results_ensemble.csv"
    if ens.exists():
        d = pd.concat([d, pd.read_csv(ens)], ignore_index=True)
    d["block"] = d.dataset + "/" + d.level
    return d


def summaries(d):
    g = d.groupby(["block", "model"])
    q = g[QUALITY].agg(["mean", "std"])
    q.columns = [f"{a}_{b}" for a, b in q.columns]
    q["n_seeds"] = g.seed.nunique()
    q.round(4).to_csv(OUT / "summary_mean_std.csv")
    g[COST].mean().round(3).to_csv(OUT / "summary_cost.csv")
    print("== test macro-F1, mean +- std over seeds (n)")
    t = g.macro_f1.agg(["mean", "std", "count"]).reset_index()
    t["cell"] = t.apply(lambda r: f"{r['mean']:.3f}+-{0 if pd.isna(r['std']) else r['std']:.3f} ({int(r['count'])})", axis=1)
    print(t.pivot(index="model", columns="block", values="cell").to_string())
    print("\n== cost (mean over seeds)")
    print(g[COST].mean().round(2).to_string())


def cd_diagram(avg, cd, title, path):
    s = avg.sort_values()
    names, ranks, k = s.index.tolist(), s.values, len(s)
    fig, ax = plt.subplots(figsize=(9, 3.4 + 0.12 * k))
    ax.set_xlim(0.7, k + 0.3)
    ax.set_ylim(-1.6, 1.9)
    ax.axhline(0, color="black", lw=1)
    for t in range(1, k + 1):
        ax.plot([t, t], [0, -0.06], color="black", lw=1)
        ax.text(t, -0.2, str(t), ha="center", va="top", fontsize=8)
    for i, (n, r) in enumerate(zip(names, ranks)):
        y = 0.45 + 0.28 * (i % 4)
        ax.plot([r, r], [0, y], color="gray", lw=0.8)
        ax.plot(r, 0, "o", color="tab:blue", ms=4)
        ax.text(r, y + 0.03, f"{n} ({r:.2f})", ha="center", va="bottom", fontsize=8)
    x0 = 1.0
    ax.plot([x0, min(x0 + cd, k + 0.3)], [-0.75, -0.75], color="black", lw=2)
    ax.text(x0, -0.62, f"CD = {cd:.2f}" + ("  (wider than the axis)" if cd > k - 0.7 else ""),
            fontsize=8, va="bottom")
    last, level = -1, 0
    for i in range(k):                      # maximal groups that are NOT significantly different
        j = max(jj for jj in range(i, k) if ranks[jj] - ranks[i] <= cd)
        if j > i and j > last:
            ax.plot([ranks[i], ranks[j]], [-1.05 - 0.15 * level] * 2, color="tab:red", lw=3)
            level += 1
            last = j
    ax.set_title(title, fontsize=10)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def friedman_nemenyi(mat, tag, title):
    mat = mat.dropna(axis=0, how="any")
    n, k = mat.shape
    if n < 2 or k < 3 or k not in Q05:
        print(f"[{tag}] skipped (blocks={n}, models={k})")
        return
    stat, p = friedmanchisquare(*[mat[c].values for c in mat.columns])
    ranks = mat.rank(axis=1, ascending=False)       # rank 1 = best macro-F1
    avg = ranks.mean()
    cd = Q05[k] * np.sqrt(k * (k + 1) / (6.0 * n))
    ranks.to_csv(OUT / f"ranks_{tag}.csv")
    cd_diagram(avg, cd, f"{title}\nFriedman chi2={stat:.2f}, p={p:.3f}; {n} blocks, {k} models; "
                        f"Nemenyi alpha=0.05", OUT / f"cd_diagram_{tag}.png")
    print(f"\n[{tag}] blocks={n} models={k}  Friedman chi2={stat:.2f} p={p:.4f}  Nemenyi CD={cd:.2f}")
    print("average ranks (1 = best):")
    print(avg.sort_values().round(2).to_string())
    sig = [(a, b, round(float(abs(avg[a] - avg[b])), 2)) for i, a in enumerate(avg.index)
           for b in avg.index[i + 1:] if abs(avg[a] - avg[b]) > cd]
    print("significantly different pairs:", sig if sig else "none")


def main():
    d = load()
    summaries(d)
    m = d.pivot_table(index="block", columns="model", values="macro_f1", aggfunc="mean")
    full = [c for c in m.columns if m[c].notna().all()]
    print("\nblocks:", list(m.index))
    # A: every model, only blocks where all of them were run (excludes attack_type: no rf/catboost there)
    A = m.dropna(axis=0, how="any")
    friedman_nemenyi(A, "all_models", "All models (blocks where every model was run)")
    # B: only models that were run on every block
    friedman_nemenyi(m[full], "all_blocks", "Models run on every block")
    print("\nWith so few blocks the Nemenyi test has little power: a non-significant result does not "
          "mean the models are equal.")


if __name__ == "__main__":
    main()
