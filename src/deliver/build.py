"""Collect every project deliverable into ONE folder (and optionally one zip). Safe to rerun: the folder is rebuilt.
  python -m src.deliver.build                 # tables + figures + CSVs + data cards
  python -m src.deliver.build --with-data     # also zip data/processed (processed Parquet files)
  python -m src.deliver.build --zip           # also write deliverables.zip
Anything it cannot find is listed at the end instead of stopping the build.
"""
import argparse
import shutil
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("deliverables")
D1, D2 = ROOT / "01_data_pipeline", ROOT / "02_classifier_benchmark"
D3, D4 = ROOT / "03_unknown_attack_detection", ROOT / "04_edge_deployment"
missing = []


def cp(src, dst_dir, name=None):
    src, dst_dir = Path(src), Path(dst_dir)
    if not src.exists():
        missing.append(str(src))
        return
    dst_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst_dir / (name or src.name))


def cp_glob(pattern, dst_dir, skip=()):
    files = sorted(Path().glob(pattern))
    if not files:
        missing.append(pattern)
    for f in files:
        if f.is_file() and not any(s in f.name for s in skip):
            cp(f, dst_dir)


def num(v, f="{:.3f}"):
    return "" if v is None or (isinstance(v, float) and np.isnan(v)) else f.format(v)


def md(df, f="{:.3f}"):
    d = df.reset_index()
    cols = [str(c) for c in d.columns]
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in d.iterrows():
        out.append("| " + " | ".join(num(v, f) if isinstance(v, (float, np.floating)) else str(v) for v in r) + " |")
    return "\n".join(out)


def read(path):
    p = Path(path)
    if not p.exists():
        missing.append(str(p))
        return None
    return pd.read_csv(p)


# ---------------------------------------------------------------- Objective 2 tables
def obj2_tables():
    parts = ["# Objective 2: classifier benchmark (TEST split)\n",
             "Cells are mean +- std over seeds (test split, fixed data split, so std understates uncertainty). "
             "Seeds: 5 for dt, lgbm, xgb, mlp; 3 for rf, catboost, cnn1d. Latency is from the development Mac CPU, "
             "not the Raspberry Pi. `ensemble` = soft vote of the top 3 models chosen on validation.\n"]
    ms = read("results/final/summary_mean_std.csv")
    if ms is not None:
        for metric, title in [("macro_f1", "Macro-F1"), ("balanced_acc", "Balanced accuracy"),
                              ("false_alarm_rate", "False-alarm rate (benign flows flagged)"), ("roc_auc", "ROC-AUC")]:
            m, s = f"{metric}_mean", f"{metric}_std"
            if m not in ms.columns:
                continue
            ms["cell"] = [("" if pd.isna(a) else f"{a:.3f} ± {0 if pd.isna(b) else b:.3f}") for a, b in zip(ms[m], ms[s])]
            parts += [f"## {title}\n", md(ms.pivot(index="model", columns="block", values="cell")), ""]
        if "n_seeds" in ms.columns:
            parts += ["## Seeds per model\n", md(ms.pivot(index="model", columns="block", values="n_seeds"), "{:.0f}"), ""]
    sc = read("results/final/summary_cost.csv")
    if sc is not None:
        fm = {"model_size_mb": "{:.1f}", "latency_ms_batch1": "{:.2f}", "throughput_flows_s": "{:,.0f}",
              "fit_seconds": "{:.0f}", "peak_ram_mb": "{:.0f}"}
        c = sc.copy()
        for k, f in fm.items():
            if k in c.columns:
                c[k] = c[k].map(lambda v, f=f: f.format(v))
        c = c.rename(columns={"model_size_mb": "size MB", "latency_ms_batch1": "latency ms (batch 1)",
                              "throughput_flows_s": "flows/s", "fit_seconds": "train s", "peak_ram_mb": "peak RAM MB"})
        parts += ["## Cost (mean over seeds)\n", md(c.set_index(["block", "model"])), ""]
    (D2 / "tables.md").parent.mkdir(parents=True, exist_ok=True)
    (D2 / "tables.md").write_text("\n".join(parts))


# ---------------------------------------------------------------- Objective 3 tables
POINT = {"ciciot2023": (99.0, 0.7), "wsn_ds": (99.0, 0.6)}      # chosen on VALIDATION
SYS_COLS = ["heldout_detected", "heldout_unknown", "heldout_missed", "known_benign_false_alarm",
            "known_attack_detected", "known_attack_unknown", "known_macro_f1"]


def pick(d, system, pctl=None, tau=None):
    m = d[d.system == system]
    if pctl is not None:
        m = m[np.isclose(m.pctl.astype(float), pctl)]
    if tau is not None:
        m = m[np.isclose(m.tau.astype(float), tau)]
    return m


def obj3_tables():
    parts = ["# Objective 3: detecting attacks the classifier was never trained on\n"]
    # (a) leave-one-category-out and three-system comparison
    for ds, (pctl, tau) in POINT.items():
        h = read(f"results/anomaly/hybrid_{ds}_ae.csv")
        if h is None:
            continue
        t = h[h.split == "test"]
        sy = {"supervised": pick(t, "supervised"), "supervised+reject": pick(t, "supervised+reject", tau=tau),
              "anomaly-only": pick(t, "anomaly-only", pctl), "hybrid (gate)": pick(t, "hybrid", pctl, tau),
              "hybrid-or": pick(t, "hybrid-or", pctl, tau), "hybrid-guard": pick(t, "hybrid-guard", pctl, tau)}
        det = pd.DataFrame({k: v.groupby("heldout").heldout_detected.mean() for k, v in sy.items()})
        unk = pd.DataFrame({k: v.groupby("heldout").heldout_unknown.mean() for k, v in sy.items()})
        cmp_ = pd.DataFrame({k: v[SYS_COLS].mean() for k, v in sy.items()}).T
        cmp_.index.name = "system"
        parts += [f"## {ds}: leave-one-category-out (TEST; anomaly detector = autoencoder; "
                  f"percentile {pctl:g}, confidence {tau:g}, chosen on validation)\n",
                  "Each row: the held-out category was removed from classifier training; detector trained on benign rows only.\n",
                  "### Share of the held-out category that raised an alarm (labelled anything but benign)\n", md(det), "",
                  "### Share labelled \"unknown\"\n", md(unk), "",
                  "### Three-system comparison (mean over held-out categories)\n",
                  "`known_*` columns measure the cost on the categories that were NOT held out.\n", md(cmp_), ""]
    # (b) detector comparison
    for ds in POINT:
        frames = {}
        for det_ in ["iforest", "ae", "ocsvm", "lof"]:
            p = read(f"results/anomaly/probe_{ds}_{det_}.csv")
            if p is not None:
                frames[det_] = p[p.level == "category"].groupby("group")[["auc", "rate_p95", "rate_p99", "rate_p99_5"]].mean()
        if frames:
            parts += [f"## {ds}: benign-only detectors per attack category (TEST)\n",
                      "The benign row shows the false-alarm rate. Isolation Forest and autoencoder: 3 seeds; "
                      "One-Class SVM and LOF: 1 seed, 20,000-row benign subsample.\n"]
            for col, title in [("auc", "ROC-AUC vs benign"), ("rate_p99", "Detection rate at the 99th-percentile threshold"),
                               ("rate_p95", "Detection rate at the 95th-percentile threshold")]:
                parts += [f"### {title}\n", md(pd.DataFrame({k: v[col] for k, v in frames.items()})), ""]
    # (c) N-BaIoT
    ae, itf = read("results/anomaly/nbaiot_ae_dedup.csv"), read("results/anomaly/nbaiot_iforest_dedup.csv")
    if ae is not None and itf is not None:
        parts += ["## N-BaIoT: one benign-only detector per device (deduplicated benign data, 3 seeds)\n",
                  "Benign rows split in time order 70/15/15; thresholds from benign validation percentiles; attack files are test-only.\n",
                  "**Caution:** `gafgyt.tcp` and `gafgyt.udp` contain only ~100 distinct rows each (at least 98% of each file is "
                  "one repeated feature vector), so their rates are a yes/no on one pattern per device, not a measure over ~100,000 flows.\n"]
        cols = ["auc", "rate_p95", "rate_p99", "rate_p99_5"]
        att = {k: v[v.family != "benign"].groupby("group")[cols].mean() for k, v in (("AE", ae), ("IF", itf))}
        both = pd.concat({k: v for k, v in att.items()}, axis=1)
        both.columns = [f"{a} {b}" for a, b in both.columns]
        parts += ["### Per attack type (mean over devices and seeds)\n", md(both), ""]
        ok = ae[(ae.family != "benign") & (~ae.group.isin(["gafgyt.tcp", "gafgyt.udp"]))]
        parts += ["### Autoencoder, the 8 non-degenerate attack types only (mean over all devices)\n",
                  md(ok[cols].mean().to_frame("AE").T.rename_axis("detector")), ""]
        fa = pd.concat({k: v[v.family == "benign"].groupby("device")[["rate_p95", "rate_p99", "rate_p99_5"]].mean()
                        for k, v in (("AE", ae), ("IF", itf))}, axis=1)
        fa.columns = [f"{a} {b}" for a, b in fa.columns]
        parts += ["### Benign false-alarm rate per device (nominal 5% / 1% / 0.5%)\n", md(fa), ""]
    # (d) WSN-DS cluster-head behaviour
    fr = {}
    for det_ in ["iforest", "ae"]:
        p = read(f"results/anomaly/wsn_ch_{det_}.csv")
        if p is not None:
            fr[det_] = p[p.level == "category"].groupby("group")[["auc", "rate_p95", "rate_p99"]].mean()
    tp = Path("data/processed/wsn_ds/test.parquet")
    if fr and tp.exists():
        te = pd.read_parquet(tp, columns=["Is_CH", "Data_Sent_To_BS", "DATA_R", "category"])
        ch = te[te.Is_CH == 1]
        rule = ((ch.Data_Sent_To_BS / (ch.DATA_R + 1)) == 0).groupby(ch.category).mean()
        tab = pd.concat({**{f"{k} {c}": v[c] for k, v in fr.items() for c in ("auc", "rate_p95", "rate_p99")},
                         "rule (forwarded==0)": rule}, axis=1)
        parts += ["## WSN-DS: node behaviour, cluster-head rows only (TEST)\n",
                  "Detectors trained on Normal cluster-head rows only; features include forwarded/received ratio. "
                  "The Normal row is the false-alarm rate on Normal cluster heads. The rule flags a cluster head that forwarded nothing "
                  "to the base station.\n", md(tab), ""]
    (D3 / "tables.md").parent.mkdir(parents=True, exist_ok=True)
    (D3 / "tables.md").write_text("\n".join(parts))


# ---------------------------------------------------------------- assembly
def listing(folder):
    rows = []
    for f in sorted(Path(folder).rglob("*")):
        if f.is_file():
            rows.append(f"- `{f.relative_to(ROOT)}` ({f.stat().st_size / 1e6:.2f} MB)")
    return "\n".join(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--with-data", action="store_true")
    ap.add_argument("--zip", action="store_true")
    a = ap.parse_args()
    if ROOT.exists():
        shutil.rmtree(ROOT)
    for d in (D1, D2, D3, D4):
        d.mkdir(parents=True)

    # Objective 1
    cp_glob("docs/data_cards/*.md", D1 / "data_cards")
    cp_glob("results/features/*", D1 / "feature_rankings")
    cp_glob("results/figures/*.png", D1 / "class_distribution_plots")
    cp_glob("results/imbalance/*.csv", D1 / "imbalance_study")
    for ds in ["ciciot2023", "wsn_ds", "ciciot2023_temporal", "wsn_ds_temporal"]:
        cp(f"data/processed/{ds}/cleaning_log.json", D1 / "cleaning_logs", f"{ds}_cleaning_log.json")
        cp(f"data/processed/{ds}/class_counts.csv", D1 / "cleaning_logs", f"{ds}_class_counts.csv")
    if a.with_data:
        src = Path("data/processed")
        if src.exists():
            with zipfile.ZipFile(D1 / "processed_data_parquet.zip", "w", zipfile.ZIP_DEFLATED) as z:
                for f in sorted(src.rglob("*")):
                    if f.is_file():
                        z.write(f, f.relative_to(src.parent))
        else:
            missing.append(str(src))
    # Objective 2
    cp_glob("results/final/*.csv", D2 / "final_test_results")
    cp_glob("results/final/*.png", D2 / "figures")
    cp_glob("results/final/figures/*.png", D2 / "figures")
    cp_glob("results/benchmark/*.csv", D2 / "default_settings_validation")
    cp_glob("results/tuning/*.json", D2 / "tuning")
    cp("results/tuning/retrained_val.csv", D2 / "tuning")
    cp("results/deep/results.csv", D2 / "deep_models_validation", "deep_results_validation.csv")
    obj2_tables()
    # Objective 3
    cp_glob("results/anomaly/*.csv", D3 / "raw_results")
    obj3_tables()
    # Objective 4 + project-level files
    (D4 / "README.md").write_text("# Objective 4: edge deployment\n\nNot started yet.\n")
    cp("docs/decisions_log.md", ROOT)
    cp("requirements.txt", ROOT)
    cp("README.md", ROOT, "PROJECT_README.md")

    index = ["# Project deliverables\n",
             "Generated by `python -m src.deliver.build`; do not edit by hand, rerun the script instead.\n",
             "- `decisions_log.md`: every deviation from the project guide and every open question.",
             "- `01_data_pipeline/`: data cards, ranked features, class plots, imbalance study, cleaning logs.",
             "- `02_classifier_benchmark/tables.md`: results and cost tables; CSVs, confusion matrices and critical-difference diagrams alongside.",
             "- `03_unknown_attack_detection/tables.md`: leave-one-category-out table, three-system comparison, detector comparison, N-BaIoT, WSN-DS.",
             "- `04_edge_deployment/`: not started.\n", "## Files\n", listing(ROOT)]
    (ROOT / "README.md").write_text("\n".join(index))
    if a.zip:
        shutil.make_archive("deliverables", "zip", ROOT)
    n = sum(1 for f in ROOT.rglob("*") if f.is_file())
    print(f"built {ROOT}/ with {n} files" + (" and deliverables.zip" if a.zip else ""))
    if missing:
        print("\nnot found (skipped):")
        for m in sorted(set(missing)):
            print("  ", m)


if __name__ == "__main__":
    main()
