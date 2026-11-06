"""Generate data cards from the pipeline outputs. Usage: python -m src.data.datacards"""
import json
from pathlib import Path

import pandas as pd

from src.data.common import load_cfg

CARDS = {
    "ciciot2023": {
        "title": "CICIoT2023",
        "source": "Canadian Institute for Cybersecurity, IoT attack dataset (flow-level features extracted from pcap captures of an IoT testbed).",
        "task": "Detection of IoT network attacks: binary (benign vs attack), 8-way category, and 34-way attack type.",
        "labels": "`binary` (0 = Benign, 1 = Attack); `category` (Benign, DDoS, DoS, Recon, Mirai, Spoofing, BruteForce, Web); `attack_type` (folder name of the capture, Benign merged into one class).",
        "sampling": "Attack folders under {keep_mb} MB keep 100% of rows (rare classes); larger folders are sampled at {frac:.0%}.",
        "time": "No timestamps. The temporal split orders rows by capture-file number within each attack. This ASSUMES file numbers follow capture order, which the dataset does not guarantee.",
        "limits": [
            "Label conflicts: in the random split, 26,706 test rows (5.6%) have exactly the same features as a train row; 21,299 of them (4.4% of test) carry a different `category`, but only 53 differ at the `binary` level. Category/attack-type accuracy therefore has a ceiling of roughly 96%, while binary detection is essentially unaffected.",
            "Heavy class imbalance (DDoS is 56% of rows, BruteForce 0.4%).",
            "Flow-level features only; no payload content. Results come from a single lab testbed and may not transfer to other networks.",
            "Cleaning removed a large share of duplicate rows, so the processed set is smaller than the published dataset.",
        ],
    },
    "wsn_ds": {
        "title": "WSN-DS",
        "source": "WSN-DS, a wireless sensor network dataset simulated with LEACH clustering (NS-2), with per-node features.",
        "task": "Detection of WSN attacks: binary (Normal vs attack) and 5-way type (Normal, Blackhole, Grayhole, Flooding, TDMA).",
        "labels": "`binary` (0 = Normal, 1 = Attack); `category` and `attack_type` are identical (the dataset has one level of attack naming).",
        "sampling": "Full dataset used, no sampling.",
        "time": "The raw `Time` column (simulation round) is used only to order rows for the temporal split, then dropped.",
        "limits": [
            "Simulated data, not real sensor traffic; results may not transfer to real deployments.",
            "Very imbalanced (Normal is 93% of rows, Flooding about 1%).",
            "Label conflicts: in the random split, 20 test rows share their features with a train row of a different attack type, none at the `binary` level.",
            "Identifier columns (`id`, `Time`, `who CH`) were dropped so models cannot memorise nodes or rounds.",
            "Duplicate rows were removed before splitting.",
        ],
    },
}


def md_table(df):
    cols = ["attack_type"] + list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for idx, row in df.iterrows():
        lines.append("| " + " | ".join([str(idx)] + [f"{int(v):,}" for v in row]) + " |")
    return "\n".join(lines)


def build_card(key, meta, cfg):
    d = Path("data/processed") / key
    log = json.load(open(d / "cleaning_log.json"))
    counts = pd.read_csv(d / "class_counts.csv", index_col=0)
    rank = pd.read_csv(f"results/features/{key}_feature_ranking.csv", index_col=0)
    tlog_path = Path("data/processed") / f"{key}_temporal" / "cleaning_log.json"
    tlog = json.load(open(tlog_path)) if tlog_path.exists() else None

    L = [f"# Data card: {meta['title']}", ""]
    L += ["## Overview", f"- **Source:** {meta['source']}", f"- **Task:** {meta['task']}",
          f"- **Labels:** {meta['labels']}",
          f"- **Sampling:** " + meta["sampling"].format(keep_mb=log.get("keep_all_mb_threshold", "-"),
                                                        frac=log.get("sample_fraction", 1.0)), ""]
    L += ["## Cleaning (identical rows removed before splitting, so duplicates cannot leak across splits)",
          f"- Rows before cleaning: {log['rows_start']:,}",
          f"- Dropped NaN/inf rows: {log['dropped_nan_or_inf']:,}",
          f"- Dropped duplicate rows: {log['dropped_duplicates']:,}",
          f"- Rows after cleaning: {log['rows_after_cleaning']:,}",
          f"- Dropped tiny classes (<10 rows): {log['dropped_tiny_classes'] or 'none'}",
          f"- Dropped constant columns (detected on train only): {log['dropped_constant_columns'] or 'none'}"]
    if "dropped_identifier_columns" in log:
        L.append(f"- Dropped identifier columns: {log['dropped_identifier_columns']}")
    L += [f"- Features kept: {log['n_features']}", ""]
    L += ["## Splits",
          f"- **Random stratified** (seed {cfg['seed']}): train {log['rows_train']:,} / "
          f"val {log['rows_val']:,} / test {log['rows_test']:,}  (`data/processed/{key}/`)"]
    if tlog:
        L.append(f"- **Temporal:** train {tlog['rows_train']:,} / val {tlog['rows_val']:,} / "
                 f"test {tlog['rows_test']:,}  (`data/processed/{key}_temporal/`)")
    L += [f"- Time note: {meta['time']}",
          "- Scaling and class-imbalance handling are fit on the train split only (`src/data/prepare.py`).", ""]
    L += ["## Class counts (random split)", md_table(counts), ""]
    L += ["## Top 10 features (mean of MI, Random Forest and RFE ranks, train split only)"]
    for i, (f, r) in enumerate(rank.head(10).iterrows(), 1):
        L.append(f"{i}. `{f}` (mean rank {r['mean_rank']:.1f})")
    L += ["", "Full ranking: `results/features/" + key + "_feature_ranking.csv`", ""]
    L += ["## Known limitations"] + [f"- {x}" for x in meta["limits"]] + [""]
    L += ["## Reproduce",
          f"`python -m src.data.prep_{'ciciot' if key == 'ciciot2023' else 'wsn'}` then "
          f"`python -m src.data.features {key}` then `python -m src.data.temporal_split {key}`", ""]
    return "\n".join(L)


def main():
    cfg = load_cfg()
    out = Path("docs/data_cards")
    out.mkdir(parents=True, exist_ok=True)
    for key, meta in CARDS.items():
        (out / f"{key}.md").write_text(build_card(key, meta, cfg))
        print("wrote", out / f"{key}.md")


if __name__ == "__main__":
    main()
