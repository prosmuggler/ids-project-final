"""Write docs/data_cards/nbaiot.md from the dataset's own summary files plus a benign-data check.
   python -m src.data.datacard_nbaiot"""
from pathlib import Path

import numpy as np
import pandas as pd

D = Path("data/raw/nbaiot")
info = pd.read_csv(D / "device_info.csv", skipinitialspace=True)
summ = pd.read_csv(D / "data_summary.csv", skipinitialspace=True)
summ.columns = [c.strip() for c in summ.columns]
summ["device"] = summ["File Name"].str.extract(r"^(\d+)\.")[0].astype(int)
summ["kind"] = summ["File Name"].str.replace(r"^\d+\.", "", regex=True).str.replace(".csv", "", regex=False)

rows = []
for dev, name in zip(info.iloc[:, 0], info.iloc[:, 1]):
    s = summ[summ.device == dev]
    b = pd.read_csv(D / f"{dev}.benign.csv", dtype=np.float32)
    nonfinite = int((~np.isfinite(b.to_numpy())).sum())
    dup = float(b.duplicated().mean())
    m, g = s[s.kind.str.startswith("mirai")], s[s.kind.str.startswith("gafgyt")]
    rows.append(f"| {dev} | {name} | {len(b):,} | {len(g)} / {int(g['Data Count'].sum()):,} | "
                f"{len(m)} / {int(m['Data Count'].sum()):,} | {nonfinite} | {dup:.1%} |")

card = f"""# Data card: N-BaIoT

## Overview
- **Source:** N-BaIoT (Meidan et al., 2018): traffic from 9 commercial IoT devices, benign and infected with BASHLITE (gafgyt) and Mirai. Add the download URL you used here.
- **Task here:** semi-supervised anomaly detection, one autoencoder / detector per device, trained on that device's benign traffic only, tested on its Mirai and BASHLITE traffic.
- **Features:** {int(summ['Feature Count'].iloc[0])} damped incremental statistics per row (names like `MI_dir_L5_weight`), computed by the dataset authors. No identifiers, timestamps or label column inside the files.
- **Labels:** from the file name: `<device>.benign.csv`, `<device>.<mirai|gafgyt>.<attack>.csv`.
  BASHLITE attacks: combo, junk, scan, tcp, udp. Mirai attacks: ack, scan, syn, udp, udpplain.
- **Files:** {len(summ)} CSV files, about 8 GB. Devices 3 and 7 have BASHLITE traffic only.

## Devices (rows; attack files / attack rows)
| ID | Device | Benign rows | BASHLITE files / rows | Mirai files / rows | Non-finite values in benign | Duplicate benign rows |
|---|---|---|---|---|---|---|
""" + "\n".join(rows) + """

## Splits
- Benign rows of each device are split **in file order** (assumed to be time order; the files carry no timestamps): first 70% train, next 15% validation (anomaly thresholds), last 15% test. No shuffling, because neighbouring rows are highly correlated.
- Attack files are used only as test data and never seen in training.
- No scaling is applied beyond the autoencoder's own standardisation, fitted on benign training rows only.

## Cleaning
- None applied: the files are used as distributed. The benign files were checked for non-finite values and duplicate rows (table above). The attack files were NOT checked.

## Known limitations
- Row order is assumed to be capture order; this was not verified.
- Benign false-alarm rates on the later test rows exceed the nominal threshold rate on several devices (benign drift), notably devices 2, 3, 7 and 9.
- Devices 2 and 9 have little benign data, so their thresholds are noisy.
- BASHLITE tcp and udp gave identical detection scores; see docs/decisions_log.md for the follow-up.

## Reproduce
`python -m src.train.nbaiot --detector ae --seeds 42,43,44` and `python -m src.train.nbaiot --detector iforest --seeds 42,43,44`
"""
Path("docs/data_cards").mkdir(parents=True, exist_ok=True)
Path("docs/data_cards/nbaiot.md").write_text(card)
print(card)
