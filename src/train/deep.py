"""PyTorch MLP and 1D-CNN on single flows. Early stopping on validation macro-F1. VALIDATION only.
  python -m src.train.deep --dataset wsn_ds --level category --models mlp --epochs 3 --out /tmp/t.csv  # smoke test
  python -m src.train.deep --dataset ciciot2023 --level category
Resumable: finished (dataset, level, model, seed) rows in the output CSV are skipped.
"""
import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
import argparse
import copy
import tempfile
import time
import warnings
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder

from src.data.common import load_cfg
from src.data.prepare import build
from src.models.evaluate import evaluate, per_class_f1, score_metrics
from src.train.benchmark import PeakRSS

warnings.filterwarnings("ignore")
K = {"ciciot2023": "30", "wsn_ds": "all"}
DETAILS = Path("results/deep/details")


class MLP(nn.Module):
    def __init__(self, n_in, n_out, hidden=(256, 128, 64), p=0.2):
        super().__init__()
        layers, d = [], n_in
        for h in hidden:
            layers += [nn.Linear(d, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(p)]
            d = h
        layers.append(nn.Linear(d, n_out))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class CNN1D(nn.Module):
    """Features are treated as a 1-channel sequence of length n_in."""
    def __init__(self, n_in, n_out, p=0.2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 32, 3, padding=1), nn.BatchNorm1d(32), nn.ReLU(),
            nn.Conv1d(32, 64, 3, padding=1), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 64, 3, padding=1), nn.BatchNorm1d(64), nn.ReLU())
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(64 * n_in, 128), nn.ReLU(),
                                  nn.Dropout(p), nn.Linear(128, n_out))

    def forward(self, x):
        return self.head(self.conv(x.unsqueeze(1)))


def n_params(m):
    return sum(p.numel() for p in m.parameters())


@torch.inference_mode()
def logits_of(model, X, device, bs=65536):
    model.eval()
    return torch.cat([model(X[i:i + bs].to(device)).float().cpu() for i in range(0, len(X), bs)])


def train(model, Xtr, ytr, Xv, yv_np, w, device, epochs, bs, lr, patience, seed):
    g = torch.Generator().manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=2)
    loss_fn = nn.CrossEntropyLoss(weight=w)
    n = len(ytr)
    best, best_state, best_ep, bad = -1.0, None, 0, 0
    for ep in range(1, epochs + 1):
        model.train()
        t0, tot = time.time(), torch.zeros((), device=device)
        perm = torch.randperm(n, generator=g)
        for i in range(0, n, bs):
            idx = perm[i:i + bs].to(device)
            if len(idx) < 2:
                continue
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(Xtr[idx]), ytr[idx])
            loss.backward()
            opt.step()
            tot += loss.detach() * len(idx)
        pred = logits_of(model, Xv, device).argmax(1).numpy()
        f1 = f1_score(yv_np, pred, average="macro")
        sched.step(f1)
        print(f"  epoch {ep:2d}  loss={tot.item() / n:.4f}  val_macro_f1={f1:.4f}  "
              f"({time.time() - t0:.0f}s)", flush=True)
        if f1 > best + 1e-4:
            best, best_ep, bad = f1, ep, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                print(f"  early stop (best epoch {best_ep}, val macro-F1 {best:.4f})")
                break
    model.load_state_dict(best_state)
    return best_ep, ep


def cpu_latency(model, X_np, n_single=200, batch=1024, reps=10):
    m = copy.deepcopy(model).cpu().eval()
    X = torch.from_numpy(X_np)
    with torch.inference_mode():
        one = []
        for i in range(n_single):
            t0 = time.perf_counter()
            m(X[i:i + 1])
            one.append(time.perf_counter() - t0)
        xb = X[:batch]
        m(xb)
        t0 = time.perf_counter()
        for _ in range(reps):
            m(xb)
        dt = (time.perf_counter() - t0) / reps
    return float(np.median(one) * 1000), dt / len(xb) * 1000, len(xb) / dt


def size_mb(model):
    with tempfile.NamedTemporaryFile(suffix=".pt") as f:
        torch.save(model.state_dict(), f.name)
        return Path(f.name).stat().st_size / 1e6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--level", required=True)
    ap.add_argument("--models", default="mlp,cnn1d")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--weights", default="auto", choices=["auto", "full", "sqrt"])
    ap.add_argument("--out", default="results/deep/results.csv")
    a = ap.parse_args()

    seed = a.seed if a.seed is not None else load_cfg()["seed"]
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    DETAILS.mkdir(parents=True, exist_ok=True)
    rows = pd.read_csv(out).to_dict("records") if out.exists() else []
    done = {(r["dataset"], r["level"], r["model"], r["seed"]) for r in rows}

    k = K[a.dataset]
    d = build(a.dataset, k=k, level=a.level, imbalance="class_weight", save=False)
    le = LabelEncoder().fit(d["y_train"])
    ytr_np, yv_np = le.transform(d["y_train"]), le.transform(d["y_val"])
    C, n_in = len(le.classes_), d["X_train"].shape[1]
    cw = {int(le.transform([c])[0]): w for c, w in d["class_weight"].items()}
    soft = (a.weights == "sqrt") or (a.weights == "auto" and C > 20)
    wt = torch.tensor([cw[i] ** 0.5 if soft else cw[i] for i in range(C)], dtype=torch.float32).to(device)
    print(f"device={device}  classes={C}  features={n_in}  weights={'sqrt' if soft else 'full'}")

    Xtr = torch.from_numpy(d["X_train"].to_numpy(dtype=np.float32)).to(device)
    ytr = torch.from_numpy(ytr_np).long().to(device)
    Xv_np = d["X_val"].to_numpy(dtype=np.float32)
    Xv = torch.from_numpy(Xv_np)
    mlflow.set_experiment("ids-deep")

    for name in a.models.split(","):
        if (a.dataset, a.level, name, seed) in done:
            print(f"[skip] {name}")
            continue
        torch.manual_seed(seed)
        np.random.seed(seed)
        model = (MLP if name == "mlp" else CNN1D)(n_in, C).to(device)
        npar = n_params(model)
        assert npar < 500_000, f"{name} has {npar:,} parameters (limit 500k)"
        print(f"\n[{name}] {npar:,} parameters  start {time.strftime('%H:%M:%S')}", flush=True)
        with PeakRSS() as ram:
            t0 = time.time()
            best_ep, last_ep = train(model, Xtr, ytr, Xv, yv_np, wt, device, a.epochs, a.batch,
                                     a.lr, a.patience, seed)
            fit_s = time.time() - t0
        probs = torch.softmax(logits_of(model, Xv, device), 1).numpy()
        pred = probs.argmax(1)
        m, recall, cm = evaluate(le.inverse_transform(yv_np), le.inverse_transform(pred), a.level)
        f1c = per_class_f1(le.inverse_transform(yv_np), le.inverse_transform(pred))
        m.update(score_metrics(yv_np, probs, a.level == "binary"))
        l1, lb, thr = cpu_latency(model, Xv_np)
        row = {"dataset": a.dataset, "level": a.level, "k": k, "model": name, "seed": seed,
               "imbalance": "class_weight", "train_rows": len(ytr_np), "val_rows": len(yv_np), **m,
               "fit_seconds": fit_s, "latency_ms_batch1": l1, "latency_ms_per_flow_batch1024": lb,
               "throughput_flows_s": thr, "model_size_mb": size_mb(model), "peak_ram_mb": ram.mb,
               "params": npar, "best_epoch": best_ep, "epochs_run": last_ep, "device": str(device)}
        rows.append(row)
        pd.DataFrame(rows).to_csv(out, index=False)
        tag = f"{a.dataset}_{a.level}_{name}_s{seed}"
        cm.to_csv(DETAILS / f"{tag}_confusion.csv")
        pd.DataFrame({"recall": recall, "f1": f1c}).to_csv(DETAILS / f"{tag}_per_class.csv")
        with mlflow.start_run(run_name=tag):
            mlflow.log_params({"dataset": a.dataset, "level": a.level, "model": name, "seed": seed,
                               "params": npar, "batch": a.batch, "lr": a.lr})
            mlflow.log_metrics({x: float(v) for x, v in row.items()
                                if isinstance(v, (int, float, np.floating)) and np.isfinite(v)
                                and x not in ("seed", "train_rows", "val_rows")})
        print(f"[{name}] macro_f1={row['macro_f1']:.4f} bal_acc={row['balanced_acc']:.4f} "
              f"FA={row['false_alarm_rate']:.4f} AUC={row.get('roc_auc', float('nan')):.4f} "
              f"fit={fit_s:.0f}s lat1(cpu)={l1:.2f}ms size={row['model_size_mb']:.2f}MB", flush=True)


if __name__ == "__main__":
    main()
