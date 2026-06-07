"""
V7 Training: Poisson-Score-Modell + Dixon-Coles + echte Kalibrierung + Ensemble.

Verbesserungen gegenüber V6:
  - Goal-Kopf: Poisson-NLL statt SmoothL1 → lambda direkt gelernt
  - W/D/L aus Score-Gitter (Dixon-Coles) statt separatem Klassifikations-Kopf
  - Kein label_smoothing, keine Klassen-Gewichte → Favoriten bekommen >85%
  - 3-Wege-Split: Train<2023, Kalibrierung 2023, Val>=2024
  - Ensemble 3 Seeds, Score-Gitter gemittelt
  - col_medians im Checkpoint gespeichert (Inferenz-NaN-Imputation)
  - dc_rho auf Trainingsdaten gefittet

Ausgabe: E:/Profilov2/public/data/wm-predictor/model.json (float32)
         E:/Profilov2/public/data/wm-predictor/model_int8.json
         E:/Projects/Fussball_ai/data/models/v7_latest.pt
"""
import json
import sys
import time
from pathlib import Path
from math import exp, factorial, log

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import minimize_scalar
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, "E:/Projects/Fussball_ai")

from src.features_v6 import (
    load_features_v6, build_feature_table_v6, PROCESSED_DIR,
    score_grid, wdl_from_grid,
)
from src.train_v2 import FootballNet, _recency_weights, _normalize_features, _split
from src.export_browser import _serialize_one_model
from scripts.quantize_v4_int8 import main as run_quantize_int8

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PROFILOV2 = Path("E:/Profilov2/public/data/wm-predictor")

print("=" * 70)
print(" V7 Training: Poisson-Score-Modell (Dixon-Coles + Ensemble)")
print("=" * 70)
print(f"   Device: {DEVICE}")
if DEVICE.type == "cuda":
    print(f"   GPU: {torch.cuda.get_device_name(0)}")

# ─────────────────────────── Data ───────────────────────────
if not (PROCESSED_DIR / "features_v6.npz").exists():
    print("\n   Baue V6 Features...")
    build_feature_table_v6()

# Expanded training window: from 2015 (full FIFA squad coverage)
X_all, y_all, y_hg_all, y_ag_all, dates_all, _, _, feat_names = load_features_v6(
    "all", "2015-01-01", "2024-01-01"
)
print(f"   Total samples (2015+): {len(X_all):,}")

# 3-way split: Train < 2023, Cal 2023, Val >= 2024
mask_tr = dates_all < pd.Timestamp("2023-01-01")
mask_ca = (dates_all >= pd.Timestamp("2023-01-01")) & (dates_all < pd.Timestamp("2024-01-01"))
mask_va = dates_all >= pd.Timestamp("2024-01-01")

X_tr_raw, X_ca_raw, X_va_raw = X_all[mask_tr], X_all[mask_ca], X_all[mask_va]
y_cls_tr, y_cls_ca, y_cls_va = y_all[mask_tr], y_all[mask_ca], y_all[mask_va]
y_hg_tr, y_hg_ca, y_hg_va = y_hg_all[mask_tr], y_hg_all[mask_ca], y_hg_all[mask_va]
y_ag_tr, y_ag_ca, y_ag_va = y_ag_all[mask_tr], y_ag_all[mask_ca], y_ag_all[mask_va]
dates_tr = dates_all[mask_tr]
print(f"   Train: {mask_tr.sum():,}  Cal: {mask_ca.sum():,}  Val: {mask_va.sum():,}")

# NaN imputation using training column medians
nan_tr = np.isnan(X_tr_raw).any(axis=1).sum()
if nan_tr > 0:
    print(f"   WARN: {nan_tr} NaN-Zeilen in Train – imputing with column median")
col_medians = np.nanmedian(X_tr_raw, axis=0)

def _impute(X, medians):
    out = X.copy()
    idx = np.where(np.isnan(out))
    out[idx] = np.take(medians, idx[1])
    return out

X_tr_raw = _impute(X_tr_raw, col_medians)
X_ca_raw = _impute(X_ca_raw, col_medians)
X_va_raw = _impute(X_va_raw, col_medians)

# Normalize (fit on train only)
X_tr_n, _, norm_stats = _normalize_features(X_tr_raw, X_ca_raw)  # fit on train
norm_mean = np.array(norm_stats["mean"], dtype=np.float32)
norm_std = np.array(norm_stats["std"], dtype=np.float32)
X_ca_n = ((X_ca_raw - norm_mean) / norm_std).astype(np.float32)
X_va_n = ((X_va_raw - norm_mean) / norm_std).astype(np.float32)

in_dim = X_tr_n.shape[1]
print(f"   in_dim = {in_dim}")

# Recency weights on training data
sample_w = _recency_weights(pd.Series(dates_tr), half_life_days=90.0)
sample_w = (sample_w / sample_w.mean()).astype(np.float32)

# Goal targets (raw integer counts for Poisson NLL)
y_goals_tr = np.column_stack([y_hg_tr, y_ag_tr]).astype(np.float32)
y_goals_ca = np.column_stack([y_hg_ca, y_ag_ca]).astype(np.float32)
y_goals_va = np.column_stack([y_hg_va, y_ag_va]).astype(np.float32)

# ─────────────────────────── Fit dc_rho ───────────────────────────
print("\n   Fitting Dixon-Coles rho on training data...")

def _dc_nll(rho, lh_arr, la_arr, y_h_arr, y_a_arr):
    """Negative log-likelihood for DC correction on training data."""
    total = 0.0
    for lh, la, yh, ya in zip(lh_arr, la_arr, y_h_arr, y_a_arr):
        lh = max(float(lh), 0.05)
        la = max(float(la), 0.05)
        yh, ya = int(yh), int(ya)
        if yh == 0 and ya == 0:
            tau = 1.0 - lh * la * rho
        elif yh == 1 and ya == 0:
            tau = 1.0 + la * rho
        elif yh == 0 and ya == 1:
            tau = 1.0 + lh * rho
        elif yh == 1 and ya == 1:
            tau = 1.0 - rho
        else:
            tau = 1.0
        tau = max(tau, 1e-9)
        total += log(tau)
    return -total

# Use mean goals as proxy lambdas to fit rho empirically
lh_mean = float(y_hg_tr.mean())
la_mean = float(y_ag_tr.mean())
# Fit rho using a sample to keep it fast
rng = np.random.default_rng(42)
sample_idx = rng.choice(len(y_hg_tr), min(5000, len(y_hg_tr)), replace=False)
lh_arr = np.full(len(sample_idx), lh_mean)
la_arr = np.full(len(sample_idx), la_mean)
res_rho = minimize_scalar(
    lambda r: _dc_nll(r, lh_arr, la_arr, y_hg_tr[sample_idx], y_ag_tr[sample_idx]),
    bounds=(-0.5, 0.0), method="bounded"
)
dc_rho = float(res_rho.x)
print(f"   Fitted dc_rho = {dc_rho:.4f}  (V6 used -0.13)")

# ─────────────────────────── Architecture ───────────────────────────
HIDDEN = 256
N_BLOCKS = 8
DROPOUT = 0.30
LR = 1e-3
EPOCHS = 200
PATIENCE = 25
N_SEEDS = 3
LAMBDA_CLS = 0.3   # aux cls loss weight
LAMBDA_POIS = 1.0  # Poisson NLL weight

poisson_loss_fn = nn.PoissonNLLLoss(log_input=True, reduction="none", full=True)
# No label smoothing, no class weights
cls_loss_fn = nn.CrossEntropyLoss(reduction="none")

# ─────────────────────────── Evaluation helper ───────────────────────────
def evaluate_v7(model, X_n: np.ndarray, y_cls: np.ndarray, y_goals: np.ndarray, rho: float):
    """Evaluate using score grid W/D/L probabilities."""
    model.eval()
    all_preds = []
    all_true = []
    n = len(X_n)
    batch_size = 1024
    with torch.no_grad():
        for i in range(0, n, batch_size):
            xb = torch.from_numpy(X_n[i:i+batch_size]).to(DEVICE)
            _, log_lam = model(xb)
            log_lam = log_lam.cpu().numpy()
            for j in range(len(log_lam)):
                lh = float(np.exp(log_lam[j, 0]))
                la = float(np.exp(log_lam[j, 1]))
                g = score_grid(lh, la, rho, n=10)
                ph, pd_, pa = wdl_from_grid(g)
                all_preds.append([pd_, ph, pa])  # [draw, home, away] to match y_cls encoding
            all_true.extend(y_cls[i:i+batch_size].tolist())

    preds = np.array(all_preds, dtype=np.float64)
    y_true = np.array(all_true)

    pred_cls = preds.argmax(axis=1)
    acc = float((pred_cls == y_true).mean())

    # Per-class accuracy
    per_class = {}
    for c in range(3):
        m = y_true == c
        if m.sum() > 0:
            per_class[c] = float((pred_cls[m] == c).mean())

    # Log-loss and Brier on W/D/L
    eps = 1e-9
    log_loss = float(-np.log(preds[np.arange(n), y_true] + eps).mean())
    # Brier: sum over classes
    one_hot = np.zeros_like(preds)
    one_hot[np.arange(n), y_true] = 1.0
    brier = float(((preds - one_hot) ** 2).sum(axis=1).mean())

    return {
        "accuracy": acc,
        "per_class_acc": {int(k): round(v, 4) for k, v in per_class.items()},
        "log_loss": log_loss,
        "brier": brier,
    }

# ─────────────────────────── Training loop (3 seeds) ───────────────────────────
trained_models = []

for seed in range(N_SEEDS):
    print()
    print(f"{'='*70}")
    print(f" Seed {seed + 1}/{N_SEEDS}")
    print(f"{'='*70}")

    torch.manual_seed(seed)
    np.random.seed(seed)

    model = FootballNet(in_dim=in_dim, hidden=HIDDEN, n_blocks=N_BLOCKS, dropout=DROPOUT).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"   Model params: {n_params:,}")

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=LR,
        total_steps=EPOCHS * max(1, len(X_tr_n) // 128),
        pct_start=0.05, anneal_strategy="cos"
    )

    Xt = torch.from_numpy(X_tr_n).to(DEVICE)
    y_cls_t = torch.from_numpy(y_cls_tr).long().to(DEVICE)
    y_goals_t = torch.from_numpy(y_goals_tr).to(DEVICE)
    w_t = torch.from_numpy(sample_w).to(DEVICE)
    train_ds = TensorDataset(Xt, y_cls_t, y_goals_t, w_t)
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)

    best_brier = float("inf")
    best_state = None
    no_imp = 0
    t0 = time.time()

    for ep in range(1, EPOCHS + 1):
        model.train()
        ep_loss = ep_n = 0
        for batch in train_loader:
            xb, yb, goals_b, wb = batch
            opt.zero_grad(set_to_none=True)
            logits, log_lam = model(xb)

            # Poisson NLL on goal head (log_input=True: expects log(lambda))
            l_pois = (poisson_loss_fn(log_lam, goals_b).sum(1) * wb).mean()
            # Auxiliary cls loss (no smoothing, no class weights)
            l_cls = (cls_loss_fn(logits, yb) * wb).mean()
            loss = LAMBDA_POIS * l_pois + LAMBDA_CLS * l_cls

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            try:
                sched.step()
            except ValueError:
                pass
            ep_loss += loss.item() * xb.size(0)
            ep_n += xb.size(0)

        if ep % 10 == 0 or ep <= 3:
            vm = evaluate_v7(model, X_ca_n, y_cls_ca, y_goals_ca, dc_rho)
            elapsed = time.time() - t0
            print(
                f"   Ep {ep:3d}/{EPOCHS}  loss={ep_loss/ep_n:.4f}  "
                f"cal_acc={vm['accuracy']:.4f}  cal_brier={vm['brier']:.4f}  "
                f"({elapsed:.0f}s)"
            )
            if vm["brier"] < best_brier - 1e-5:
                best_brier = vm["brier"]
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                no_imp = 0
            else:
                no_imp += 1
                if no_imp >= PATIENCE:
                    print(f"   Early stopping @ Ep {ep}")
                    break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    cal_final = evaluate_v7(model, X_ca_n, y_cls_ca, y_goals_ca, dc_rho)
    val_final = evaluate_v7(model, X_va_n, y_cls_va, y_goals_va, dc_rho)
    print(f"\n   Seed {seed+1} Cal:  acc={cal_final['accuracy']:.4f}  brier={cal_final['brier']:.4f}  "
          f"logloss={cal_final['log_loss']:.4f}")
    print(f"   Seed {seed+1} Val:  acc={val_final['accuracy']:.4f}  brier={val_final['brier']:.4f}  "
          f"logloss={val_final['log_loss']:.4f}")
    print(f"   Per-class: {val_final['per_class_acc']}")

    trained_models.append(model)

# ─────────────────────────── Ensemble evaluation ───────────────────────────
print()
print("=" * 70)
print(" Ensemble Evaluation (Val >= 2024)")
print("=" * 70)

def evaluate_ensemble(models, X_n, y_cls, y_goals, rho):
    n = len(X_n)
    batch_size = 1024
    all_grids = []
    for model in models:
        model.eval()
        grids_m = []
        with torch.no_grad():
            for i in range(0, n, batch_size):
                xb = torch.from_numpy(X_n[i:i+batch_size]).to(DEVICE)
                _, log_lam = model(xb)
                log_lam = log_lam.cpu().numpy()
                for j in range(len(log_lam)):
                    lh = float(np.exp(log_lam[j, 0]))
                    la = float(np.exp(log_lam[j, 1]))
                    grids_m.append(score_grid(lh, la, rho, n=10))
        all_grids.append(grids_m)

    # Average grids across ensemble
    avg_grids = [
        sum(all_grids[m][i] for m in range(len(models))) / len(models)
        for i in range(n)
    ]

    preds = []
    for g in avg_grids:
        ph, pd_, pa = wdl_from_grid(g)
        preds.append([pd_, ph, pa])
    preds = np.array(preds, dtype=np.float64)

    pred_cls = preds.argmax(axis=1)
    acc = float((pred_cls == y_cls).mean())
    per_class = {}
    for c in range(3):
        m = y_cls == c
        if m.sum() > 0:
            per_class[c] = float((pred_cls[m] == c).mean())
    eps = 1e-9
    log_loss = float(-np.log(preds[np.arange(n), y_cls] + eps).mean())
    one_hot = np.zeros_like(preds)
    one_hot[np.arange(n), y_cls] = 1.0
    brier = float(((preds - one_hot) ** 2).sum(axis=1).mean())
    return {"accuracy": acc, "per_class_acc": per_class, "log_loss": log_loss, "brier": brier}

ens_val = evaluate_ensemble(trained_models, X_va_n, y_cls_va, y_goals_va, dc_rho)
print(f"   Ensemble Val: acc={ens_val['accuracy']:.4f}  brier={ens_val['brier']:.4f}  "
      f"logloss={ens_val['log_loss']:.4f}")
print(f"   Per-class:  {ens_val['per_class_acc']}")

# Compare vs V6 baseline
print()
print("=" * 70)
print(" V7 vs V6 Vergleich")
print("=" * 70)
print(f"   V7 Ensemble:  acc={ens_val['accuracy']:.4f}  brier={ens_val['brier']:.4f}")
print(f"   V6 (reference): acc=0.5899  brier=0.5358")

# ─────────────────────────── Export ───────────────────────────
print()
print("   Exportiere Modell...")

state_dicts = [_serialize_one_model(m) for m in trained_models]

bundle = {
    "architecture": {
        "n_models": N_SEEDS,
        "in_dim": in_dim,
        "hidden": HIDDEN,
        "n_blocks": N_BLOCKS,
        "n_classes": 3,
        "version": "v7-poisson",
        "n_params": n_params,
        "n_blocks_configs": [N_BLOCKS] * N_SEEDS,
    },
    "models": state_dicts,
    "norm_stats": norm_stats,
    "col_medians": col_medians.tolist(),
    "dc_rho": dc_rho,
    "goal_head_mode": "poisson_log",
    "temperature": 1.0,
    "model_version": "v7-poisson",
    "feature_names": list(feat_names),
    "n_features": in_dim,
    "train_start": "2015-01-01",
    "half_life_days": 90.0,
    "val_metrics": ens_val,
}

PROFILOV2.mkdir(parents=True, exist_ok=True)
out_path = PROFILOV2 / "model.json"
with open(out_path, "w", encoding="utf-8") as fh:
    json.dump(bundle, fh, separators=(",", ":"))
print(f"   Gespeichert (f32): {out_path} ({out_path.stat().st_size/1e6:.2f} MB)")

# Int8 quantize
try:
    run_quantize_int8()
    int8_path = PROFILOV2 / "model_int8.json"
    print(f"   Gespeichert (int8): {int8_path} ({int8_path.stat().st_size/1e6:.2f} MB)")
except Exception as e:
    print(f"   Int8-Quantisierung fehlgeschlagen: {e}")

# PyTorch checkpoint (for Python inference)
pt_dir = Path("E:/Projects/Fussball_ai/data/models")
pt_dir.mkdir(parents=True, exist_ok=True)
pt_path = pt_dir / "v7_latest.pt"
torch.save({
    "state_dicts": [m.state_dict() for m in trained_models],
    "in_dim": in_dim,
    "hidden": HIDDEN,
    "n_blocks": N_BLOCKS,
    "norm_stats": norm_stats,
    "col_medians": col_medians.tolist(),
    "dc_rho": dc_rho,
    "goal_head_mode": "poisson_log",
    "model_version": "v7-poisson",
    "val_metrics": ens_val,
    "feature_names": list(feat_names),
}, pt_path)
print(f"   Gespeichert (PT):  {pt_path}")

print()
print("   FERTIG. V7-Modell bereit.")
