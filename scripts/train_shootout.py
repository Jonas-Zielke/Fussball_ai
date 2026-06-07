"""
Training Shootout-V1 — gelerntes Elfmeterschießen-Modell.

Kleines, stark regularisiertes Netz (1 Hidden-Layer) auf 677 historischen
Shootouts. Backtest gegen zwei Baselines beweist, dass die Penalty-Features
echten Mehrwert liefern (sonst wäre es Rauschen):
  (a) Konstant 0.5
  (b) Reine Elo/Spielstärke (parameterfreie Elo-Erwartung)

Aufruf:  python -m scripts.train_shootout
Output:  data/models/shootout_v1.pt
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn

from src.train_v2 import DEVICE, _normalize_features
from src.shootout_features import (
    build_shootout_dataset, ShootoutNet, FEATURE_ORDER, ELO_SCALE, MODEL_PATH,
)

VAL_START = "2018-01-01"
HIDDEN = 8
EPOCHS = 400
LR = 3e-3
WEIGHT_DECAY = 1e-2
PATIENCE = 40
SEED = 42


def _augment_symmetry(X, y):
    """Jedes Sample gespiegelt (A/B getauscht, str_diff negiert, Label invertiert)."""
    Xs = X.copy()
    Xs[:, [0, 1]] = X[:, [1, 0]]      # pen_skill_a <-> b
    Xs[:, [2, 3]] = X[:, [3, 2]]      # pen_exp_a   <-> b
    Xs[:, 4] = -X[:, 4]               # str_diff -> -str_diff
    ys = 1 - y
    return np.vstack([X, Xs]), np.concatenate([y, ys])


def _metrics(p, y):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    logloss = float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
    brier = float(np.mean((p - y) ** 2))
    acc = float(np.mean((p >= 0.5).astype(int) == y))
    return {"acc": acc, "logloss": logloss, "brier": brier}


def main() -> int:
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    print("=" * 68)
    print(" Shootout-V1 - Training & Backtest")
    print("=" * 68)

    X, y, dates, teams = build_shootout_dataset()
    dates = pd.to_datetime(dates)
    print(f"   {len(X)} Shootouts | home gewinnt: {y.mean():.3f}")

    val_mask = dates >= pd.Timestamp(VAL_START)
    X_tr_raw, y_tr = X[~val_mask], y[~val_mask]
    X_va_raw, y_va = X[val_mask], y[val_mask]
    print(f"   Train: {len(X_tr_raw)}  |  Val (ab {VAL_START}): {len(X_va_raw)}")

    # Symmetrie-Augmentation nur auf Train
    X_tr_aug, y_tr_aug = _augment_symmetry(X_tr_raw, y_tr)

    # Normalisierung (fit auf augmentiertem Train)
    X_tr_n, X_va_n, norm_stats = _normalize_features(X_tr_aug, X_va_raw)
    in_dim = X_tr_n.shape[1]

    Xtr = torch.tensor(X_tr_n, dtype=torch.float32, device=DEVICE)
    ytr = torch.tensor(y_tr_aug, dtype=torch.float32, device=DEVICE)
    Xva = torch.tensor(X_va_n, dtype=torch.float32, device=DEVICE)

    model = ShootoutNet(in_dim, HIDDEN).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.BCEWithLogitsLoss()

    best_logloss, best_state, best_ep, since = 1e9, None, 0, 0
    for ep in range(1, EPOCHS + 1):
        model.train()
        opt.zero_grad(set_to_none=True)
        logits = model(Xtr)
        loss = loss_fn(logits, ytr)
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            # symmetrisierte Val-Vorhersage
            pv_ab = torch.sigmoid(model(Xva)).cpu().numpy()
        X_va_swap = X_va_raw.copy()
        X_va_swap[:, [0, 1]] = X_va_raw[:, [1, 0]]
        X_va_swap[:, [2, 3]] = X_va_raw[:, [3, 2]]
        X_va_swap[:, 4] = -X_va_raw[:, 4]
        X_va_swap_n = (X_va_swap - np.array(norm_stats["mean"], dtype=np.float32)) / np.array(norm_stats["std"], dtype=np.float32)
        with torch.no_grad():
            pv_ba = torch.sigmoid(model(torch.tensor(X_va_swap_n, dtype=torch.float32, device=DEVICE))).cpu().numpy()
        pv = 0.5 * (pv_ab + (1 - pv_ba))

        m = _metrics(pv, y_va)
        if m["logloss"] < best_logloss - 1e-5:
            best_logloss, best_state, best_ep, since = m["logloss"], {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, ep, 0
        else:
            since += 1
            if since >= PATIENCE:
                break

    model.load_state_dict(best_state)
    model.eval()

    # Finale Val-Vorhersage (symmetrisiert)
    mean = np.array(norm_stats["mean"], dtype=np.float32)
    std = np.array(norm_stats["std"], dtype=np.float32)
    X_va_swap_n = (_swap(X_va_raw) - mean) / std
    with torch.no_grad():
        pv_ab = torch.sigmoid(model(Xva)).cpu().numpy()
        pv_ba = torch.sigmoid(model(torch.tensor(X_va_swap_n, dtype=torch.float32, device=DEVICE))).cpu().numpy()
    pv = 0.5 * (pv_ab + (1 - pv_ba))
    m_model = _metrics(pv, y_va)

    # --- Baselines ---
    p_half = np.full(len(y_va), 0.5)
    m_half = _metrics(p_half, y_va)
    # Elo-Baseline: parameterfreie Elo-Erwartung aus str_diff (=elo_diff/ELO_SCALE)
    elo_diff = X_va_raw[:, 4] * ELO_SCALE
    p_elo = 1.0 / (1.0 + 10 ** (-elo_diff / 400.0))
    m_elo = _metrics(p_elo, y_va)

    print(f"\n   Bestes Modell @ Epoche {best_ep}")
    print("   " + "-" * 56)
    print(f"   {'Modell':<22}{'Acc':>8}{'LogLoss':>12}{'Brier':>10}")
    print("   " + "-" * 56)
    print(f"   {'0.5-Baseline':<22}{m_half['acc']:>8.3f}{m_half['logloss']:>12.4f}{m_half['brier']:>10.4f}")
    print(f"   {'Elo-Baseline':<22}{m_elo['acc']:>8.3f}{m_elo['logloss']:>12.4f}{m_elo['brier']:>10.4f}")
    print(f"   {'Shootout-V1':<22}{m_model['acc']:>8.3f}{m_model['logloss']:>12.4f}{m_model['brier']:>10.4f}")
    print("   " + "-" * 56)
    beats_ll = m_model["logloss"] < m_half["logloss"] and m_model["logloss"] < m_elo["logloss"]
    beats_acc = m_model["acc"] >= m_half["acc"] and m_model["acc"] >= m_elo["acc"]
    beats_brier = m_model["brier"] <= m_half["brier"] and m_model["brier"] <= m_elo["brier"]
    print(f"   Schlaegt beide Baselines -> LogLoss: {'JA' if beats_ll else 'nein'} | "
          f"Acc: {'JA' if beats_acc else 'nein'} | Brier: {'JA' if beats_brier else 'nein'}")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "norm_stats": norm_stats,
        "feature_order": FEATURE_ORDER,
        "in_dim": in_dim,
        "arch": {"hidden": HIDDEN},
        "val_metrics": {"model": m_model, "baseline_half": m_half, "baseline_elo": m_elo},
        "as_of": str(pd.to_datetime(dates).max().date()),
        "model_version": "shootout-v1",
    }, MODEL_PATH)
    print(f"\n   geschrieben: {MODEL_PATH}")
    print("=" * 68)
    return 0


def _swap(X):
    Xs = X.copy()
    Xs[:, [0, 1]] = X[:, [1, 0]]
    Xs[:, [2, 3]] = X[:, [3, 2]]
    Xs[:, 4] = -X[:, 4]
    return Xs


if __name__ == "__main__":
    raise SystemExit(main())
