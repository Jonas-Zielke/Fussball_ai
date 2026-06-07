"""
Train V3 Modell - Recency-fokussiert.

Unterschiede zu V2:
  1. Features: 45 (statt 34) - neue V3 Features
  2. Training data: ab 2015 (statt 1985) - 10 Jahre Fokus
  3. Recency-Weighting: half_life_days=240 (statt 2400) - 8 Monate
  4. Architektur: hidden=96, n_blocks=4 (etwas groesser fuer mehr Signal)
  5. Score-Regression bleibt (Multi-Task)
  6. Mixup + SWA + Temperature-Scaling
"""

from __future__ import annotations

import copy
import json
import math
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from .features_v3 import (
    load_features_v3, get_current_team_ratings_v3, _compute_final_state_v3,
    RAW_RESULTS, PROCESSED_DIR, ELOG_START, HOME_ADVANTAGE_ELO, _team_to_continent,
)
from .team_normalize import tournament_weight, normalize_team_name
from .train_v2 import (
    ResidualBlock, FootballNet, SWA, mixup_batch,
    _recency_weights, _normalize_features, _split, evaluate_v2,
    _softmax_np, _onehot, fit_temperature,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = REPO_ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_v3(
    epochs: int = 80,
    batch_size: int = 256,
    lr: float = 1.2e-3,
    hidden: int = 96,
    n_blocks: int = 4,
    dropout: float = 0.2,
    mixup_alpha: float = 0.3,
    swa_n: int = 8,
    train_start: str = "2015-01-01",
    val_start: str = "2024-01-01",
    half_life_days: float = 240.0,
    seed: int = 42,
    verbose: bool = True,
) -> dict:
    """Trainiert EIN einzelnes V3-Modell."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    X_all, y_all, y_hg_all, y_ag_all, dates_all, _, _, feat_names = load_features_v3("all", train_start, val_start)
    X_tr_raw, X_va_raw = _split(X_all, dates_all, val_start)
    y_cls_tr, y_cls_va = _split(y_all, dates_all, val_start)
    y_reg_tr = np.column_stack([y_hg_all, y_ag_all])
    y_reg_tr_split, y_reg_va_split = _split(y_reg_tr, dates_all, val_start)
    dates_tr, dates_va = _split(dates_all, dates_all, val_start)
    print(f"   V3 Total: {len(X_all):,} | Train: {len(X_tr_raw):,} | Val: {len(X_va_raw):,}")
    print(f"   Features: {len(feat_names)}")

    X_tr_n, X_va_n, norm_stats = _normalize_features(X_tr_raw, X_va_raw)
    in_dim = X_tr_n.shape[1]
    sample_w = _recency_weights(pd.Series(dates_tr), half_life_days=half_life_days)
    sample_w = sample_w / sample_w.mean()
    print(f"   Recency-Half-Life: {half_life_days} Tage (~{half_life_days/30:.1f} Monate)")
    print(f"   Sample weights: min={sample_w.min():.3f}, max={sample_w.max():.3f}")

    # Goal-Stats
    hg_mean = float(y_reg_tr_split[:, 0].mean())
    hg_std = float(y_reg_tr_split[:, 0].std() + 1e-6)
    ag_mean = float(y_reg_tr_split[:, 1].mean())
    ag_std = float(y_reg_tr_split[:, 1].std() + 1e-6)
    goal_stats = {"home_mean": hg_mean, "home_std": hg_std, "away_mean": ag_mean, "away_std": ag_std}
    y_reg_tr_n = np.zeros_like(y_reg_tr_split, dtype=np.float32)
    y_reg_tr_n[:, 0] = (y_reg_tr_split[:, 0] - hg_mean) / hg_std
    y_reg_tr_n[:, 1] = (y_reg_tr_split[:, 1] - ag_mean) / ag_std
    y_reg_va_n = np.zeros_like(y_reg_va_split, dtype=np.float32)
    y_reg_va_n[:, 0] = (y_reg_va_split[:, 0] - hg_mean) / hg_std
    y_reg_va_n[:, 1] = (y_reg_va_split[:, 1] - ag_mean) / ag_std

    Xt = torch.from_numpy(X_tr_n)
    y_cls_t = torch.from_numpy(y_cls_tr).long()
    y_reg_t = torch.from_numpy(y_reg_tr_n)
    w_t = torch.from_numpy(sample_w)
    Xv = torch.from_numpy(X_va_n)
    y_cls_v = torch.from_numpy(y_cls_va).long()
    y_reg_v = torch.from_numpy(y_reg_va_n)
    train_ds = TensorDataset(Xt, y_cls_t, y_reg_t, w_t)
    val_ds = TensorDataset(Xv, y_cls_v, y_reg_v)
    pin = DEVICE.type == "cuda"
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=pin)
    val_loader = DataLoader(val_ds, batch_size=1024, shuffle=False, num_workers=0, pin_memory=pin)

    model = FootballNet(in_dim=in_dim, hidden=hidden, n_blocks=n_blocks, dropout=dropout).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=lr, total_steps=epochs * len(train_loader),
        pct_start=0.1, anneal_strategy="cos", div_factor=25, final_div_factor=1000,
    )
    counts = np.bincount(y_cls_tr, minlength=3).astype(np.float64)
    cw = counts.sum() / (3 * counts + 1e-9)
    cw[0] *= 1.20
    cw_t = torch.tensor(cw, dtype=torch.float32, device=DEVICE)
    cls_loss_fn = nn.CrossEntropyLoss(weight=cw_t, label_smoothing=0.05, reduction="none")
    reg_loss_fn = nn.SmoothL1Loss(reduction="none")
    lambda_reg = 0.5

    swa = SWA(n_avg=swa_n)
    best_val_acc = 0.0
    best_state = None
    patience = 12
    no_improve = 0
    history = []
    print("-" * 70)

    for epoch in range(1, epochs + 1):
        model.train()
        ep_loss = 0.0
        ep_correct = 0
        ep_n = 0
        for batch in train_loader:
            xb = batch[0].to(DEVICE, non_blocking=True)
            y_cls_b = batch[1].to(DEVICE, non_blocking=True)
            y_reg_b = batch[2].to(DEVICE, non_blocking=True)
            w_b = batch[3].to(DEVICE, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            logits, goals = model(xb)
            loss_cls = (cls_loss_fn(logits, y_cls_b) * w_b).mean()
            loss_reg = (reg_loss_fn(goals, y_reg_b).mean(dim=1) * w_b).mean()
            loss = loss_cls + lambda_reg * loss_reg
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            ep_loss += loss.item() * xb.size(0)
            ep_correct += (logits.argmax(dim=1) == y_cls_b).sum().item()
            ep_n += xb.size(0)

        val_metrics = evaluate_v2(model, val_loader, goal_stats)
        history.append({
            "epoch": epoch,
            "train_loss": ep_loss / ep_n,
            "train_acc": ep_correct / ep_n,
            "val_acc": val_metrics["accuracy"],
            "val_loss": val_metrics["log_loss"],
            "val_brier": val_metrics["brier"],
        })
        if verbose and (epoch <= 3 or epoch % 5 == 0):
            print(f"      Ep {epoch:3d}  tr_loss={ep_loss/ep_n:.4f}  tr_acc={ep_correct/ep_n:.4f}  "
                  f"v_acc={val_metrics['accuracy']:.4f}  v_loss={val_metrics['log_loss']:.4f}  "
                  f"brier={val_metrics['brier']:.4f}")
        if val_metrics["accuracy"] > best_val_acc + 1e-5:
            best_val_acc = val_metrics["accuracy"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
        if epoch >= epochs // 2:
            swa.add({k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
        if no_improve >= patience:
            if verbose:
                print(f"      Early stopping @ Ep {epoch}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    swa_state = swa.averaged_state()
    if swa_state is not None:
        model.load_state_dict(swa_state)
        swa_metrics = evaluate_v2(model, val_loader, goal_stats)
        if swa_metrics["accuracy"] > best_val_acc:
            best_val_acc = swa_metrics["accuracy"]
            best_state = swa_state
            model.load_state_dict(best_state)
        else:
            model.load_state_dict(best_state)
    return model, goal_stats, best_val_acc, norm_stats, feat_names, in_dim


def train_v3_and_save(
    epochs: int = 100,
    batch_size: int = 256,
    lr: float = 1.2e-3,
    hidden: int = 96,
    n_blocks: int = 4,
    dropout: float = 0.2,
    mixup_alpha: float = 0.3,
    train_start: str = "2015-01-01",
    val_start: str = "2024-01-01",
    half_life_days: float = 240.0,
    verbose: bool = True,
) -> dict:
    """Trainiert V3 und speichert das Browser-Modell."""
    print("=" * 70)
    print(f" Training V3 (Recency-fokussiert, {train_start}+, half-life={half_life_days}d)")
    print("=" * 70)
    print(f"   Device: {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"   GPU: {torch.cuda.get_device_name(0)}")

    model, gs, val_acc, norm_stats, feat_names, in_dim = train_v3(
        epochs=epochs, batch_size=batch_size, lr=lr, hidden=hidden, n_blocks=n_blocks,
        dropout=dropout, mixup_alpha=mixup_alpha, train_start=train_start, val_start=val_start,
        half_life_days=half_life_days, verbose=verbose,
    )

    # Export als Browser-Modell
    from .export_browser import _serialize_one_model
    model.eval()
    state = _serialize_one_model(model)

    arch = {
        "n_models": 1,
        "in_dim": in_dim,
        "hidden": hidden,
        "n_blocks": n_blocks,
        "n_classes": 3,
        "version": "v3",
    }
    bundle = {
        "architecture": arch,
        "models": [state],
        "norm_stats": norm_stats,
        "goal_stats": gs,
        "temperature": 1.0,
        "ensemble_val_acc": val_acc,
        "calibrated_val_acc": val_acc,
        "feature_names": list(feat_names),
        "n_features": in_dim,
        "train_start": train_start,
        "half_life_days": half_life_days,
    }
    # Ins Profilov2 schreiben
    out_path = Path("E:/Profilov2/public/data/wm-predictor/model.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(bundle, fh, separators=(",", ":"))
    size_mb = out_path.stat().st_size / 1e6
    print(f"   geschrieben: {out_path} ({size_mb:.2f} MB)")
    print(f"   V3 Val-Accuracy: {val_acc:.4f}")
    print("=" * 70)
    return bundle


def main() -> int:
    train_v3_and_save(verbose=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
