"""
Trainings-Pipeline V2 - deutlich verbessert.

Verbesserungen gegenueber V1:
  1. Multi-Task: Classification (3 Klassen) + Score Regression (Heimtore, Auswärtstore)
  2. Deeper ResNet-style MLP mit Skip-Connections
  3. Mixup Data Augmentation
  4. Stochastic Weight Averaging (SWA)
  5. Recency-Weighting (neuere Spiele zaehlen mehr)
  6. Ensemble aus 5 Modellen mit verschiedenen Seeds
  7. Temperature Scaling auf Validation fuer kalibrierte Wahrscheinlichkeiten
  8. Laengeres Training mit besserem LR-Schedule
  9. Mehr Daten (1990+) statt 2000+

Ziel: Validation Accuracy > 60% (von 54.6% in V1)
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

from .features_v2 import (
    load_features_v2, get_current_team_ratings, _compute_final_state,
    RAW_RESULTS, PROCESSED_DIR, ELOG_START, HOME_ADVANTAGE_ELO, _team_to_continent,
)
from .team_normalize import tournament_weight, normalize_team_name

REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = REPO_ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -------------------- Model --------------------

class ResidualBlock(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.2):
        super().__init__()
        self.lin1 = nn.Linear(dim, dim)
        self.bn1 = nn.BatchNorm1d(dim)
        self.lin2 = nn.Linear(dim, dim)
        self.bn2 = nn.BatchNorm1d(dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        h = F.relu(self.bn1(self.lin1(x)), inplace=True)
        h = self.drop(h)
        h = self.bn2(self.lin2(h))
        return F.relu(x + h, inplace=True)


class FootballNet(nn.Module):
    """Multi-Task Netz: Classification + Score Regression.

    Backbone: ResNet-style (ResidualBlocks) auf numerischen Features
    Heads:
      - cls_head: 3 Klassen (Draw/HomeWin/AwayWin)
      - reg_head: 2 Werte (Heimtore, Auswaertstore) - vorher Poisson-softplus
    """

    def __init__(self, in_dim: int = 32, hidden: int = 128, n_blocks: int = 4,
                 dropout: float = 0.2, num_classes: int = 3):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.blocks = nn.Sequential(*[ResidualBlock(hidden, dropout) for _ in range(n_blocks)])
        # Heads
        self.cls_head = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, num_classes),
        )
        self.reg_head = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 2),
        )
        # Initialisiere reg_head_bias auf durchschnittliche Tore
        # 1.76 Heim, 1.18 Auswaerts (raw scale) - wir normalisieren spaeter
        self.in_dim = in_dim
        self.hidden = hidden
        self.n_blocks = n_blocks
        self.dropout = dropout
        self.num_classes = num_classes

    def forward(self, x):
        h = self.input_proj(x)
        h = self.blocks(h)
        logits = self.cls_head(h)
        goals = self.reg_head(h)  # raw, wird per Softplus >= 0 gemacht
        return logits, goals


# -------------------- SWA --------------------

class SWA:
    """Stochastic Weight Averaging - mittelt die letzten n_checkpoints weights."""

    def __init__(self, n_avg: int = 5):
        self.n_avg = n_avg
        self.weights_queue: list[dict] = []

    def add(self, state_dict: dict):
        self.weights_queue.append({k: v.detach().cpu().clone() for k, v in state_dict.items()})
        if len(self.weights_queue) > self.n_avg:
            self.weights_queue.pop(0)

    def averaged_state(self) -> dict | None:
        if not self.weights_queue:
            return None
        n = len(self.weights_queue)
        avg = {}
        for k in self.weights_queue[0].keys():
            stacked = torch.stack([w[k].float() for w in self.weights_queue])
            avg[k] = stacked.mean(dim=0).to(self.weights_queue[0][k].dtype)
        return avg


# -------------------- Mixup --------------------

def mixup_batch(x: torch.Tensor, y_cls: torch.Tensor, y_reg: torch.Tensor, alpha: float = 0.4):
    """Mixup data augmentation.

    Mischt Samples mit einer Lambda aus Beta(alpha, alpha).
    """
    if alpha <= 0:
        return x, y_cls, y_reg, None, 1.0
    lam = float(np.random.beta(alpha, alpha))
    perm = torch.randperm(x.size(0), device=x.device)
    x_mix = lam * x + (1 - lam) * x[perm]
    y_cls_mix = (y_cls, y_cls[perm], lam)
    y_reg_mix = lam * y_reg + (1 - lam) * y_reg[perm]
    return x_mix, y_cls_mix, y_reg_mix, perm, lam


# -------------------- Training --------------------

def _recency_weights(dates: pd.Series, half_life_days: float = 1800.0) -> np.ndarray:
    """Gibt jedem Sample ein Gewicht, das exponentiell mit der Zeit abnimmt.

    half_life_days: Anzahl Tage, nach denen das Gewicht auf 50% faellt.
    """
    latest = dates.max()
    days_old = (latest - dates).dt.days.astype(np.float64).to_numpy()
    w = np.exp(-days_old * math.log(2) / half_life_days)
    return w.astype(np.float32)


def _normalize_features(X_tr, X_va):
    mean = X_tr.mean(axis=0)
    std = X_tr.std(axis=0) + 1e-6
    X_tr_n = ((X_tr - mean) / std).astype(np.float32)
    X_va_n = ((X_va - mean) / std).astype(np.float32)
    return X_tr_n, X_va_n, {"mean": mean.tolist(), "std": std.tolist()}


def _split(X, dates, val_start: str):
    mask = dates < pd.Timestamp(val_start)
    return X[mask], X[~mask]


def evaluate_v2(model: nn.Module, loader: DataLoader, goal_stats: dict) -> dict:
    """Evaluiert Multi-Task Model: Accuracy + Goal-MAE + Kalibrierung."""
    model.eval()
    all_logits = []
    all_goals = []
    all_y = []
    n = 0
    with torch.no_grad():
        for batch in loader:
            xb = batch[0].to(DEVICE, non_blocking=True)
            logits, goals = model(xb)
            all_logits.append(logits.cpu())
            all_goals.append(goals.cpu())
            # y_cls ist batch[1] (TensorDataset(X, y_cls, y_reg))
            all_y.append(batch[1])
            n += xb.size(0)
    logits = torch.cat(all_logits).numpy()
    goals_raw = torch.cat(all_goals).numpy()
    y_true = torch.cat(all_y).numpy()

    # Denormalisiere Goals
    hg_mean = goal_stats["home_mean"]
    hg_std = goal_stats["home_std"]
    ag_mean = goal_stats["away_mean"]
    ag_std = goal_stats["away_std"]
    pred_hg = goals_raw[:, 0] * hg_std + hg_mean
    pred_ag = goals_raw[:, 1] * ag_std + ag_mean
    # Softplus -> >= 0
    pred_hg = np.log1p(np.exp(pred_hg))  # softplus
    pred_ag = np.log1p(np.exp(pred_ag))

    # Classification Accuracy
    pred = logits.argmax(axis=1)
    correct = (pred == y_true).sum()
    acc = correct / n

    # Per-class accuracy
    per_class = {}
    for c in range(3):
        m = y_true == c
        if m.sum() > 0:
            per_class[int(c)] = float((pred[m] == c).mean())

    # Goal-MAE (auf Validation)
    y_hg = loader.dataset.tensors[2][:, 0].numpy()  # y_reg column 0
    y_ag = loader.dataset.tensors[2][:, 1].numpy()  # y_reg column 1
    hg_mae = float(np.abs(pred_hg - y_hg).mean())
    ag_mae = float(np.abs(pred_ag - y_ag).mean())

    # Brier Score
    probs = _softmax_np(logits)
    brier = float(((probs - _onehot(y_true, 3)) ** 2).sum(axis=1).mean())

    # Log-loss
    eps = 1e-9
    log_loss = float(-np.log(probs[np.arange(len(y_true)), y_true] + eps).mean())

    # Top-1 accuracy als Funktion von Confidence
    confidence = probs.max(axis=1)
    high_conf_mask = confidence > 0.5
    high_conf_acc = float((pred[high_conf_mask] == y_true[high_conf_mask]).mean()) if high_conf_mask.sum() > 0 else 0.0

    return {
        "accuracy": float(acc),
        "log_loss": log_loss,
        "brier": brier,
        "per_class_acc": per_class,
        "home_goal_mae": hg_mae,
        "away_goal_mae": ag_mae,
        "high_conf_acc": high_conf_acc,
        "n_high_conf": int(high_conf_mask.sum()),
        "n": int(n),
    }


def _softmax_np(x):
    z = x - x.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def _onehot(y, k):
    out = np.zeros((len(y), k), dtype=np.float32)
    out[np.arange(len(y)), y] = 1.0
    return out


def train_single_model(
    X_tr, y_cls_tr, y_reg_tr, sample_weights_tr, dates_tr,
    X_va, y_cls_va, y_reg_va, dates_va,
    in_dim: int = 32,
    hidden: int = 128,
    n_blocks: int = 4,
    dropout: float = 0.2,
    epochs: int = 80,
    batch_size: int = 256,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    mixup_alpha: float = 0.3,
    swa_n: int = 5,
    seed: int = 42,
    verbose: bool = False,
) -> tuple[nn.Module, dict, float]:
    """Trainiert EIN einzelnes Modell und gibt (model, swa_state, best_val_acc) zurueck."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Goal-Stats
    hg_mean = float(y_reg_tr[:, 0].mean())
    hg_std = float(y_reg_tr[:, 0].std() + 1e-6)
    ag_mean = float(y_reg_tr[:, 1].mean())
    ag_std = float(y_reg_tr[:, 1].std() + 1e-6)
    goal_stats = {"home_mean": hg_mean, "home_std": hg_std, "away_mean": ag_mean, "away_std": ag_std}
    # Normalisiere Goals
    y_reg_tr_n = np.zeros_like(y_reg_tr, dtype=np.float32)
    y_reg_tr_n[:, 0] = (y_reg_tr[:, 0] - hg_mean) / hg_std
    y_reg_tr_n[:, 1] = (y_reg_tr[:, 1] - ag_mean) / ag_std
    y_reg_va_n = np.zeros_like(y_reg_va, dtype=np.float32)
    y_reg_va_n[:, 0] = (y_reg_va[:, 0] - hg_mean) / hg_std
    y_reg_va_n[:, 1] = (y_reg_va[:, 1] - ag_mean) / ag_std

    # Tensoren
    Xt = torch.from_numpy(X_tr)
    y_cls_t = torch.from_numpy(y_cls_tr).long()
    y_reg_t = torch.from_numpy(y_reg_tr_n)
    w_t = torch.from_numpy(sample_weights_tr)

    Xv = torch.from_numpy(X_va)
    y_cls_v = torch.from_numpy(y_cls_va).long()
    y_reg_v = torch.from_numpy(y_reg_va_n)

    train_ds = TensorDataset(Xt, y_cls_t, y_reg_t, w_t)
    val_ds = TensorDataset(Xv, y_cls_v, y_reg_v)
    pin = DEVICE.type == "cuda"
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=pin)
    val_loader = DataLoader(val_ds, batch_size=1024, shuffle=False, num_workers=0, pin_memory=pin)

    model = FootballNet(in_dim=in_dim, hidden=hidden, n_blocks=n_blocks, dropout=dropout).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=lr, total_steps=epochs * len(train_loader),
        pct_start=0.1, anneal_strategy="cos", div_factor=25, final_div_factor=1000,
    )

    # Class weights
    counts = np.bincount(y_cls_tr, minlength=3).astype(np.float64)
    cw = counts.sum() / (3 * counts + 1e-9)
    cw[0] *= 1.20  # draws etwas staerker gewichten
    cw_t = torch.tensor(cw, dtype=torch.float32, device=DEVICE)

    cls_loss_fn = nn.CrossEntropyLoss(weight=cw_t, label_smoothing=0.05, reduction="none")
    reg_loss_fn = nn.SmoothL1Loss(reduction="none")
    lambda_reg = 0.5  # balance classification vs regression

    swa = SWA(n_avg=swa_n)
    best_val_acc = 0.0
    best_state = None
    patience = 12
    no_improve = 0
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        ep_loss = 0.0
        ep_correct = 0
        ep_n = 0
        for xb, y_cls_b, y_reg_b, w_b in train_loader:
            xb = xb.to(DEVICE, non_blocking=True)
            y_cls_b = y_cls_b.to(DEVICE, non_blocking=True)
            y_reg_b = y_reg_b.to(DEVICE, non_blocking=True)
            w_b = w_b.to(DEVICE, non_blocking=True)

            # Mixup
            if mixup_alpha > 0 and np.random.random() < 0.5:
                # Mixup anwenden
                lam = float(np.random.beta(mixup_alpha, mixup_alpha))
                perm = torch.randperm(xb.size(0), device=xb.device)
                xb = lam * xb + (1 - lam) * xb[perm]
                y_cls_b_perm = y_cls_b[perm]
                y_reg_b_perm = y_reg_b[perm]
                w_b_perm = w_b[perm]
            else:
                lam = 1.0
                perm = None
                y_cls_b_perm = None
                y_reg_b_perm = None
                w_b_perm = None

            opt.zero_grad(set_to_none=True)
            logits, goals = model(xb)
            # Classification loss (mit Mixup)
            if perm is not None:
                loss_cls_a = cls_loss_fn(logits, y_cls_b)
                loss_cls_b = cls_loss_fn(logits, y_cls_b_perm)
                loss_cls = (lam * loss_cls_a * w_b + (1 - lam) * loss_cls_b * w_b_perm).mean()
            else:
                loss_cls = (cls_loss_fn(logits, y_cls_b) * w_b).mean()
            # Regression loss
            if perm is not None:
                loss_reg_a = reg_loss_fn(goals, y_reg_b).mean(dim=1)
                loss_reg_b = reg_loss_fn(goals, y_reg_b_perm).mean(dim=1)
                loss_reg = (lam * loss_reg_a * w_b + (1 - lam) * loss_reg_b * w_b_perm).mean()
            else:
                loss_reg = (reg_loss_fn(goals, y_reg_b).mean(dim=1) * w_b).mean()
            loss = loss_cls + lambda_reg * loss_reg
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            ep_loss += loss.item() * xb.size(0)
            ep_correct += (logits.argmax(dim=1) == y_cls_b).sum().item()
            ep_n += xb.size(0)

        # Eval
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
                  f"brier={val_metrics['brier']:.4f}  hgMAE={val_metrics['home_goal_mae']:.3f}  "
                  f"agMAE={val_metrics['away_goal_mae']:.3f}")

        if val_metrics["accuracy"] > best_val_acc + 1e-5:
            best_val_acc = val_metrics["accuracy"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        # SWA: sammle weights ab epoch 50% (oder ab Epoche 30)
        if epoch >= epochs // 2:
            swa.add({k: v.detach().cpu().clone() for k, v in model.state_dict().items()})

        if no_improve >= patience:
            if verbose:
                print(f"      Early stopping @ Ep {epoch}")
            break

    # Lade beste Single-Weight state
    if best_state is not None:
        model.load_state_dict(best_state)

    # Versuche SWA, falls verfuegbar
    swa_state = swa.averaged_state()
    if swa_state is not None:
        # Eval SWA
        model.load_state_dict(swa_state)
        swa_metrics = evaluate_v2(model, val_loader, goal_stats)
        if swa_metrics["accuracy"] > best_val_acc:
            best_val_acc = swa_metrics["accuracy"]
            best_state = swa_state
            model.load_state_dict(best_state)
            if verbose:
                print(f"      SWA improved to {best_val_acc:.4f}")
        else:
            # Wieder zu Best Single zurueck
            model.load_state_dict(best_state)
    return model, goal_stats, best_val_acc


def train_ensemble(
    n_models: int = 5,
    epochs: int = 80,
    batch_size: int = 256,
    lr: float = 1e-3,
    hidden: int = 128,
    n_blocks: int = 4,
    dropout: float = 0.2,
    mixup_alpha: float = 0.3,
    swa_n: int = 5,
    train_start: str = "1990-01-01",
    val_start: str = "2024-01-01",
    half_life_days: float = 1800.0,
    verbose: bool = True,
) -> dict:
    """Trainiert ein Ensemble aus n_models mit verschiedenen Seeds + Architekturen."""
    print("=" * 70)
    print(f" Training V2 (Ensemble={n_models}, Multi-Task, SWA, Mixup)")
    print("=" * 70)
    print(f"   Device: {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"   GPU: {torch.cuda.get_device_name(0)}")

    X_all, y_all, y_hg_all, y_ag_all, dates_all, _, _, feat_names = load_features_v2("all", train_start, val_start)
    print(f"   Total: {len(X_all):,} Samples, {len(feat_names)} Features")

    # Split
    X_tr_raw, X_va_raw = _split(X_all, dates_all, val_start)
    y_cls_tr, y_cls_va = _split(y_all, dates_all, val_start)
    y_reg_tr = np.column_stack([y_hg_all, y_ag_all])
    y_reg_tr_split, y_reg_va_split = _split(y_reg_tr, dates_all, val_start)
    dates_tr, dates_va = _split(dates_all, dates_all, val_start)
    print(f"   Train: {len(X_tr_raw):,} | Val: {len(X_va_raw):,}")

    # Normalize features
    X_tr_n, X_va_n, norm_stats = _normalize_features(X_tr_raw, X_va_raw)
    in_dim = X_tr_n.shape[1]
    print(f"   Features normalisiert, in_dim={in_dim}")

    # Recency weights
    sample_w = _recency_weights(pd.Series(dates_tr), half_life_days=half_life_days)
    sample_w = sample_w / sample_w.mean()  # normalize to mean 1
    print(f"   Recency-Gewichte: min={sample_w.min():.3f}, max={sample_w.max():.3f}, mean={sample_w.mean():.3f}")

    # Class distribution
    cls_dist = np.bincount(y_cls_tr, minlength=3)
    print(f"   Klassenverteilung Train: {cls_dist.tolist()}")

    # Ensemble training
    models = []
    goal_stats_list = []
    val_accs = []
    t0 = time.time()
    configs = [
        {"seed": 42, "hidden": 128, "n_blocks": 4, "dropout": 0.20},
        {"seed": 7,  "hidden": 96,  "n_blocks": 3, "dropout": 0.15},
        {"seed": 13, "hidden": 128, "n_blocks": 5, "dropout": 0.25},
        {"seed": 23, "hidden": 160, "n_blocks": 4, "dropout": 0.20},
        {"seed": 99, "hidden": 96,  "n_blocks": 4, "dropout": 0.30},
    ][:n_models]

    for i, cfg in enumerate(configs):
        if verbose:
            print(f"\n   --- Modell {i+1}/{n_models} (seed={cfg['seed']}, "
                  f"hidden={cfg['hidden']}, blocks={cfg['n_blocks']}, dropout={cfg['dropout']}) ---")
        m, gs, va = train_single_model(
            X_tr_n, y_cls_tr, y_reg_tr_split, sample_w, dates_tr,
            X_va_n, y_cls_va, y_reg_va_split, dates_va,
            in_dim=in_dim,
            hidden=cfg["hidden"], n_blocks=cfg["n_blocks"], dropout=cfg["dropout"],
            epochs=epochs, batch_size=batch_size, lr=lr, weight_decay=1e-4,
            mixup_alpha=mixup_alpha, swa_n=swa_n, seed=cfg["seed"],
            verbose=verbose,
        )
        models.append(m)
        goal_stats_list.append(gs)
        val_accs.append(va)
        if verbose:
            print(f"      Single val_acc = {va:.4f}")

    # Ensemble evaluation
    print("\n   --- Ensemble Evaluation ---")
    ensemble_logits = []
    ensemble_goals = []
    Xv_t = torch.from_numpy(X_va_n).to(DEVICE)
    with torch.no_grad():
        for m in models:
            m.eval()
            logits, goals = m(Xv_t)
            ensemble_logits.append(logits.cpu().numpy())
            ensemble_goals.append(goals.cpu().numpy())
    avg_logits = np.mean(ensemble_logits, axis=0)
    avg_goals = np.mean(ensemble_goals, axis=0)

    # Denormalize goals
    avg_gs = goal_stats_list[0]  # alle gleich (basierend auf Train)
    pred_hg = avg_goals[:, 0] * avg_gs["home_std"] + avg_gs["home_mean"]
    pred_ag = avg_goals[:, 1] * avg_gs["away_std"] + avg_gs["away_mean"]
    pred_hg = np.log1p(np.exp(pred_hg))
    pred_ag = np.log1p(np.exp(pred_ag))

    y_pred = avg_logits.argmax(axis=1)
    ens_acc = (y_pred == y_cls_va).mean()
    probs = _softmax_np(avg_logits)
    eps = 1e-9
    ens_logloss = float(-np.log(probs[np.arange(len(y_cls_va)), y_cls_va] + eps).mean())
    ens_brier = float(((probs - _onehot(y_cls_va, 3)) ** 2).sum(axis=1).mean())

    # Per-class
    per_class = {}
    for c in range(3):
        m_c = y_cls_va == c
        if m_c.sum() > 0:
            per_class[int(c)] = float((y_pred[m_c] == c).mean())

    _ensure_y_va()
    hg_mae = float(np.abs(pred_hg - _y_hg_va).mean())
    ag_mae = float(np.abs(pred_ag - _y_ag_va).mean())

    print(f"   Ensemble val_acc = {ens_acc:.4f}")
    print(f"   Ensemble val_loss = {ens_logloss:.4f}")
    print(f"   Ensemble brier   = {ens_brier:.4f}")
    print(f"   Ensemble per-class acc: {per_class}")
    print(f"   Heimtore MAE: {hg_mae:.3f}, Auswaertstore MAE: {ag_mae:.3f}")

    # ---------- Temperature scaling ----------
    T, cal_acc, cal_loss = fit_temperature(avg_logits, y_cls_va)
    print(f"   Temperature T = {T:.3f} (vor T={1.0})")
    print(f"   Nach Kalibrierung: val_acc={cal_acc:.4f}, val_loss={cal_loss:.4f}")

    train_time = time.time() - t0
    print(f"\n   Trainingsdauer Ensemble: {train_time:.1f}s")
    print(f"   Singletons val_accs: {[f'{v:.4f}' for v in val_accs]}")
    print(f"   Ensemble val_acc: {ens_acc:.4f}")

    # Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = MODELS_DIR / f"v2_ensemble_{timestamp}.pt"
    latest_path = MODELS_DIR / "latest_v2.pt"
    state_dicts = [{k: v.cpu() for k, v in m.state_dict().items()} for m in models]
    bundle = {
        "model_class": "FootballNetEnsemble",
        "state_dicts": state_dicts,
        "in_dim": in_dim,
        "hidden_configs": [c["hidden"] for c in configs],
        "n_blocks_configs": [c["n_blocks"] for c in configs],
        "dropout_configs": [c["dropout"] for c in configs],
        "feature_names": list(feat_names),
        "norm_stats": norm_stats,
        "goal_stats": avg_gs,
        "temperature": T,
        "train_start": train_start,
        "val_start": val_start,
        "half_life_days": half_life_days,
        "n_models": n_models,
        "val_accs_single": val_accs,
        "ensemble_val_acc": float(ens_acc),
        "ensemble_val_loss": float(ens_logloss),
        "ensemble_brier": float(ens_brier),
        "calibrated_val_acc": float(cal_acc),
        "calibrated_val_loss": float(cal_loss),
        "per_class_acc": per_class,
        "hg_mae": hg_mae, "ag_mae": ag_mae,
        "train_time_sec": train_time,
        "device": str(DEVICE),
    }
    torch.save(bundle, model_path)
    tmp = latest_path.with_suffix(".pt.tmp")
    torch.save(bundle, tmp)
    tmp.replace(latest_path)
    with open(MODELS_DIR / "latest_v2_meta.json", "w", encoding="utf-8") as fh:
        json.dump({k: v for k, v in bundle.items() if k != "state_dicts"}, fh, indent=2, ensure_ascii=False)
    print(f"   Modell gespeichert: {latest_path}")
    print("=" * 70)
    return bundle


# Globale Helfer fuer Ensemble-Auswertung
_y_hg_va = None
_y_ag_va = None


def _ensure_y_va():
    global _y_hg_va, _y_ag_va
    if _y_hg_va is None:
        _, y_all, y_hg_all, y_ag_all, dates_all, _, _, _ = load_features_v2("all", "1990-01-01", "2024-01-01")
        mask = dates_all >= pd.Timestamp("2024-01-01")
        _y_hg_va = y_hg_all[mask]
        _y_ag_va = y_ag_all[mask]


def fit_temperature(logits: np.ndarray, y_true: np.ndarray, max_iter: int = 100) -> tuple[float, float, float]:
    """Optimiert eine einzige Temperatur T sodass NLL minimiert wird."""
    from scipy.optimize import minimize_scalar
    _ensure_y_va()
    logits_t = torch.tensor(logits, dtype=torch.float32)
    y_t = torch.tensor(y_true, dtype=torch.long)

    def nll(T):
        T = max(T, 0.01)
        p = F.softmax(logits_t / T, dim=1).numpy()
        eps = 1e-9
        return -np.log(p[np.arange(len(y_true)), y_true] + eps).mean()

    res = minimize_scalar(nll, bounds=(0.3, 5.0), method="bounded")
    T = float(res.x)
    # Recompute
    p_cal = F.softmax(logits_t / T, dim=1).numpy()
    acc = float((p_cal.argmax(axis=1) == y_true).mean())
    eps = 1e-9
    loss = float(-np.log(p_cal[np.arange(len(y_true)), y_true] + eps).mean())
    return T, acc, loss


# -------------------- Inference helpers --------------------

def _load_latest_v2() -> tuple[list[nn.Module], dict]:
    """Laedt das aktuellste v2 Ensemble."""
    bundle = torch.load(MODELS_DIR / "latest_v2.pt", map_location=DEVICE, weights_only=False)
    models = []
    for i, sd in enumerate(bundle["state_dicts"]):
        m = FootballNet(
            in_dim=bundle["in_dim"],
            hidden=bundle["hidden_configs"][i],
            n_blocks=bundle["n_blocks_configs"][i],
            dropout=bundle["dropout_configs"][i],
        ).to(DEVICE)
        m.load_state_dict(sd)
        m.eval()
        models.append(m)
    return models, bundle


def _build_inference_vector_v2(home: str, away: str, neutral: bool, tournament: str, today: datetime) -> np.ndarray:
    """32-dim Feature-Vektor fuer ein hypothetisches Spiel (V2)."""
    state = get_current_team_ratings()
    home_n = normalize_team_name(home)
    away_n = normalize_team_name(away)
    h = state.get(home_n)
    a = state.get(away_n)
    if h is None or a is None:
        missing = [t for t, s in [(home_n, h), (away_n, a)] if s is None]
        raise ValueError(f"Unbekanntes Team: {missing}. Verfuegbar: "
                         f"{', '.join(sorted(state.keys(), key=lambda t: -state[t]['elo'])[:20])}")
    # Rest days
    rest_h = 30
    rest_a = 30
    if h["last_match"]:
        last = datetime.fromisoformat(h["last_match"])
        rest_h = min((today - last).days, 365)
    if a["last_match"]:
        last = datetime.fromisoformat(a["last_match"])
        rest_a = min((today - last).days, 365)
    # H2H
    raw = pd.read_csv(RAW_RESULTS, parse_dates=["date"])
    raw = raw.dropna(subset=["home_score", "away_score"])
    pair = raw[((raw["home_team"] == home_n) & (raw["away_team"] == away_n)) |
               ((raw["home_team"] == away_n) & (raw["away_team"] == home_n))].tail(10)
    h2h_h, h2h_a = 0.5, 0.5
    if len(pair) > 0:
        wins_h, wins_a = 0, 0
        for _, r in pair.iterrows():
            if r["home_team"] == home_n:
                if r["home_score"] > r["away_score"]:
                    wins_h += 1
                elif r["home_score"] < r["away_score"]:
                    wins_a += 1
            else:
                if r["away_score"] > r["home_score"]:
                    wins_h += 1
                elif r["away_score"] < r["home_score"]:
                    wins_a += 1
        n = len(pair)
        h2h_h = wins_h / n
        h2h_a = wins_a / n
    # Continent
    cont_h = _team_to_continent(home_n)
    cont_a = _team_to_continent(away_n)
    # Tournament form
    raw2 = raw[raw["tournament"] == tournament] if tournament else raw
    tform_h = 1.0
    tform_a = 1.0
    h_recent = raw2[raw2["home_team"] == home_n].tail(5)
    a_recent = raw2[raw2["away_team"] == away_n].tail(5)
    if len(h_recent) > 0:
        pts = sum(3 for _, r in h_recent.iterrows() if r["home_score"] > r["away_score"])
        pts += sum(1 for _, r in h_recent.iterrows() if r["home_score"] == r["away_score"])
        tform_h = pts / (3 * len(h_recent))
    if len(a_recent) > 0:
        pts = sum(3 for _, r in a_recent.iterrows() if r["away_score"] > r["home_score"])
        pts += sum(1 for _, r in a_recent.iterrows() if r["home_score"] == r["away_score"])
        tform_a = pts / (3 * len(a_recent))

    elo_h_eff = h["elo"] + (0 if neutral else HOME_ADVANTAGE_ELO)
    feature_cols = [
        "neutral", "tournament_w",
        "elo_a", "elo_b", "elo_diff", "elo_mom_a", "elo_mom_b",
        "form3_a", "form3_b", "form5_a", "form5_b", "form10_a", "form10_b",
        "gf5_a", "gf5_b", "ga5_a", "ga5_b", "gd5_a", "gd5_b",
        "home_form5_a", "away_form5_b",
        "h2h_a", "h2h_b",
        "rest_a", "rest_b",
        "win_streak_a", "win_streak_b", "unbeaten_a",
        "continent_a", "continent_b", "oppo_elo5_a", "oppo_elo5_b",
        "tour_form_a", "tour_form_b",
    ]
    d = {
        "neutral": int(neutral), "tournament_w": tournament_weight(tournament),
        "elo_a": h["elo"], "elo_b": a["elo"],
        "elo_diff": elo_h_eff - a["elo"],
        "elo_mom_a": h["elo_mom"], "elo_mom_b": a["elo_mom"],
        "form3_a": h["form3"], "form3_b": a["form3"],
        "form5_a": h["form5"], "form5_b": a["form5"],
        "form10_a": h["form10"], "form10_b": a["form10"],
        "gf5_a": h["gf5"], "gf5_b": a["gf5"],
        "ga5_a": h["ga5"], "ga5_b": a["ga5"],
        "gd5_a": h["gd5"], "gd5_b": a["gd5"],
        "home_form5_a": h["home_form5"], "away_form5_b": a["away_form5"],
        "h2h_a": h2h_h, "h2h_b": h2h_a,
        "rest_a": rest_h, "rest_b": rest_a,
        "win_streak_a": h["win_streak"], "win_streak_b": a["win_streak"],
        "unbeaten_a": h["unbeaten"],
        "continent_a": cont_h, "continent_b": cont_a,
        "oppo_elo5_a": h["oppo_elo5"], "oppo_elo5_b": a["oppo_elo5"],
        "tour_form_a": tform_h, "tour_form_b": tform_a,
    }
    return np.array([d[k] for k in feature_cols], dtype=np.float32)


def predict_match_v2(
    home: str, away: str, neutral: bool = True, tournament: str = "FIFA World Cup",
    today: datetime | None = None,
) -> dict:
    """Prognose mit dem V2-Ensemble + Score-Regression."""
    if today is None:
        today = datetime.now()
    models, bundle = _load_latest_v2()
    norm = bundle["norm_stats"]
    mean = np.array(norm["mean"], dtype=np.float32)
    std = np.array(norm["std"], dtype=np.float32)
    gs = bundle["goal_stats"]
    T = bundle.get("temperature", 1.0)

    home_n = normalize_team_name(home)
    away_n = normalize_team_name(away)

    vec = _build_inference_vector_v2(home, away, neutral, tournament, today)
    vec_n = (vec - mean) / std
    x = torch.from_numpy(vec_n).to(DEVICE).unsqueeze(0)
    with torch.no_grad():
        logits_list = []
        goals_list = []
        for m in models:
            l, g = m(x)
            logits_list.append(F.softmax(l, dim=1).cpu().numpy()[0])
            goals_list.append(g.cpu().numpy()[0])
    probs = np.mean(logits_list, axis=0)
    # Apply temperature
    logits_avg = np.log(probs + 1e-9).reshape(1, -1)  # invert softmax, ensure 2D
    probs_cal = _softmax_np(logits_avg / T)[0]

    goals_avg = np.mean(goals_list, axis=0)
    # Denormalize + softplus
    pred_hg = goals_avg[0] * gs["home_std"] + gs["home_mean"]
    pred_ag = goals_avg[1] * gs["away_std"] + gs["away_mean"]
    pred_hg = float(np.log1p(np.exp(pred_hg)))
    pred_ag = float(np.log1p(np.exp(pred_ag)))

    out = {
        "home": home_n,
        "away": away_n,
        "neutral": neutral,
        "tournament": tournament,
        "as_of": today.isoformat(),
        "probabilities": {
            "draw": float(probs_cal[0]),
            "home_win": float(probs_cal[1]),
            "away_win": float(probs_cal[2]),
        },
        "probabilities_uncalibrated": {
            "draw": float(probs[0]),
            "home_win": float(probs[1]),
            "away_win": float(probs[2]),
        },
        "labels": {
            "draw": "Unentschieden",
            "home_win": f"Sieg {home_n}",
            "away_win": f"Sieg {away_n}",
        },
        "argmax_label": {0: "Unentschieden", 1: f"Sieg {home_n}", 2: f"Sieg {away_n}"}[int(probs_cal.argmax())],
        "expected_score": {
            "home_goals": pred_hg,
            "away_goals": pred_ag,
            "display": f"{pred_hg:.2f} : {pred_ag:.2f}",
        },
        "elo_home": float(vec[2]),
        "elo_away": float(vec[3]),
        "model_version": "v2_ensemble",
        "ensemble_size": bundle["n_models"],
    }
    # Wahrscheinlichste exakte Scores via unabhängige Poissons
    out["most_likely_scores"] = _most_likely_scores(pred_hg, pred_ag, top_k=5)
    return out


def _most_likely_scores(exp_h: float, exp_a: float, top_k: int = 5) -> list[dict]:
    """Berechnet die wahrscheinlichsten exakten Spielergebnisse (unabhaengige Poissons)."""
    from math import exp, factorial
    exp_h = min(max(exp_h, 0.1), 5.0)
    exp_a = min(max(exp_a, 0.1), 5.0)

    def poisson_p(k: int, lam: float) -> float:
        return exp(-lam) * (lam ** k) / factorial(k)

    results = []
    for h in range(7):
        for a in range(7):
            p = poisson_p(h, exp_h) * poisson_p(a, exp_a)
            results.append({"home": h, "away": a, "prob": float(p)})
    results.sort(key=lambda x: -x["prob"])
    return results[:top_k]


def main() -> int:
    train_ensemble(verbose=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
