"""V4: Drop cumulative Elo entirely. Use only Recent-Elo + form-based features.

Hypothesis: Cumulative Elo über 150 Jahre verrauscht die aktuelle Signalstärke.
Lösung: Komplett auf Recent-Elo (2y) + Opponent-Weighted Form setzen.
"""
import json
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path

import sys
sys.path.insert(0, "E:/Projects/Fussball_ai")
from src.features_v3 import load_features_v3, get_current_team_ratings_v3, _compute_final_state_v3, PROCESSED_DIR
from src.train_v2 import FootballNet, _recency_weights, _normalize_features, _split, evaluate_v2, fit_temperature
from src.export_browser import _serialize_one_model

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Lade V3 features
X_all, y_all, y_hg_all, y_ag_all, dates_all, _, _, feat_names = load_features_v3("all", "2010-01-01", "2024-01-01")

# DROPPED FEATURES: cumulative Elo (idx 2, 3, 4)
# Keep: re_elo, form, momentum, wins_top, etc.
DROP = ["elo_a", "elo_b", "elo_diff"]
keep_idx = [i for i, n in enumerate(feat_names) if n not in DROP]
X_all = X_all[:, keep_idx]
feat_names_kept = [n for n in feat_names if n not in DROP]
print(f"V4: {X_all.shape[1]} features (dropped: {DROP})")

X_tr_raw, X_va_raw = _split(X_all, dates_all, "2024-01-01")
y_cls_tr, y_cls_va = _split(y_all, dates_all, "2024-01-01")
y_reg_tr = np.column_stack([y_hg_all, y_ag_all])
y_reg_tr_split, y_reg_va_split = _split(y_reg_tr, dates_all, "2024-01-01")
dates_tr, dates_va = _split(dates_all, dates_all, "2024-01-01")

X_tr_n, X_va_n, norm_stats = _normalize_features(X_tr_raw, X_va_raw)
in_dim = X_tr_n.shape[1]
sample_w = _recency_weights(pd.Series(dates_tr), half_life_days=240.0)
sample_w = sample_w / sample_w.mean()

# Goal stats
hg_mean = float(y_reg_tr_split[:, 0].mean())
hg_std = float(y_reg_tr_split[:, 0].std() + 1e-6)
ag_mean = float(y_reg_tr_split[:, 1].mean())
ag_std = float(y_reg_tr_split[:, 1].std() + 1e-6)
goal_stats = {"home_mean": hg_mean, "home_std": hg_std, "away_mean": ag_mean, "away_std": ag_std}
y_reg_tr_n = (y_reg_tr_split - np.array([hg_mean, ag_mean])) / np.array([hg_std, ag_std])
y_reg_va_n = (y_reg_va_split - np.array([hg_mean, ag_mean])) / np.array([hg_std, ag_std])

Xt = torch.from_numpy(X_tr_n)
y_cls_t = torch.from_numpy(y_cls_tr).long()
y_reg_t = torch.from_numpy(y_reg_tr_n.astype(np.float32))
w_t = torch.from_numpy(sample_w)
Xv = torch.from_numpy(X_va_n)
y_cls_v = torch.from_numpy(y_cls_va).long()
y_reg_v = torch.from_numpy(y_reg_va_n.astype(np.float32))

train_ds = TensorDataset(Xt, y_cls_t, y_reg_t, w_t)
val_ds = TensorDataset(Xv, y_cls_v, y_reg_v)
train_loader = DataLoader(train_ds, batch_size=256, shuffle=True, pin_memory=True)
val_loader = DataLoader(val_ds, batch_size=1024, shuffle=False, pin_memory=True)

hidden = 128
n_blocks = 4
model = FootballNet(in_dim=in_dim, hidden=hidden, n_blocks=n_blocks, dropout=0.25).to(DEVICE)
opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, total_steps=100*len(train_loader),
                                            pct_start=0.1, anneal_strategy="cos")
counts = np.bincount(y_cls_tr, minlength=3).astype(np.float64)
cw = counts.sum() / (3 * counts + 1e-9)
cw[0] *= 1.2
cw_t = torch.tensor(cw, dtype=torch.float32, device=DEVICE)
cls_loss = nn.CrossEntropyLoss(weight=cw_t, label_smoothing=0.05, reduction="none")
reg_loss = nn.SmoothL1Loss(reduction="none")
lambda_reg = 0.5

best_val_acc = 0.0
best_state = None
patience = 15
no_imp = 0
print(f"\nTraining V4 ({in_dim} features, no cumulative Elo)...")

for ep in range(1, 101):
    model.train()
    for batch in train_loader:
        xb, yb, rb, wb = [t.to(DEVICE, non_blocking=True) for t in batch]
        opt.zero_grad(set_to_none=True)
        logits, goals = model(xb)
        l = (cls_loss(logits, yb) * wb).mean() + lambda_reg * (reg_loss(goals, rb).mean(dim=1) * wb).mean()
        l.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
    vm = evaluate_v2(model, val_loader, goal_stats)
    if ep <= 3 or ep % 5 == 0:
        print(f"  Ep {ep:3d}  v_acc={vm['accuracy']:.4f}  v_loss={vm['log_loss']:.4f}  brier={vm['brier']:.4f}")
    if vm["accuracy"] > best_val_acc + 1e-5:
        best_val_acc = vm["accuracy"]
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        no_imp = 0
    else:
        no_imp += 1
        if no_imp >= patience:
            print(f"  Early stopping @ Ep {ep}")
            break

if best_state is not None:
    model.load_state_dict(best_state)
final = evaluate_v2(model, val_loader, goal_stats)
print(f"\nV4 best val_acc: {best_val_acc:.4f}")
print(f"V4 final: {final}")

# Test Germany vs Uruguay
model.eval()
state = get_current_team_ratings_v3()
g, u = state["Germany"], state["Uruguay"]
re_elo_h, re_elo_a = g["re_elo"], u["re_elo"]
print(f"\nGermany re_elo: {re_elo_h:.0f}, Uruguay re_elo: {re_elo_a:.0f}, diff: {re_elo_h - re_elo_a:+.0f}")
print(f"Germany form5: {g['form5']:.2f}, momentum: {g['momentum']:+.2f}, w_form: {g['w_form']:.2f}, streak: {g['win_streak']}")
print(f"Uruguay form5: {u['form5']:.2f}, momentum: {u['momentum']:+.2f}, w_form: {u['w_form']:.2f}, streak: {u['win_streak']}")
