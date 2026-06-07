"""V5 Training: 90-Tage Recency + V4-Big Architektur."""
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
from src.features_v5 import load_features_v5
from src.train_v2 import FootballNet, _recency_weights, _normalize_features, _split, evaluate_v2
from src.export_browser import _serialize_one_model
from scripts.quantize_v4_int8 import main as run_quantize_v4_int8

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 70)
print(" V5 Training: 90-Tage Recency + V4-Big Architektur")
print("=" * 70)

X_all, y_all, y_hg_all, y_ag_all, dates_all, _, _, feat_names = load_features_v5("all", "2018-01-01", "2024-01-01")
X_tr_raw, X_va_raw = _split(X_all, dates_all, "2024-01-01")
y_cls_tr, y_cls_va = _split(y_all, dates_all, "2024-01-01")
y_reg_tr = np.column_stack([y_hg_all, y_ag_all])
y_reg_tr_split, y_reg_va_split = _split(y_reg_tr, dates_all, "2024-01-01")
dates_tr, dates_va = _split(dates_all, dates_all, "2024-01-01")
print(f"   Total: {len(X_all):,} | Train: {len(X_tr_raw):,} | Val: {len(X_va_raw):,}")

X_tr_n, X_va_n, norm_stats = _normalize_features(X_tr_raw, X_va_raw)
in_dim = X_tr_n.shape[1]
sample_w = _recency_weights(pd.Series(dates_tr), half_life_days=90.0)  # AGGRESSIVE: 90 Tage
sample_w = sample_w / sample_w.mean()
print(f"   Half-Life: 90 Tage (sehr aggressiv)")
print(f"   Sample weights: min={sample_w.min():.4f}, max={sample_w.max():.2f}")

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
train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, pin_memory=True)
val_loader = DataLoader(val_ds, batch_size=1024, shuffle=False, pin_memory=True)

# V4 Big Architektur
HIDDEN = 256
N_BLOCKS = 8
DROPOUT = 0.30
LR = 1e-3
EPOCHS = 200
LABEL_SMOOTH = 0.10

torch.manual_seed(42)
np.random.seed(42)

model = FootballNet(in_dim=in_dim, hidden=HIDDEN, n_blocks=N_BLOCKS, dropout=DROPOUT).to(DEVICE)
n_params = sum(p.numel() for p in model.parameters())
print(f"   Model params: {n_params:,}")

opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=5e-4)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=EPOCHS*len(train_loader),
                                            pct_start=0.05, anneal_strategy="cos")

counts = np.bincount(y_cls_tr, minlength=3).astype(np.float64)
cw = counts.sum() / (3 * counts + 1e-9)
cw[0] *= 1.2
cw_t = torch.tensor(cw, dtype=torch.float32, device=DEVICE)
cls_loss = nn.CrossEntropyLoss(weight=cw_t, label_smoothing=LABEL_SMOOTH, reduction="none")
reg_loss = nn.SmoothL1Loss(reduction="none")
lambda_reg = 0.4

best_val_acc = 0.0
best_state = None
patience = 20
no_imp = 0

t0 = time.time()
for ep in range(1, EPOCHS + 1):
    model.train()
    ep_loss = 0.0
    ep_correct = 0
    ep_n = 0
    for batch in train_loader:
        xb, yb, rb, wb = [t.to(DEVICE, non_blocking=True) for t in batch]
        opt.zero_grad(set_to_none=True)
        logits, goals = model(xb)
        l_cls = (cls_loss(logits, yb) * wb).mean()
        l_reg = (reg_loss(goals, rb).mean(dim=1) * wb).mean()
        loss = l_cls + lambda_reg * l_reg
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        ep_loss += loss.item() * xb.size(0)
        ep_correct += (logits.argmax(dim=1) == yb).sum().item()
        ep_n += xb.size(0)
    if ep <= 3 or ep % 5 == 0:
        vm = evaluate_v2(model, val_loader, goal_stats)
        elapsed = time.time() - t0
        print(f"   Ep {ep:3d}/{EPOCHS}  tr_loss={ep_loss/ep_n:.4f}  tr_acc={ep_correct/ep_n:.4f}  "
              f"v_acc={vm['accuracy']:.4f}  v_loss={vm['log_loss']:.4f}  brier={vm['brier']:.4f}  ({elapsed:.0f}s)")
        if vm["accuracy"] > best_val_acc + 1e-5:
            best_val_acc = vm["accuracy"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_imp = 0
        else:
            no_imp += 1
            if no_imp >= patience:
                print(f"   Early stopping @ Ep {ep}")
                break
    else:
        # Eval alle 2-3 Epochen
        if ep % 3 == 0:
            vm = evaluate_v2(model, val_loader, goal_stats)
            if vm["accuracy"] > best_val_acc + 1e-5:
                best_val_acc = vm["accuracy"]
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                no_imp = 0
            else:
                no_imp += 1
                if no_imp >= patience:
                    print(f"   Early stopping @ Ep {ep}")
                    break

if best_state is not None:
    model.load_state_dict(best_state)
final = evaluate_v2(model, val_loader, goal_stats)
print()
print(f"   Best val_acc: {best_val_acc:.4f}")
print(f"   Per-class: {final['per_class_acc']}")
print(f"   High-conf-acc: {final['high_conf_acc']:.4f} on {final['n_high_conf']} samples")

# Speichere (float32)
model.eval()
state_dict = _serialize_one_model(model)
bundle = {
    "architecture": {
        "n_models": 1,
        "in_dim": in_dim,
        "hidden": HIDDEN,
        "n_blocks": N_BLOCKS,
        "n_classes": 3,
        "version": "v5-aggressive-recency",
        "n_params": n_params,
    },
    "models": [state_dict],
    "norm_stats": norm_stats,
    "goal_stats": goal_stats,
    "temperature": 1.0,
    "ensemble_val_acc": final["accuracy"],
    "calibrated_val_acc": final["accuracy"],
    "feature_names": list(feat_names),
    "n_features": in_dim,
    "train_start": "2018-01-01",
    "half_life_days": 90.0,
}
out_path = Path("E:/Profilov2/public/data/wm-predictor/model.json")
with open(out_path, "w", encoding="utf-8") as fh:
    json.dump(bundle, fh, separators=(",", ":"))
print(f"   geschrieben (f32): {out_path} ({out_path.stat().st_size/1e6:.2f} MB)")

# Int8 quantize
print()
print("   Quantisiere zu int8...")
run_quantize_v4_int8()
print()
print("=" * 70)
