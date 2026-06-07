"""Final V3: Train ensemble of 5 models with different seeds for stability."""
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path

import sys
sys.path.insert(0, "E:/Projects/Fussball_ai")
from src.features_v3 import load_features_v3, get_current_team_ratings_v3
from src.train_v2 import FootballNet, _recency_weights, _normalize_features, _split, evaluate_v2
from src.export_browser import _serialize_one_model

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

X_all, y_all, y_hg_all, y_ag_all, dates_all, _, _, feat_names = load_features_v3("all", "2010-01-01", "2024-01-01")
print(f"V3 features: {X_all.shape[1]}")

X_tr_raw, X_va_raw = _split(X_all, dates_all, "2024-01-01")
y_cls_tr, y_cls_va = _split(y_all, dates_all, "2024-01-01")
y_reg_tr = np.column_stack([y_hg_all, y_ag_all])
y_reg_tr_split, y_reg_va_split = _split(y_reg_tr, dates_all, "2024-01-01")
dates_tr, dates_va = _split(dates_all, dates_all, "2024-01-01")

X_tr_n, X_va_n, norm_stats = _normalize_features(X_tr_raw, X_va_raw)
in_dim = X_tr_n.shape[1]
sample_w = _recency_weights(pd.Series(dates_tr), half_life_days=240.0)
sample_w = sample_w / sample_w.mean()

hg_mean = float(y_reg_tr_split[:, 0].mean())
hg_std = float(y_reg_tr_split[:, 0].std() + 1e-6)
ag_mean = float(y_reg_tr_split[:, 1].mean())
ag_std = float(y_reg_tr_split[:, 1].std() + 1e-6)
goal_stats = {"home_mean": hg_mean, "home_std": hg_std, "away_mean": ag_mean, "away_std": ag_std}
y_reg_tr_n = (y_reg_tr_split - np.array([hg_mean, ag_mean])) / np.array([hg_std, ag_std])
y_reg_va_n = (y_reg_va_split - np.array([hg_mean, ag_mean])) / np.array([hg_std, ag_std])

# Common tensors
Xt = torch.from_numpy(X_tr_n)
y_cls_t = torch.from_numpy(y_cls_tr).long()
y_reg_t = torch.from_numpy(y_reg_tr_n.astype(np.float32))
w_t = torch.from_numpy(sample_w)
Xv = torch.from_numpy(X_va_n)
y_cls_v = torch.from_numpy(y_cls_va).long()
y_reg_v = torch.from_numpy(y_reg_va_n.astype(np.float32))
val_ds = TensorDataset(Xv, y_cls_v, y_reg_v)
val_loader = DataLoader(val_ds, batch_size=1024, shuffle=False, pin_memory=True)

# 5 different configs
configs = [
    {"seed": 42, "hidden": 96, "n_blocks": 4, "dropout": 0.20},
    {"seed": 7,  "hidden": 128, "n_blocks": 3, "dropout": 0.15},
    {"seed": 13, "hidden": 96, "n_blocks": 5, "dropout": 0.25},
    {"seed": 23, "hidden": 128, "n_blocks": 4, "dropout": 0.20},
    {"seed": 99, "hidden": 64, "n_blocks": 4, "dropout": 0.30},
]

models_serialized = []
ensemble_val_accs = []

for cfg in configs:
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    print(f"\n--- Model seed={cfg['seed']} h={cfg['hidden']} b={cfg['n_blocks']} d={cfg['dropout']} ---")
    train_ds = TensorDataset(Xt, y_cls_t, y_reg_t, w_t)
    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True, pin_memory=True)
    model = FootballNet(in_dim=in_dim, hidden=cfg["hidden"], n_blocks=cfg["n_blocks"], dropout=cfg["dropout"]).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, total_steps=100*len(train_loader),
                                                pct_start=0.1, anneal_strategy="cos")
    counts = np.bincount(y_cls_tr, minlength=3).astype(np.float64)
    cw = counts.sum() / (3 * counts + 1e-9)
    cw[0] *= 1.2
    cw_t = torch.tensor(cw, dtype=torch.float32, device=DEVICE)
    cls_loss = nn.CrossEntropyLoss(weight=cw_t, label_smoothing=0.05, reduction="none")
    reg_loss = nn.SmoothL1Loss(reduction="none")

    best_val = 0
    best_st = None
    no_imp = 0
    for ep in range(1, 81):
        model.train()
        for batch in train_loader:
            xb, yb, rb, wb = [t.to(DEVICE, non_blocking=True) for t in batch]
            opt.zero_grad(set_to_none=True)
            logits, goals = model(xb)
            l = (cls_loss(logits, yb) * wb).mean() + 0.5 * (reg_loss(goals, rb).mean(dim=1) * wb).mean()
            l.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
        vm = evaluate_v2(model, val_loader, goal_stats)
        if vm["accuracy"] > best_val + 1e-5:
            best_val = vm["accuracy"]
            best_st = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_imp = 0
        else:
            no_imp += 1
            if no_imp >= 12:
                break
    model.load_state_dict(best_st)
    model.eval()
    print(f"   single val_acc: {best_val:.4f}")
    ensemble_val_accs.append(best_val)
    models_serialized.append(_serialize_one_model(model))

# Ensemble eval
print(f"\n--- Ensemble Evaluation ---")
avg_logits = []
avg_goals = []
Xv_t = torch.from_numpy(X_va_n).to(DEVICE)
for sd in models_serialized:
    # Load model
    pass
# Easier: re-load from bundle architecture
arch = {"n_models": len(models_serialized), "in_dim": in_dim, "hidden": 96, "n_blocks": 4, "n_classes": 3, "version": "v3-ensemble"}
# Use first model's arch info
for i, sd in enumerate(models_serialized):
    pass

# Manual ensemble: pass inputs through each model
all_probs = []
all_goals = []
for i, sd in enumerate(models_serialized):
    cfg = configs[i]
    m = FootballNet(in_dim=in_dim, hidden=cfg["hidden"], n_blocks=cfg["n_blocks"], dropout=cfg["dropout"]).to(DEVICE)
    # Convert folded keys to unfolded
    new_sd = {}
    new_sd["input_proj.0.weight"] = torch.tensor(sd["input_w"])
    new_sd["input_proj.0.bias"] = torch.tensor(sd["input_b"])
    for b in range(cfg["n_blocks"]):
        new_sd[f"blocks.{b}.lin1.weight"] = torch.tensor(sd[f"block{b}_w1"])
        new_sd[f"blocks.{b}.lin1.bias"] = torch.tensor(sd[f"block{b}_b1"])
        new_sd[f"blocks.{b}.lin2.weight"] = torch.tensor(sd[f"block{b}_w2"])
        new_sd[f"blocks.{b}.lin2.bias"] = torch.tensor(sd[f"block{b}_b2"])
    new_sd["cls_head.0.weight"] = torch.tensor(sd["cls_w1"])
    new_sd["cls_head.0.bias"] = torch.tensor(sd["cls_b1"])
    new_sd["cls_head.3.weight"] = torch.tensor(sd["cls_w2"])
    new_sd["cls_head.3.bias"] = torch.tensor(sd["cls_b2"])
    new_sd["reg_head.0.weight"] = torch.tensor(sd["reg_w1"])
    new_sd["reg_head.0.bias"] = torch.tensor(sd["reg_b1"])
    new_sd["reg_head.3.weight"] = torch.tensor(sd["reg_w2"])
    new_sd["reg_head.3.bias"] = torch.tensor(sd["reg_b2"])
    m.load_state_dict(new_sd, strict=False)
    m.eval()
    with torch.no_grad():
        logits, goals = m(Xv_t)
        probs = F.softmax(logits, dim=1).cpu().numpy()
        all_probs.append(probs)
        all_goals.append(goals.cpu().numpy())
    del m
    torch.cuda.empty_cache()

avg_probs = np.mean(all_probs, axis=0)
avg_goals = np.mean(all_goals, axis=0)
ens_pred = avg_probs.argmax(axis=1)
ens_acc = float((ens_pred == y_cls_va).mean())
print(f"Ensemble val_acc: {ens_acc:.4f}")
print(f"Single val_accs: {[f'{v:.4f}' for v in ensemble_val_accs]}")

# Save
bundle = {
    "architecture": {
        "n_models": len(models_serialized),
        "in_dim": in_dim,
        "hidden": 96,  # default for inference
        "n_blocks": 4,
        "n_classes": 3,
        "version": "v3-ensemble",
        "hidden_configs": [c["hidden"] for c in configs],
        "n_blocks_configs": [c["n_blocks"] for c in configs],
        "dropout_configs": [c["dropout"] for c in configs],
    },
    "models": models_serialized,
    "norm_stats": norm_stats,
    "goal_stats": goal_stats,
    "temperature": 1.0,
    "ensemble_val_acc": ens_acc,
    "calibrated_val_acc": ens_acc,
    "feature_names": list(feat_names),
    "n_features": in_dim,
    "train_start": "2010-01-01",
    "half_life_days": 240.0,
}
out_path = Path("E:/Profilov2/public/data/wm-predictor/model.json")
with open(out_path, "w", encoding="utf-8") as fh:
    json.dump(bundle, fh, separators=(",", ":"))
print(f"\ngeschrieben: {out_path} ({out_path.stat().st_size/1e6:.2f} MB)")
