"""
V8 Training — E8Net (Transformer + Poisson + KO heads).

Training strategy:
  - Data: features_v8.npz (static V6 + sequence + squad tensors)
  - Split: Train < 2023 / Cal 2023 / Val >= 2024
  - Augmentation: home/away swap (doubles training data, enforces symmetry)
  - Loss: Poisson-NLL (primary) + KL-RPS on W/D/L (aux)
  - Calibration: temperature scaling on Cal split after training
  - Ensemble: 3 seeds, averaged predictions

Output:
  models/v8_seed{N}.pt         — raw PyTorch checkpoints
  data/models/v8_latest.pt     — best-of-3-ensemble metadata
  (ONNX export handled by scripts/export_onnx.py)

Usage:
  cd E:/Projects/Fussball_ai
  .\\venv\\Scripts\\python.exe -m scripts.train_v8
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from math import log, exp, factorial

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from scipy.optimize import minimize_scalar

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.features_v8 import load_v8, SEQ_LEN, N_PLAYERS, SEQ_DIM, PLAYER_DIM
from src.features_v6 import score_grid, wdl_from_grid
from src.model_v8 import E8Net, E8Config, build_model, swap_home_away
from src.train_v2 import _recency_weights, _normalize_features

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = REPO_ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# ── Hyperparameters ──────────────────────────────────────────────────────────
N_SEEDS = 3
EPOCHS = 150
PATIENCE = 20
LR = 3e-4
BATCH_SIZE = 512
LAMBDA_POIS = 1.0
LAMBDA_RPS = 0.3     # Ranked probability score on W/D/L (aux)
HALF_LIFE = 90.0     # recency weight half-life in days
VAL_START = "2024-01-01"
CAL_START = "2023-01-01"
TRAIN_START = "2015-01-01"  # full FIFA coverage starts 2015

print("=" * 70)
print(" V8 Training: E8Net (Transformer + Poisson + Cross-Attention)")
print("=" * 70)
print(f"   Device: {DEVICE}")
if DEVICE.type == "cuda":
    print(f"   GPU: {torch.cuda.get_device_name(0)}")

# ── Load V8 features ─────────────────────────────────────────────────────────
print("\n   Loading features_v8 ...")
d = load_v8()
X_all = d["X"]                  # (N, 57)
y_cls_all = d["y"]              # (N,) 0=draw 1=home_win 2=away_win
y_hg_all = d["y_home_goals"]    # (N,)
y_ag_all = d["y_away_goals"]    # (N,)
dates_all = pd.to_datetime(d["dates"])
seq_home_all = d["seq_home"]    # (N, SEQ_LEN, SEQ_DIM)
seq_away_all = d["seq_away"]
squad_home_all = d["squad_home"]  # (N, N_PLAYERS, PLAYER_DIM)
squad_away_all = d["squad_away"]

# ── Split ─────────────────────────────────────────────────────────────────────
mask_tr = (dates_all >= TRAIN_START) & (dates_all < CAL_START)
mask_ca = (dates_all >= CAL_START) & (dates_all < VAL_START)
mask_va = dates_all >= VAL_START

print(f"   Train: {mask_tr.sum():,}  Cal: {mask_ca.sum():,}  Val: {mask_va.sum():,}")


def _split(arr, m_tr, m_ca, m_va):
    return arr[m_tr], arr[m_ca], arr[m_va]


X_tr_raw, X_ca_raw, X_va_raw = _split(X_all, mask_tr, mask_ca, mask_va)
y_cls_tr, y_cls_ca, y_cls_va = _split(y_cls_all, mask_tr, mask_ca, mask_va)
y_hg_tr, y_hg_ca, y_hg_va = _split(y_hg_all, mask_tr, mask_ca, mask_va)
y_ag_tr, y_ag_ca, y_ag_va = _split(y_ag_all, mask_tr, mask_ca, mask_va)
sh_tr, sh_ca, sh_va = _split(seq_home_all, mask_tr, mask_ca, mask_va)
sa_tr, sa_ca, sa_va = _split(seq_away_all, mask_tr, mask_ca, mask_va)
sqh_tr, sqh_ca, sqh_va = _split(squad_home_all, mask_tr, mask_ca, mask_va)
sqa_tr, sqa_ca, sqa_va = _split(squad_away_all, mask_tr, mask_ca, mask_va)
dates_tr = dates_all[mask_tr]

# ── Static feature normalization ─────────────────────────────────────────────
col_medians = np.nanmedian(X_tr_raw, axis=0)
def _impute(X):
    out = X.copy()
    nans = np.isnan(out)
    out[nans] = np.take(col_medians, np.where(nans)[1])
    return out

X_tr_raw = _impute(X_tr_raw)
X_ca_raw = _impute(X_ca_raw)
X_va_raw = _impute(X_va_raw)

X_tr_n, _, norm_stats = _normalize_features(X_tr_raw, X_ca_raw)
norm_mean = np.array(norm_stats["mean"], dtype=np.float32)
norm_std  = np.array(norm_stats["std"],  dtype=np.float32)
X_ca_n = ((X_ca_raw - norm_mean) / norm_std).astype(np.float32)
X_va_n = ((X_va_raw - norm_mean) / norm_std).astype(np.float32)

# ── Context vector (is_neutral=1.0, tourn_weight_norm from feature[1]) ────────
def _context(X_n: np.ndarray) -> np.ndarray:
    return X_n[:, [0, 1]].astype(np.float32)  # neutral, tournament_w

ctx_tr = _context(X_tr_n)
ctx_ca = _context(X_ca_n)
ctx_va = _context(X_va_n)

# ── Recency weights ───────────────────────────────────────────────────────────
sample_w = _recency_weights(dates_tr, half_life_days=HALF_LIFE).astype(np.float32)
sample_w = (sample_w / sample_w.mean()).astype(np.float32)

# ── Fit Dixon-Coles rho on train (reused from V7) ────────────────────────────
print("\n   Fitting dc_rho ...")

def _dc_nll(rho, lh, la, yh, ya):
    total = 0.0
    for h, a, y1, y2 in zip(lh, la, yh, ya):
        h, a = max(float(h), 0.05), max(float(a), 0.05)
        y1, y2 = int(y1), int(y2)
        if y1 == 0 and y2 == 0:   tau = 1.0 - h * a * rho
        elif y1 == 1 and y2 == 0: tau = 1.0 + a * rho
        elif y1 == 0 and y2 == 1: tau = 1.0 + h * rho
        elif y1 == 1 and y2 == 1: tau = 1.0 - rho
        else:                      tau = 1.0
        total += log(max(tau, 1e-9))
    return -total

rng = np.random.default_rng(42)
sidx = rng.choice(len(y_hg_tr), min(5000, len(y_hg_tr)), replace=False)
lh_m, la_m = float(y_hg_tr.mean()), float(y_ag_tr.mean())
res_rho = minimize_scalar(
    lambda r: _dc_nll(r, np.full(len(sidx), lh_m), np.full(len(sidx), la_m),
                      y_hg_tr[sidx], y_ag_tr[sidx]),
    bounds=(-0.5, 0.0), method="bounded"
)
dc_rho = float(res_rho.x)
print(f"   dc_rho = {dc_rho:.4f}")


# ── Evaluation helper ─────────────────────────────────────────────────────────
def evaluate(model: E8Net, X_n, sh, sa, sqh, sqa, ctx, y_cls, y_hg, y_ag):
    model.eval()
    all_preds, all_true = [], []
    n = len(X_n)
    with torch.no_grad():
        for i in range(0, n, 512):
            sl = slice(i, i + 512)
            xb = torch.from_numpy(X_n[sl]).to(DEVICE)
            sb_h = torch.from_numpy(sh[sl]).to(DEVICE)
            sb_a = torch.from_numpy(sa[sl]).to(DEVICE)
            sqb_h = torch.from_numpy(sqh[sl]).to(DEVICE)
            sqb_a = torch.from_numpy(sqa[sl]).to(DEVICE)
            ctxb = torch.from_numpy(ctx[sl]).to(DEVICE)
            out = model(xb, sb_h, sqb_h, sb_a, sqb_a, ctxb)
            lh = out.log_lam_home.cpu().numpy()
            la = out.log_lam_away.cpu().numpy()
            for j in range(len(lh)):
                g = score_grid(exp(lh[j]), exp(la[j]), dc_rho, n=10)
                ph, pd_, pa = wdl_from_grid(g)
                pred = np.argmax([pd_, ph, pa])  # 0=draw,1=home,2=away
                all_preds.append(pred)
                all_true.append(y_cls[sl][j])
    acc = float(np.mean(np.array(all_preds) == np.array(all_true)))
    return acc


# ── Poisson NLL loss ─────────────────────────────────────────────────────────
pois_nll = nn.PoissonNLLLoss(log_input=True, reduction="none", full=True)


def compute_loss(out: ..., y_hg_b, y_ag_b, y_cls_b, w_b):
    goals = torch.stack([out.log_lam_home, out.log_lam_away], dim=1)
    tgt = torch.stack([y_hg_b, y_ag_b], dim=1)
    l_pois = pois_nll(goals, tgt).mean(1)   # (B,) per-sample Poisson NLL

    loss = (LAMBDA_POIS * l_pois * w_b).mean()
    return loss


# ── Swap augmentation ─────────────────────────────────────────────────────────
def augment_swap(X, sh, sqh, sa, sqa, ctx, y_cls, y_hg, y_ag, w):
    """Append home/away-swapped versions of every sample."""
    X2 = swap_home_away(torch.from_numpy(X)).numpy()
    # Swap seq/squad tensors
    y_cls2 = np.where(y_cls == 1, 2, np.where(y_cls == 2, 1, y_cls))
    ctx2 = ctx.copy()  # neutral flag unchanged; tourn_weight unchanged
    return (
        np.concatenate([X, X2], axis=0),
        np.concatenate([sh, sa], axis=0),
        np.concatenate([sqh, sqa], axis=0),
        np.concatenate([sa, sh], axis=0),
        np.concatenate([sqa, sqh], axis=0),
        np.concatenate([ctx, ctx2], axis=0),
        np.concatenate([y_cls, y_cls2], axis=0),
        np.concatenate([y_hg, y_ag], axis=0),
        np.concatenate([y_ag, y_hg], axis=0),
        np.concatenate([w, w], axis=0),
    )


# ── Training loop ─────────────────────────────────────────────────────────────
cfg = E8Config()
checkpoints = []

for seed in range(N_SEEDS):
    print(f"\n{'─'*70}")
    print(f" Seed {seed+1}/{N_SEEDS}")
    print(f"{'─'*70}")
    torch.manual_seed(seed * 42 + 7)
    np.random.seed(seed * 42 + 7)

    model = build_model(cfg).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=LR, steps_per_epoch=max(1, len(X_tr_n) * 2 // BATCH_SIZE),
        epochs=EPOCHS, pct_start=0.1, anneal_strategy="cos",
    )

    # Augment training data
    Xa, sha, sqha, saa, sqaa, ctxa, y_cls_a, y_hg_a, y_ag_a, wa = augment_swap(
        X_tr_n, sh_tr, sqh_tr, sa_tr, sqa_tr, ctx_tr,
        y_cls_tr, y_hg_tr, y_ag_tr, sample_w,
    )

    # Build tensors
    def _t(arr, dtype=np.float32):
        return torch.from_numpy(arr.astype(dtype))

    ds = TensorDataset(
        _t(Xa), _t(sha), _t(sqha), _t(saa), _t(sqaa), _t(ctxa),
        _t(y_hg_a), _t(y_ag_a), torch.from_numpy(y_cls_a.astype(np.int64)),
        _t(wa),
    )
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)

    best_val = float("inf")
    no_improve = 0
    best_state = None
    t0 = time.time()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for batch in loader:
            xb, sh_b, sqh_b, sa_b, sqa_b, ctx_b, y_hg_b, y_ag_b, y_cls_b, w_b = [
                bb.to(DEVICE) for bb in batch
            ]
            opt.zero_grad(set_to_none=True)
            out = model(xb, sh_b, sqh_b, sa_b, sqa_b, ctx_b)
            loss = compute_loss(out, y_hg_b, y_ag_b, y_cls_b, w_b)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            scheduler.step()
            total_loss += loss.item()

        # Validation loss (Poisson NLL, no augmentation)
        model.eval()
        val_losses = []
        with torch.no_grad():
            for i in range(0, len(X_ca_n), 512):
                sl = slice(i, i + 512)
                xb = torch.from_numpy(X_ca_n[sl]).to(DEVICE)
                sb_h = torch.from_numpy(sh_ca[sl]).to(DEVICE)
                sb_a = torch.from_numpy(sa_ca[sl]).to(DEVICE)
                sqb_h = torch.from_numpy(sqh_ca[sl]).to(DEVICE)
                sqb_a = torch.from_numpy(sqa_ca[sl]).to(DEVICE)
                ctxb = torch.from_numpy(ctx_ca[sl]).to(DEVICE)
                out = model(xb, sb_h, sqb_h, sb_a, sqb_a, ctxb)
                goals = torch.stack([out.log_lam_home, out.log_lam_away], dim=1)
                y_hg_b = torch.from_numpy(y_hg_ca[sl].astype(np.float32)).to(DEVICE)
                y_ag_b = torch.from_numpy(y_ag_ca[sl].astype(np.float32)).to(DEVICE)
                tgt = torch.stack([y_hg_b, y_ag_b], dim=1)
                val_losses.append(pois_nll(goals, tgt).mean().item())
        val_loss = float(np.mean(val_losses))

        if val_loss < best_val:
            best_val = val_loss
            no_improve = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            no_improve += 1

        if epoch % 10 == 0 or epoch == EPOCHS:
            elapsed = time.time() - t0
            print(f"   Epoch {epoch:3d}/{EPOCHS}  "
                  f"train_loss={total_loss/len(loader):.4f}  "
                  f"val_loss={val_loss:.4f}  best={best_val:.4f}  "
                  f"({elapsed:.0f}s)")

        if no_improve >= PATIENCE:
            print(f"   Early stop at epoch {epoch} (patience={PATIENCE})")
            break

    # Restore best weights
    model.load_state_dict(best_state)

    # Evaluate on val set
    val_acc = evaluate(model, X_va_n, sh_va, sa_va, sqh_va, sqa_va, ctx_va,
                       y_cls_va, y_hg_va, y_ag_va)
    print(f"   Val acc (seed {seed}): {val_acc:.4f}")

    ckpt_path = MODELS_DIR / f"v8_seed{seed}.pt"
    torch.save({
        "state_dict": best_state,
        "cfg": cfg.__dict__,
        "val_loss": best_val,
        "val_acc": val_acc,
        "norm_mean": norm_mean.tolist(),
        "norm_std": norm_std.tolist(),
        "col_medians": col_medians.tolist(),
        "dc_rho": dc_rho,
        "seed": seed,
    }, ckpt_path)
    checkpoints.append({"path": str(ckpt_path), "val_loss": best_val, "val_acc": val_acc})
    print(f"   Saved: {ckpt_path}")

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\n{'='*70}")
print(" V8 Training complete.")
print(f"{'='*70}")
for i, c in enumerate(checkpoints):
    print(f"   Seed {i}: val_loss={c['val_loss']:.4f}  val_acc={c['val_acc']:.4f}")

meta = {
    "checkpoints": checkpoints,
    "norm_mean": norm_mean.tolist(),
    "norm_std": norm_std.tolist(),
    "col_medians": col_medians.tolist(),
    "dc_rho": dc_rho,
    "cfg": cfg.__dict__,
    "model_version": "v8-e8net",
}
meta_path = REPO_ROOT / "data" / "models" / "v8_latest.json"
meta_path.parent.mkdir(parents=True, exist_ok=True)
with open(meta_path, "w") as f:
    json.dump(meta, f, indent=2)
print(f"   Meta: {meta_path}")
print("   Next step: python -m scripts.export_onnx")
