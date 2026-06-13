"""
V8 Training-Library — parametrisiertes E8Net-Training für Sweeps & Backtests.

Extrahiert aus scripts/train_v8.py (das jetzt nur noch CLI-Wrapper ist).
Kernfunktion `train_v8(data, cfg)` trainiert N Seeds auf einem Zeitfenster
und gibt rohe λ-Vorhersagen (Cal + Val) zurück, damit Metriken/Kalibrierung
zentral im Backtest-Harness berechnet werden können.

Neu gegenüber dem alten Skript:
  - half_life / train_start / val_end / anchor / seeds als Parameter
  - ESS-Logging: ESS = (Σw)² / Σw²  — effektive Stichprobengröße der
    Recency-Gewichte (HL=90d ⇒ ESS≈600 von ~7k nominal; das war der Bug)
  - Vektorisiertes Score-Grid (score_grid_batch) statt Python-Schleife

Hinweis Anchor: Bei rein exponentiellem Decay + Mean-Normalisierung ist der
Anchor mathematisch irrelevant (konstanter Faktor kürzt sich weg). Der
Parameter existiert nur zur Dokumentation der Intention.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from scipy.optimize import minimize_scalar

from .model_v8 import E8Net, E8Config, build_model, swap_home_away

REPO_ROOT = Path(__file__).resolve().parent.parent
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Config ───────────────────────────────────────────────────────────────────

@dataclass
class TrainConfig:
    half_life_days: float | None = 90.0   # None/inf = keine Recency-Gewichtung
    train_start: str = "2015-01-01"
    cal_start: str = "2023-01-01"          # Train = [train_start, cal_start)
    val_start: str = "2024-01-01"          # Cal   = [cal_start, val_start)
    val_end: str | None = None             # Val   = [val_start, val_end)
    anchor: str = "max-train"              # "max-train" | "val-start" | ISO-Datum
    n_seeds: int = 3
    epochs: int = 150
    patience: int = 20
    lr: float = 3e-4
    batch_size: int = 512
    lambda_pois: float = 1.0
    model_cfg: E8Config = field(default_factory=E8Config)
    save_dir: str | None = None            # z.B. "models" → v8_seed{N}.pt
    tag: str = ""                          # Suffix für Checkpoint-Dateien
    keep_states: bool = False              # state_dicts im Ergebnis behalten
    quiet: bool = False


# ── Recency-Gewichte + ESS ───────────────────────────────────────────────────

def recency_weights(
    dates: pd.DatetimeIndex | pd.Series,
    half_life_days: float | None,
    anchor: pd.Timestamp,
) -> np.ndarray:
    """w = exp(-ln2 · (anchor - date) / HL), unnormalisiert."""
    d = pd.DatetimeIndex(dates)
    if half_life_days is None or not math.isfinite(half_life_days):
        return np.ones(len(d), dtype=np.float64)
    days_old = np.asarray((anchor - d).days, dtype=np.float64)
    return np.exp(-days_old * math.log(2.0) / float(half_life_days))


def effective_sample_size(w: np.ndarray) -> float:
    """ESS = (Σw)² / Σw² — skaleninvariant."""
    s = float(w.sum())
    s2 = float((w * w).sum())
    return s * s / s2 if s2 > 0 else 0.0


# ── Vektorisiertes Score-Grid (Konventionen wie features_v6.score_grid) ─────

_K = 10
_LOGFAC = np.zeros(_K, dtype=np.float64)
for _i in range(2, _K):
    _LOGFAC[_i] = _LOGFAC[_i - 1] + np.log(_i)


def score_grid_batch(lh: np.ndarray, la: np.ndarray, rho: float, n: int = _K) -> np.ndarray:
    """(N,) λ-Arrays → (N, n, n) Grids; P[i, h, a]. Dixon-Coles-Korrektur wie score_grid."""
    lh = np.maximum(np.asarray(lh, dtype=np.float64), 0.05)
    la = np.maximum(np.asarray(la, dtype=np.float64), 0.05)
    k = np.arange(n, dtype=np.float64)
    logfac = _LOGFAC[:n] if n <= _K else None
    if logfac is None:
        logfac = np.zeros(n)
        for i in range(2, n):
            logfac[i] = logfac[i - 1] + np.log(i)
    ph = np.exp(-lh[:, None] + k[None, :] * np.log(lh[:, None]) - logfac[None, :])
    pa = np.exp(-la[:, None] + k[None, :] * np.log(la[:, None]) - logfac[None, :])
    g = ph[:, :, None] * pa[:, None, :]
    g[:, 0, 0] *= np.maximum(1.0 - lh * la * rho, 1e-9)
    g[:, 1, 0] *= np.maximum(1.0 + la * rho, 1e-9)
    g[:, 0, 1] *= np.maximum(1.0 + lh * rho, 1e-9)
    g[:, 1, 1] *= np.maximum(1.0 - rho, 1e-9)
    g /= g.sum(axis=(1, 2), keepdims=True)
    return g


def wdl_from_grid_batch(g: np.ndarray) -> np.ndarray:
    """(N, n, n) Grids → (N, 3) [p_home, p_draw, p_away], normalisiert."""
    n = g.shape[1]
    h_idx, a_idx = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    p_home = g[:, h_idx > a_idx].sum(axis=1)
    p_draw = g[:, h_idx == a_idx].sum(axis=1)
    p_away = g[:, h_idx < a_idx].sum(axis=1)
    out = np.stack([p_home, p_draw, p_away], axis=1)
    out /= out.sum(axis=1, keepdims=True)
    return out


# ── Dixon-Coles rho-Fit (aus scripts/train_v8.py übernommen) ────────────────

def fit_dc_rho(y_hg: np.ndarray, y_ag: np.ndarray, n_sample: int = 5000, seed: int = 42) -> float:
    def _dc_nll(rho, lh, la, yh, ya):
        total = 0.0
        for h, a, y1, y2 in zip(lh, la, yh, ya):
            h, a = max(float(h), 0.05), max(float(a), 0.05)
            y1, y2 = int(y1), int(y2)
            if y1 == 0 and y2 == 0:
                tau = 1.0 - h * a * rho
            elif y1 == 1 and y2 == 0:
                tau = 1.0 + a * rho
            elif y1 == 0 and y2 == 1:
                tau = 1.0 + h * rho
            elif y1 == 1 and y2 == 1:
                tau = 1.0 - rho
            else:
                tau = 1.0
            total += math.log(max(tau, 1e-9))
        return -total

    rng = np.random.default_rng(seed)
    sidx = rng.choice(len(y_hg), min(n_sample, len(y_hg)), replace=False)
    lh_m, la_m = float(y_hg.mean()), float(y_ag.mean())
    res = minimize_scalar(
        lambda r: _dc_nll(r, np.full(len(sidx), lh_m), np.full(len(sidx), la_m),
                          y_hg[sidx], y_ag[sidx]),
        bounds=(-0.5, 0.0), method="bounded",
    )
    return float(res.x)


# ── Hilfsfunktionen ──────────────────────────────────────────────────────────

def _normalize(X_tr: np.ndarray):
    mean = X_tr.mean(axis=0)
    std = X_tr.std(axis=0) + 1e-6
    return mean.astype(np.float32), std.astype(np.float32)


def _impute(X: np.ndarray, medians: np.ndarray) -> np.ndarray:
    out = X.copy()
    nans = np.isnan(out)
    out[nans] = np.take(medians, np.where(nans)[1])
    return out


def _context(X_n: np.ndarray) -> np.ndarray:
    return X_n[:, [0, 1]].astype(np.float32)  # neutral, tournament_w


def _augment_swap(X, sh, sqh, sa, sqa, ctx, y_cls, y_hg, y_ag, w):
    """Home/Away-Swap-Augmentierung (verdoppelt Trainingsdaten)."""
    X2 = swap_home_away(torch.from_numpy(X)).numpy()
    y_cls2 = np.where(y_cls == 1, 2, np.where(y_cls == 2, 1, y_cls))
    return (
        np.concatenate([X, X2], axis=0),
        np.concatenate([sh, sa], axis=0),
        np.concatenate([sqh, sqa], axis=0),
        np.concatenate([sa, sh], axis=0),
        np.concatenate([sqa, sqh], axis=0),
        np.concatenate([ctx, ctx.copy()], axis=0),
        np.concatenate([y_cls, y_cls2], axis=0),
        np.concatenate([y_hg, y_ag], axis=0),
        np.concatenate([y_ag, y_hg], axis=0),
        np.concatenate([w, w], axis=0),
    )


def _forward_log_lams(model: E8Net, X_n, sh, sa, sqh, sqa, ctx, batch: int = 1024) -> np.ndarray:
    """Modell-Forward in Batches → (N, 2) [log_lam_home, log_lam_away]."""
    model.eval()
    outs = []
    n = len(X_n)
    with torch.no_grad():
        for i in range(0, n, batch):
            sl = slice(i, i + batch)
            out = model(
                torch.from_numpy(X_n[sl]).to(DEVICE),
                torch.from_numpy(sh[sl]).to(DEVICE),
                torch.from_numpy(sqh[sl]).to(DEVICE),
                torch.from_numpy(sa[sl]).to(DEVICE),
                torch.from_numpy(sqa[sl]).to(DEVICE),
                torch.from_numpy(ctx[sl]).to(DEVICE),
            )
            outs.append(np.stack([out.log_lam_home.cpu().numpy(),
                                  out.log_lam_away.cpu().numpy()], axis=1))
    return np.concatenate(outs, axis=0) if outs else np.zeros((0, 2), dtype=np.float32)


def accuracy_from_log_lams(log_lams: np.ndarray, y_cls: np.ndarray, dc_rho: float) -> float:
    """Argmax-Tendenz-Accuracy aus λs via Grid (0=draw, 1=home, 2=away)."""
    if len(log_lams) == 0:
        return float("nan")
    g = score_grid_batch(np.exp(log_lams[:, 0]), np.exp(log_lams[:, 1]), dc_rho)
    wdl = wdl_from_grid_batch(g)  # [home, draw, away]
    pred = np.argmax(wdl[:, [1, 0, 2]], axis=1)  # → 0=draw, 1=home, 2=away
    return float(np.mean(pred == y_cls))


# ── Haupt-Trainingsfunktion ──────────────────────────────────────────────────

def train_v8(data: dict, cfg: TrainConfig) -> dict:
    """Trainiert E8Net auf dem konfigurierten Zeitfenster.

    data: dict aus src.features_v8.load_v8() (X, y, y_home_goals, ..., seq_*, squad_*)
    """
    log = (lambda *a, **k: None) if cfg.quiet else print

    dates_all = pd.to_datetime(data["dates"])
    X_all = data["X"]
    y_cls_all = data["y"]
    y_hg_all = data["y_home_goals"].astype(np.float32)
    y_ag_all = data["y_away_goals"].astype(np.float32)
    sh_all, sa_all = data["seq_home"], data["seq_away"]
    sqh_all, sqa_all = data["squad_home"], data["squad_away"]

    # ── Splits ───────────────────────────────────────────────────────────────
    m_tr = (dates_all >= cfg.train_start) & (dates_all < cfg.cal_start)
    m_ca = (dates_all >= cfg.cal_start) & (dates_all < cfg.val_start)
    m_va = dates_all >= cfg.val_start
    if cfg.val_end:
        m_va &= dates_all < cfg.val_end

    n_tr, n_ca, n_va = int(m_tr.sum()), int(m_ca.sum()), int(m_va.sum())
    log(f"   Train: {n_tr:,}  Cal: {n_ca:,}  Val: {n_va:,}")
    if n_tr < 100 or n_ca < 50:
        raise ValueError(f"Zu wenig Daten: train={n_tr}, cal={n_ca}")

    def _s(arr):
        return arr[np.asarray(m_tr)], arr[np.asarray(m_ca)], arr[np.asarray(m_va)]

    X_tr_raw, X_ca_raw, X_va_raw = _s(X_all)
    y_cls_tr, y_cls_ca, y_cls_va = _s(y_cls_all)
    y_hg_tr, y_hg_ca, y_hg_va = _s(y_hg_all)
    y_ag_tr, y_ag_ca, y_ag_va = _s(y_ag_all)
    sh_tr, sh_ca, sh_va = _s(sh_all)
    sa_tr, sa_ca, sa_va = _s(sa_all)
    sqh_tr, sqh_ca, sqh_va = _s(sqh_all)
    sqa_tr, sqa_ca, sqa_va = _s(sqa_all)
    dates_tr = dates_all[m_tr]

    # ── Imputation + Normalisierung (Train-Statistiken) ─────────────────────
    col_medians = np.nanmedian(X_tr_raw, axis=0)
    X_tr_raw = _impute(X_tr_raw, col_medians)
    X_ca_raw = _impute(X_ca_raw, col_medians)
    X_va_raw = _impute(X_va_raw, col_medians)
    norm_mean, norm_std = _normalize(X_tr_raw)
    X_tr_n = ((X_tr_raw - norm_mean) / norm_std).astype(np.float32)
    X_ca_n = ((X_ca_raw - norm_mean) / norm_std).astype(np.float32)
    X_va_n = ((X_va_raw - norm_mean) / norm_std).astype(np.float32)
    ctx_tr, ctx_ca, ctx_va = _context(X_tr_n), _context(X_ca_n), _context(X_va_n)

    # ── Recency-Gewichte + ESS ───────────────────────────────────────────────
    if cfg.anchor == "max-train":
        anchor = pd.Timestamp(dates_tr.max())
    elif cfg.anchor == "val-start":
        anchor = pd.Timestamp(cfg.val_start)
    else:
        anchor = pd.Timestamp(cfg.anchor)
    w_raw = recency_weights(dates_tr, cfg.half_life_days, anchor)
    ess = effective_sample_size(w_raw)
    sample_w = (w_raw / w_raw.mean()).astype(np.float32)
    hl_str = "inf" if cfg.half_life_days is None or not math.isfinite(cfg.half_life_days or float("inf")) \
        else f"{cfg.half_life_days:.0f}d"
    log(f"   Recency: HL={hl_str}  anchor={anchor.date()}  ESS={ess:,.0f}/{n_tr:,}")

    # ── dc_rho auf Train fitten ──────────────────────────────────────────────
    dc_rho = fit_dc_rho(y_hg_tr, y_ag_tr)
    log(f"   dc_rho = {dc_rho:.4f}")

    pois_nll = nn.PoissonNLLLoss(log_input=True, reduction="none", full=False)

    # ── Seeds trainieren ─────────────────────────────────────────────────────
    seed_results = []
    cal_log_lams, val_log_lams = [], []
    states = []

    for seed in range(cfg.n_seeds):
        torch.manual_seed(seed * 42 + 7)
        np.random.seed(seed * 42 + 7)

        model = build_model(cfg.model_cfg).to(DEVICE)
        opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)

        Xa, sha, sqha, saa, sqaa, ctxa, y_cls_a, y_hg_a, y_ag_a, wa = _augment_swap(
            X_tr_n, sh_tr, sqh_tr, sa_tr, sqa_tr, ctx_tr,
            y_cls_tr, y_hg_tr, y_ag_tr, sample_w,
        )

        def _t(arr, dtype=np.float32):
            return torch.from_numpy(arr.astype(dtype))

        ds = TensorDataset(
            _t(Xa), _t(sha), _t(sqha), _t(saa), _t(sqaa), _t(ctxa),
            _t(y_hg_a), _t(y_ag_a), torch.from_numpy(y_cls_a.astype(np.int64)), _t(wa),
        )
        loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True,
                            num_workers=0, pin_memory=True)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=cfg.lr, steps_per_epoch=len(loader),
            epochs=cfg.epochs, pct_start=0.1, anneal_strategy="cos",
        )

        best_val = float("inf")
        no_improve = 0
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        best_epoch = 0
        t0 = time.time()

        for epoch in range(1, cfg.epochs + 1):
            model.train()
            total_loss = 0.0
            for batch in loader:
                xb, sh_b, sqh_b, sa_b, sqa_b, ctx_b, y_hg_b, y_ag_b, y_cls_b, w_b = [
                    bb.to(DEVICE) for bb in batch
                ]
                opt.zero_grad(set_to_none=True)
                out = model(xb, sh_b, sqh_b, sa_b, sqa_b, ctx_b)
                goals = torch.stack([out.log_lam_home, out.log_lam_away], dim=1)
                tgt = torch.stack([y_hg_b, y_ag_b], dim=1)
                l_pois = pois_nll(goals, tgt).mean(1)
                loss = (cfg.lambda_pois * l_pois * w_b).mean()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                scheduler.step()
                total_loss += loss.item()

            # Cal-Loss (ungewichtete Poisson-NLL) für Early Stopping
            cal_ll = _forward_log_lams(model, X_ca_n, sh_ca, sa_ca, sqh_ca, sqa_ca, ctx_ca)
            with torch.no_grad():
                goals = torch.from_numpy(cal_ll)
                tgt = torch.from_numpy(np.stack([y_hg_ca, y_ag_ca], axis=1))
                val_loss = float(pois_nll(goals, tgt).mean().item())

            if val_loss < best_val and val_loss == val_loss:
                best_val = val_loss
                best_epoch = epoch
                no_improve = 0
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                no_improve += 1

            if not cfg.quiet and (epoch % 10 == 0 or epoch == cfg.epochs):
                log(f"   Seed {seed} Ep {epoch:3d}/{cfg.epochs}  "
                    f"train={total_loss / len(loader):.4f}  cal={val_loss:.4f}  "
                    f"best={best_val:.4f}@{best_epoch}  ({time.time() - t0:.0f}s)")

            if no_improve >= cfg.patience:
                log(f"   Seed {seed}: Early stop Ep {epoch} (best Ep {best_epoch})")
                break

        model.load_state_dict(best_state)
        cal_ll = _forward_log_lams(model, X_ca_n, sh_ca, sa_ca, sqh_ca, sqa_ca, ctx_ca)
        val_ll = _forward_log_lams(model, X_va_n, sh_va, sa_va, sqh_va, sqa_va, ctx_va)
        cal_log_lams.append(cal_ll)
        val_log_lams.append(val_ll)

        val_acc = accuracy_from_log_lams(val_ll, y_cls_va, dc_rho) if n_va else float("nan")
        log(f"   Seed {seed}: cal_loss={best_val:.4f}  val_acc={val_acc:.4f}  ({time.time() - t0:.0f}s)")
        seed_results.append({"seed": seed, "cal_loss": best_val,
                             "best_epoch": best_epoch, "val_acc": val_acc})

        if cfg.save_dir:
            save_dir = Path(cfg.save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            suffix = f"_{cfg.tag}" if cfg.tag else ""
            ckpt_path = save_dir / f"v8_seed{seed}{suffix}.pt"
            torch.save({
                "state_dict": best_state,
                "cfg": cfg.model_cfg.__dict__,
                "val_loss": best_val,
                "val_acc": val_acc,
                "norm_mean": norm_mean.tolist(),
                "norm_std": norm_std.tolist(),
                "col_medians": col_medians.tolist(),
                "dc_rho": dc_rho,
                "seed": seed,
                "train_cfg": {**asdict(cfg), "model_cfg": cfg.model_cfg.__dict__},
            }, ckpt_path)
            seed_results[-1]["path"] = str(ckpt_path)
            log(f"   Saved: {ckpt_path}")
        if cfg.keep_states:
            states.append(best_state)

    # ── Ensemble-λ (Mittel über Seeds im log-Raum) ───────────────────────────
    cal_ll_ens = np.mean(np.stack(cal_log_lams), axis=0) if cal_log_lams else None
    val_ll_ens = np.mean(np.stack(val_log_lams), axis=0) if val_log_lams else None
    ens_acc = accuracy_from_log_lams(val_ll_ens, y_cls_va, dc_rho) if n_va else float("nan")
    log(f"   Ensemble val_acc: {ens_acc:.4f}")

    return {
        "cfg": {**asdict(cfg), "model_cfg": cfg.model_cfg.__dict__},
        "ess": ess,
        "n_train": n_tr, "n_cal": n_ca, "n_val": n_va,
        "dc_rho": dc_rho,
        "norm_mean": norm_mean, "norm_std": norm_std, "col_medians": col_medians,
        "seeds": seed_results,
        "ensemble_val_acc": ens_acc,
        "cal_log_lams": np.stack(cal_log_lams) if cal_log_lams else None,   # (S, Nc, 2)
        "val_log_lams": np.stack(val_log_lams) if val_log_lams else None,   # (S, Nv, 2)
        "cal_log_lam_ens": cal_ll_ens,  # (Nc, 2)
        "val_log_lam_ens": val_ll_ens,  # (Nv, 2)
        "cal_idx": np.where(np.asarray(m_ca))[0],
        "val_idx": np.where(np.asarray(m_va))[0],
        "states": states,
    }
