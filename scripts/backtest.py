"""
Rolling-Origin Backtest-Harness für den WM-2026-Predictor.

Pro Fold wird mit identischer Pipeline neu trainiert und auf dem Val-Fenster
bewertet — Tendenz-Metriken (RPS, LogLoss, Brier, ECE, Accuracy), Tor-MAE und
KickTipp-Punkte/Spiel (Risiko-Bonus aus, damit quoten-unabhängig vergleichbar).

Folds (Plan §5):
  F1  Train≤2016-06   Cal 2016-07..2017-06   Val 2017-07..2018-08  (WM 2018)
  F2  Train≤2019-06   Cal 2019-07..2020-12   Val 2021              (EM20/Copa21)
  F3  Train≤2021-12   Cal 2022-01..2022-10   Val 2022-11..2023-01  (WM 2022)
  F4  Train≤2023-06   Cal 2023-07..2023-12   Val 2024-01..2026-01  (EM/Copa 24, 2025)

Modelle:
  elo           — Poisson-GLM auf elo_diff (Floor-Baseline)
  v8            — E8Net via src.train_v8_lib (HL/Fenster/Seeds parametrisierbar)
  v8-deployed   — vorhandene models/v8_seed*.pt (nur F4 leakage-frei!)

Usage:
  python -m scripts.backtest --folds F4 --models elo,v8-deployed
  python -m scripts.backtest --folds F3,F4 --models v8 --half-life 1095 --train-start 2010-01-01
  python -m scripts.backtest --folds F3,F4 --models v8 --half-life 90 365 730 1095 1825 inf \
      --train-start 2010-01-01 2015-01-01 --seeds 1 --tag hl_sweep

Ergebnisse: stdout-Tabelle + Append nach data/processed/backtest_results.jsonl
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.features_v8 import load_v8
from src.train_v8_lib import (
    DEVICE, TrainConfig, train_v8, fit_dc_rho,
    score_grid_batch, wdl_from_grid_batch, recency_weights, effective_sample_size,
)

RESULTS_JSONL = REPO_ROOT / "data" / "processed" / "backtest_results.jsonl"

FOLDS: dict[str, dict] = {
    "F1": dict(cal_start="2016-07-01", val_start="2017-07-01", val_end="2018-08-01"),
    "F2": dict(cal_start="2019-07-01", val_start="2021-01-01", val_end="2022-01-01"),
    "F3": dict(cal_start="2022-01-01", val_start="2022-11-01", val_end="2023-01-01"),
    "F4": dict(cal_start="2023-07-01", val_start="2024-01-01", val_end="2026-01-01"),
}

# ── Slices ───────────────────────────────────────────────────────────────────

_FINALS_KEYS = ("fifa world cup", "uefa euro", "copa am", "africa cup",
                "afc asian cup", "gold cup")


def slice_masks(tournaments: np.ndarray) -> dict[str, np.ndarray]:
    t = np.array([str(x).lower() for x in tournaments])
    is_qual = np.char.find(t, "qualification") >= 0
    is_final = np.zeros(len(t), dtype=bool)
    for k in _FINALS_KEYS:
        is_final |= (np.char.find(t, k) >= 0)
    is_final &= ~is_qual
    return {
        "all": np.ones(len(t), dtype=bool),
        "finals": is_final,
        "wc": (t == "fifa world cup"),
        "qual": is_qual,
        "friendly": (t == "friendly"),
    }


# ── Metriken ─────────────────────────────────────────────────────────────────

def _one_hot_ordered(y_cls: np.ndarray) -> np.ndarray:
    """y_cls (0=draw,1=home,2=away) → one-hot in Reihenfolge [home, draw, away]."""
    out = np.zeros((len(y_cls), 3))
    out[y_cls == 1, 0] = 1.0
    out[y_cls == 0, 1] = 1.0
    out[y_cls == 2, 2] = 1.0
    return out


def metrics_from_lams(log_lams: np.ndarray, dc_rho: float,
                      y_cls: np.ndarray, y_hg: np.ndarray, y_ag: np.ndarray) -> dict:
    """Alle Metriken aus (N,2) log-λ. Reihenfolge W/D/L-Vektoren: [home, draw, away]."""
    lh, la = np.exp(log_lams[:, 0]), np.exp(log_lams[:, 1])
    g = score_grid_batch(lh, la, dc_rho)          # (N, 10, 10)
    wdl = wdl_from_grid_batch(g)                   # (N, 3) [H, D, A]
    y = _one_hot_ordered(y_cls)

    # RPS (3 geordnete Outcomes H < D < A)
    cdf_p = np.cumsum(wdl, axis=1)[:, :2]
    cdf_y = np.cumsum(y, axis=1)[:, :2]
    rps = float(np.mean(np.sum((cdf_p - cdf_y) ** 2, axis=1) / 2.0))

    p_clip = np.clip(wdl, 1e-12, 1.0)
    logloss = float(-np.mean(np.sum(y * np.log(p_clip), axis=1)))
    brier = float(np.mean(np.sum((wdl - y) ** 2, axis=1)))

    pred = np.argmax(wdl, axis=1)          # 0=H,1=D,2=A
    true = np.argmax(y, axis=1)
    acc = float(np.mean(pred == true))

    # ECE (10 Bins auf max-Prob)
    conf = wdl[np.arange(len(wdl)), pred]
    correct = (pred == true).astype(float)
    ece = 0.0
    for lo in np.linspace(0.0, 0.9, 10):
        m = (conf >= lo) & (conf < lo + 0.1)
        if m.sum() > 0:
            ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
    goal_mae = float(0.5 * (np.mean(np.abs(lh - y_hg)) + np.mean(np.abs(la - y_ag))))

    # KickTipp-Punkte (Bonus aus): optimaler Tipp aus dem Grid
    tips, _ = optimal_tips_batch(g)
    pts = realized_points_batch(tips, y_hg.astype(int), y_ag.astype(int))
    return {
        "n": int(len(y_cls)), "rps": rps, "logloss": logloss, "brier": brier,
        "ece": float(ece), "acc": acc, "goal_mae": goal_mae,
        "kt_pts": float(np.mean(pts)),
    }


# ── KickTipp vektorisiert (Basis-Schema 2/3/4, Bonus aus) ────────────────────

_N = 10          # Grid-Größe (Ergebnisse)
_TMAX = 7        # Kandidaten-Tipps 0..6


def _build_tip_masks():
    """(T, N, N)-Punkte-Matrix: pts[t, ah, aa] = Punkte für Tipp t bei Ergebnis (ah, aa)."""
    hh, aa = np.meshgrid(np.arange(_N), np.arange(_N), indexing="ij")
    cell_tend = np.sign(hh - aa)                       # 1 home, 0 draw, -1 away
    cell_gd = hh - aa
    tips = [(th, ta) for th in range(_TMAX) for ta in range(_TMAX)]
    pts = np.zeros((len(tips), _N, _N))
    for i, (th, ta) in enumerate(tips):
        tend = np.sign(th - ta)
        m_tend = cell_tend == tend
        m_gd = m_tend & (cell_gd == (th - ta))
        m_exact = (hh == th) & (aa == ta)
        pts[i] = 2.0 * m_tend + 1.0 * m_gd + 1.0 * m_exact
    return np.array(tips), pts


_TIPS, _TIP_PTS = _build_tip_masks()


def optimal_tips_batch(grids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(N,10,10) Grids → (N,2) beste Tipps + (N,) erwartete Punkte (Bonus aus)."""
    ev = np.einsum("nha,tha->nt", grids, _TIP_PTS)
    best = np.argmax(ev, axis=1)
    return _TIPS[best], ev[np.arange(len(best)), best]


def realized_points_batch(tips: np.ndarray, ah: np.ndarray, aa: np.ndarray) -> np.ndarray:
    """Realisierte Punkte (2/3/4, Bonus aus) für Tipps vs. tatsächliche Ergebnisse."""
    th, ta = tips[:, 0], tips[:, 1]
    tend_tip = np.sign(th - ta)
    tend_act = np.sign(ah - aa)
    m_tend = tend_tip == tend_act
    m_gd = m_tend & ((th - ta) == (ah - aa))
    m_exact = (th == ah) & (ta == aa)
    return 2.0 * m_tend + 1.0 * m_gd + 1.0 * m_exact


# ── Modelle ──────────────────────────────────────────────────────────────────

def run_elo_baseline(data, fold: dict, train_start: str) -> dict:
    """Poisson-GLM: log λ ~ elo_diff_eff. Floor-Benchmark."""
    from sklearn.linear_model import PoissonRegressor

    dates = pd.to_datetime(data["dates"])
    m_tr = (dates >= train_start) & (dates < fold["cal_start"])
    m_va = (dates >= fold["val_start"]) & (dates < fold["val_end"])
    X = data["X"]
    elo_diff = np.nan_to_num(X[:, 4:5], nan=0.0)   # elo_diff (inkl. Heimvorteil)

    lams = []
    for tgt in (data["y_home_goals"], data["y_away_goals"]):
        reg = PoissonRegressor(alpha=1e-6, max_iter=300)
        reg.fit(elo_diff[m_tr], tgt[m_tr])
        lams.append(reg.predict(elo_diff[m_va]))
    log_lams = np.log(np.clip(np.stack(lams, axis=1), 0.05, None))
    dc_rho = fit_dc_rho(data["y_home_goals"][m_tr], data["y_away_goals"][m_tr])
    return {"val_log_lams": log_lams, "dc_rho": dc_rho, "val_mask": m_va,
            "meta": {"model": "elo", "train_start": train_start}}


def _cal_targets(data, fold: dict):
    dates = pd.to_datetime(data["dates"])
    m_ca = np.asarray((dates >= fold["cal_start"]) & (dates < fold["val_start"]))
    return (data["y_home_goals"][m_ca].astype(np.float64),
            data["y_away_goals"][m_ca].astype(np.float64))


def _maybe_calibrate(cal_log_lams, val_log_lams, data, fold, calibrate: bool, meta: dict):
    """Affine log-λ-Korrektur auf Cal fitten und auf Val anwenden."""
    if not calibrate or cal_log_lams is None or len(cal_log_lams) == 0:
        return val_log_lams
    from src.calibrate import fit_affine_lam, apply_affine_lam
    y_hg_ca, y_ag_ca = _cal_targets(data, fold)
    cal = fit_affine_lam(cal_log_lams, y_hg_ca, y_ag_ca)
    meta["cal"] = "affine"
    meta["cal_b"] = (round(cal["b_home"], 3), round(cal["b_away"], 3))
    return apply_affine_lam(val_log_lams, cal)


def run_v8(data, fold: dict, train_start: str, half_life: float | None,
           n_seeds: int, epochs: int, calibrate: bool = False) -> dict:
    cfg = TrainConfig(
        half_life_days=half_life,
        train_start=train_start,
        cal_start=fold["cal_start"],
        val_start=fold["val_start"],
        val_end=fold["val_end"],
        anchor="val-start",
        n_seeds=n_seeds,
        epochs=epochs,
        quiet=True,
    )
    res = train_v8(data, cfg)
    dates = pd.to_datetime(data["dates"])
    m_va = (dates >= fold["val_start"]) & (dates < fold["val_end"])
    hl_str = "inf" if half_life is None else f"{half_life:.0f}"
    meta = {"model": "v8", "half_life": hl_str, "train_start": train_start,
            "n_seeds": n_seeds, "ess": round(res["ess"]),
            "n_train": res["n_train"],
            "cal_loss": round(float(np.mean([s["cal_loss"] for s in res["seeds"]])), 4)}
    val_ll = _maybe_calibrate(res["cal_log_lam_ens"], res["val_log_lam_ens"],
                              data, fold, calibrate, meta)
    return {"val_log_lams": val_ll, "dc_rho": res["dc_rho"], "val_mask": m_va,
            "meta": meta,
            "cal_log_lams": res["cal_log_lam_ens"]}


# ── LightGBM-Poisson (symmetrisch: λ_home = f(X), λ_away = f(swap(X))) ───────

def _swap_static_np(X: np.ndarray) -> np.ndarray:
    from src.model_v8 import SWAP_PAIRS, DIFF_IDX
    X2 = X.copy()
    for a, b in SWAP_PAIRS:
        X2[:, a] = X[:, b]
        X2[:, b] = X[:, a]
    for d in DIFF_IDX:
        X2[:, d] = -X[:, d]
    return X2


def run_lgbm(data, fold: dict, train_start: str, half_life: float | None,
             calibrate: bool = False) -> dict:
    import lightgbm as lgb

    dates = pd.to_datetime(data["dates"])
    m_tr = np.asarray((dates >= train_start) & (dates < fold["cal_start"]))
    m_ca = np.asarray((dates >= fold["cal_start"]) & (dates < fold["val_start"]))
    m_va = np.asarray((dates >= fold["val_start"]) & (dates < fold["val_end"]))

    X = data["X"].astype(np.float64)   # NaNs bleiben drin — LightGBM kann das nativ
    y_hg, y_ag = data["y_home_goals"].astype(np.float64), data["y_away_goals"].astype(np.float64)

    anchor = pd.Timestamp(fold["val_start"])
    w = recency_weights(pd.DatetimeIndex(dates[m_tr]), half_life, anchor)
    ess = effective_sample_size(w)
    w = w / w.mean()

    # Symmetrisches Training: eine Maschine für "Tore des a-Teams"
    X_tr = np.concatenate([X[m_tr], _swap_static_np(X[m_tr])], axis=0)
    y_tr = np.concatenate([y_hg[m_tr], y_ag[m_tr]])
    w_tr = np.concatenate([w, w])
    X_ca = np.concatenate([X[m_ca], _swap_static_np(X[m_ca])], axis=0)
    y_ca = np.concatenate([y_hg[m_ca], y_ag[m_ca]])

    booster = lgb.LGBMRegressor(
        objective="poisson", n_estimators=2000, learning_rate=0.03,
        num_leaves=31, min_child_samples=40, subsample=0.8, subsample_freq=1,
        colsample_bytree=0.8, reg_lambda=1.0, verbose=-1,
    )
    booster.fit(X_tr, y_tr, sample_weight=w_tr,
                eval_set=[(X_ca, y_ca)], eval_metric="poisson",
                callbacks=[lgb.early_stopping(100, verbose=False)])

    def _log_lams(mask):
        lh = np.clip(booster.predict(X[mask]), 0.05, None)
        la = np.clip(booster.predict(_swap_static_np(X[mask])), 0.05, None)
        return np.log(np.stack([lh, la], axis=1))

    dc_rho = fit_dc_rho(y_hg[m_tr], y_ag[m_tr])
    hl_str = "inf" if half_life is None else f"{half_life:.0f}"
    meta = {"model": "lgbm", "half_life": hl_str, "train_start": train_start,
            "ess": round(ess), "n_trees": booster.best_iteration_ or booster.n_estimators}
    cal_ll = _log_lams(m_ca)
    val_ll = _maybe_calibrate(cal_ll, _log_lams(m_va), data, fold, calibrate, meta)
    return {"val_log_lams": val_ll, "dc_rho": dc_rho, "val_mask": m_va,
            "meta": meta, "cal_log_lams": cal_ll}


def run_ensemble(data, fold: dict, train_start: str, half_life: float | None,
                 n_seeds: int, epochs: int, calibrate: bool = False,
                 ens_weight: float | None = None) -> dict:
    """v8 + lgbm: gewichtetes Mittel der log-λs.

    ens_weight=None → Gewicht auf Cal gefittet (kann zwischen Folds instabil
    sein); fester Wert (z.B. 0.5) ist die robuste Wahl.
    """
    from scipy.optimize import minimize_scalar

    r_v8 = run_v8(data, fold, train_start, half_life, n_seeds, epochs, calibrate=False)
    r_lg = run_lgbm(data, fold, train_start, half_life, calibrate=False)
    y_hg_ca, y_ag_ca = _cal_targets(data, fold)
    y_ca = np.stack([y_hg_ca, y_ag_ca], axis=1)

    def _nll(wt):
        ll = wt * r_v8["cal_log_lams"] + (1 - wt) * r_lg["cal_log_lams"]
        lam = np.exp(ll)
        return float(np.mean(lam - y_ca * ll))

    if ens_weight is not None:
        wt = float(ens_weight)
    else:
        res = minimize_scalar(_nll, bounds=(0.0, 1.0), method="bounded")
        wt = float(res.x)
    cal_ll = wt * r_v8["cal_log_lams"] + (1 - wt) * r_lg["cal_log_lams"]
    val_ll = wt * r_v8["val_log_lams"] + (1 - wt) * r_lg["val_log_lams"]
    dc_rho = 0.5 * (r_v8["dc_rho"] + r_lg["dc_rho"])
    hl_str = "inf" if half_life is None else f"{half_life:.0f}"
    meta = {"model": "ens", "half_life": hl_str, "train_start": train_start,
            "w_v8": round(wt, 3),
            "ess": r_lg["meta"]["ess"], "n_seeds": n_seeds}
    val_ll = _maybe_calibrate(cal_ll, val_ll, data, fold, calibrate, meta)
    return {"val_log_lams": val_ll, "dc_rho": dc_rho,
            "val_mask": r_v8["val_mask"], "meta": meta, "cal_log_lams": cal_ll}


def run_v8_deployed(data, fold: dict) -> dict:
    """Bestehende models/v8_seed*.pt auf dem Fold-Val auswerten (nur F4 leakage-frei)."""
    import torch
    from src.model_v8 import E8Config, build_model
    from src.train_v8_lib import _forward_log_lams, _impute, _context

    dates = pd.to_datetime(data["dates"])
    m_va = np.asarray((dates >= fold["val_start"]) & (dates < fold["val_end"]))
    X_va = data["X"][m_va]
    sh, sa = data["seq_home"][m_va], data["seq_away"][m_va]
    sqh, sqa = data["squad_home"][m_va], data["squad_away"][m_va]

    lams, rhos = [], []
    for seed in range(3):
        p = REPO_ROOT / "models" / f"v8_seed{seed}.pt"
        if not p.exists():
            continue
        ck = torch.load(p, map_location=DEVICE, weights_only=False)
        model = build_model(E8Config(**ck["cfg"])).to(DEVICE)
        model.load_state_dict(ck["state_dict"])
        med = np.array(ck["col_medians"], dtype=np.float64)
        mean = np.array(ck["norm_mean"], dtype=np.float32)
        std = np.array(ck["norm_std"], dtype=np.float32)
        X_n = ((_impute(X_va, med) - mean) / std).astype(np.float32)
        lams.append(_forward_log_lams(model, X_n, sh, sa, sqh, sqa, _context(X_n)))
        rhos.append(float(ck["dc_rho"]))
    if not lams:
        raise FileNotFoundError("Keine v8_seed*.pt Checkpoints gefunden.")
    return {"val_log_lams": np.mean(np.stack(lams), axis=0),
            "dc_rho": float(np.mean(rhos)), "val_mask": m_va,
            "meta": {"model": "v8-deployed", "n_ckpts": len(lams)}}


# ── Hauptablauf ──────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(description="Rolling-Origin Backtest")
    ap.add_argument("--folds", type=str, default="F4")
    ap.add_argument("--models", type=str, default="elo,v8-deployed")
    ap.add_argument("--half-life", type=str, nargs="+", default=["90"],
                    help="Tage oder 'inf'; mehrere Werte = Sweep")
    ap.add_argument("--train-start", type=str, nargs="+", default=["2015-01-01"])
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--calibrate", action="store_true",
                    help="affine log-λ-Korrektur (auf Cal gefittet) anwenden")
    ap.add_argument("--ens-weight", type=float, default=None,
                    help="festes w_v8 für --models ens (default: auf Cal gefittet)")
    ap.add_argument("--tag", type=str, default="")
    args = ap.parse_args()

    folds = [f.strip().upper() for f in args.folds.split(",")]
    models = [m.strip() for m in args.models.split(",")]
    hls = [None if h.lower() in ("inf", "none") else float(h) for h in args.half_life]

    print(f"Loading features_v8 ... (device: {DEVICE})")
    data = load_v8()
    dates = pd.to_datetime(data["dates"])

    # Tournament-Spalte alignen (results.csv → npz-Zeilen via (date, home, away))
    df = pd.read_csv(REPO_ROOT / "data" / "raw" / "results.csv", parse_dates=["date"])
    df = df.dropna(subset=["home_score", "away_score"]).sort_values("date").reset_index(drop=True)
    key2t = {(d.strftime("%Y-%m-%d"), h, a): t for d, h, a, t in
             zip(df["date"], df["home_team"], df["away_team"], df["tournament"])}
    tournaments = np.array([
        key2t.get((pd.Timestamp(dates[i]).strftime("%Y-%m-%d"),
                   str(data["home"][i]), str(data["away"][i])), "?")
        for i in range(len(dates))
    ])

    rows = []
    t_start = time.time()
    for fold_name in folds:
        fold = FOLDS[fold_name]
        for model in models:
            if model == "elo":
                combos = [(args.train_start[0], None)]
            elif model in ("v8", "lgbm", "ens"):
                combos = list(itertools.product(args.train_start, hls))
            elif model == "v8-deployed":
                combos = [(None, None)]
                if fold["val_start"] < "2024-01-01":
                    print(f"  !! {fold_name}/v8-deployed: Val < 2024 → Leakage, übersprungen.")
                    continue
            else:
                print(f"  ?? Unbekanntes Modell: {model}");  continue

            for train_start, hl in combos:
                t0 = time.time()
                if model == "elo":
                    r = run_elo_baseline(data, fold, train_start)
                elif model == "v8":
                    r = run_v8(data, fold, train_start, hl, args.seeds, args.epochs,
                               calibrate=args.calibrate)
                elif model == "lgbm":
                    r = run_lgbm(data, fold, train_start, hl, calibrate=args.calibrate)
                elif model == "ens":
                    r = run_ensemble(data, fold, train_start, hl, args.seeds,
                                     args.epochs, calibrate=args.calibrate,
                                     ens_weight=args.ens_weight)
                else:
                    r = run_v8_deployed(data, fold)

                m_va = np.asarray(r["val_mask"])
                y_cls = data["y"][m_va]
                y_hg = data["y_home_goals"][m_va].astype(np.float64)
                y_ag = data["y_away_goals"][m_va].astype(np.float64)
                sl = slice_masks(tournaments[m_va])

                rec = {"ts": datetime.now().isoformat(timespec="seconds"),
                       "tag": args.tag, "fold": fold_name, **r["meta"],
                       "secs": round(time.time() - t0, 1)}
                for sname, smask in sl.items():
                    if smask.sum() == 0:
                        continue
                    rec[sname] = metrics_from_lams(
                        r["val_log_lams"][smask], r["dc_rho"],
                        y_cls[smask], y_hg[smask], y_ag[smask])
                rows.append(rec)

                a, f_ = rec.get("all", {}), rec.get("finals", {})
                label = r["meta"].get("model", model)
                extra = "  ".join(f"{k}={v}" for k, v in r["meta"].items()
                                  if k not in ("model",))
                print(f"  {fold_name} {label:12s} {extra}")
                print(f"      all    n={a.get('n', 0):5d}  rps={a.get('rps', 0):.4f}  "
                      f"ll={a.get('logloss', 0):.4f}  acc={a.get('acc', 0):.3f}  "
                      f"ece={a.get('ece', 0):.3f}  kt={a.get('kt_pts', 0):.3f}")
                if f_:
                    print(f"      finals n={f_['n']:5d}  rps={f_['rps']:.4f}  "
                          f"ll={f_['logloss']:.4f}  acc={f_['acc']:.3f}  "
                          f"ece={f_['ece']:.3f}  kt={f_['kt_pts']:.3f}")

    RESULTS_JSONL.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_JSONL, "a", encoding="utf-8") as fh:
        for rec in rows:
            fh.write(json.dumps(rec) + "\n")
    print(f"\n{len(rows)} Läufe in {time.time() - t_start:.0f}s → {RESULTS_JSONL}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
