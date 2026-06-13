"""
V9-Final-Training für den Tournament-Einsatz.

Trainiert das komplette Sieger-Rezept aus dem Backtest auf allen Daten bis
heute und schreibt deploybare Artefakte:

  1. E8Net, 3 Seeds   (HL/Fenster aus Sweep, Anker = Turnierstart)
  2. LightGBM-Poisson (symmetrisch, gleiche Gewichte)
  3. Ensemble-Gewicht w_v8 (Cal-NLL-optimal)
  4. Affine λ-Kalibrierung auf dem geblendeten Cal-Split

Output (tag-geschützt, deployte v8_seed{N}.pt bleiben unberührt):
  models/v8_seed{0,1,2}_{TAG}.pt
  models/lgbm_{TAG}.txt
  data/models/v8_latest_{TAG}.json   (inkl. "lgbm" + "lam_cal" für predict_v8)

Usage:
  .\\venv\\Scripts\\python.exe -m scripts.train_v9_final --half-life 1095 \\
      --train-start 2010-01-01 --tag v9
Danach:
  .\\venv\\Scripts\\python.exe -m scripts.make_tips --model v8 --tag v9
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.features_v8 import load_v8
from src.train_v8_lib import (
    DEVICE, TrainConfig, train_v8, recency_weights, effective_sample_size,
)
from src.calibrate import fit_affine_lam


def main() -> int:
    ap = argparse.ArgumentParser(description="V9-Final-Training (E8Net + LGBM + Kalibrierung)")
    ap.add_argument("--half-life", type=float, default=1095.0)
    ap.add_argument("--train-start", type=str, default="2010-01-01")
    ap.add_argument("--cal-start", type=str, default="2025-06-15",
                    help="Cal = [cal-start, val-start); für Early-Stopping/Kalibrierung")
    ap.add_argument("--val-start", type=str, default="2026-06-10")
    ap.add_argument("--anchor", type=str, default="2026-06-11", help="WM-Eröffnung")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--ens-weight", type=float, default=0.5,
                    help="festes w_v8 (NaN = auf Cal fitten); 0.5 = Gate-Sieger")
    ap.add_argument("--tag", type=str, default="v9")
    args = ap.parse_args()

    print("=" * 70)
    print(f" V9 Final Training  (tag={args.tag}, device={DEVICE})")
    print("=" * 70)

    data = load_v8()
    dates = pd.to_datetime(data["dates"])

    # ── 1) E8Net-Seeds ───────────────────────────────────────────────────────
    cfg = TrainConfig(
        half_life_days=args.half_life,
        train_start=args.train_start,
        cal_start=args.cal_start,
        val_start=args.val_start,
        anchor=args.anchor,
        n_seeds=args.seeds,
        save_dir="models",
        tag=args.tag,
    )
    res = train_v8(data, cfg)

    # ── 2) LightGBM (gleiche Fenster + Gewichte) ─────────────────────────────
    import lightgbm as lgb
    from scripts.backtest import _swap_static_np

    m_tr = np.asarray((dates >= args.train_start) & (dates < args.cal_start))
    m_ca = np.asarray((dates >= args.cal_start) & (dates < args.val_start))
    X = data["X"].astype(np.float64)
    y_hg = data["y_home_goals"].astype(np.float64)
    y_ag = data["y_away_goals"].astype(np.float64)

    w = recency_weights(pd.DatetimeIndex(dates[m_tr]), args.half_life,
                        pd.Timestamp(args.anchor))
    print(f"   LGBM: ESS={effective_sample_size(w):,.0f}/{int(m_tr.sum()):,}")
    w = w / w.mean()

    X_tr = np.concatenate([X[m_tr], _swap_static_np(X[m_tr])], axis=0)
    y_tr = np.concatenate([y_hg[m_tr], y_ag[m_tr]])
    w_tr = np.concatenate([w, w])
    X_ca2 = np.concatenate([X[m_ca], _swap_static_np(X[m_ca])], axis=0)
    y_ca2 = np.concatenate([y_hg[m_ca], y_ag[m_ca]])

    booster = lgb.LGBMRegressor(
        objective="poisson", n_estimators=2000, learning_rate=0.03,
        num_leaves=31, min_child_samples=40, subsample=0.8, subsample_freq=1,
        colsample_bytree=0.8, reg_lambda=1.0, verbose=-1,
    )
    booster.fit(X_tr, y_tr, sample_weight=w_tr,
                eval_set=[(X_ca2, y_ca2)], eval_metric="poisson",
                callbacks=[lgb.early_stopping(100, verbose=False)])
    print(f"   LGBM: {booster.best_iteration_} Bäume")

    lgbm_path = REPO_ROOT / "models" / f"lgbm_{args.tag}.txt"
    booster.booster_.save_model(str(lgbm_path))

    # ── 3) Ensemble-Gewicht auf Cal ──────────────────────────────────────────
    from scipy.optimize import minimize_scalar

    lgb_cal = np.log(np.clip(np.stack([
        booster.predict(X[m_ca]),
        booster.predict(_swap_static_np(X[m_ca])),
    ], axis=1), 0.05, None))
    v8_cal = res["cal_log_lam_ens"]
    y_ca = np.stack([y_hg[m_ca], y_ag[m_ca]], axis=1)

    def _nll(wt):
        ll = wt * v8_cal + (1 - wt) * lgb_cal
        return float(np.mean(np.exp(ll) - y_ca * ll))

    if args.ens_weight is not None and not np.isnan(args.ens_weight):
        w_v8 = float(args.ens_weight)
    else:
        opt = minimize_scalar(_nll, bounds=(0.0, 1.0), method="bounded")
        w_v8 = float(opt.x)
    print(f"   Ensemble: w_v8={w_v8:.3f}  (Cal-NLL {_nll(w_v8):.4f} "
          f"vs v8-only {_nll(1.0):.4f}, lgbm-only {_nll(0.0):.4f})")

    # ── 4) Affine λ-Kalibrierung auf dem geblendeten Cal ────────────────────
    blend_cal = w_v8 * v8_cal + (1 - w_v8) * lgb_cal
    lam_cal = fit_affine_lam(blend_cal, y_ca[:, 0], y_ca[:, 1])
    print(f"   Kalibrierung: b_home={lam_cal['b_home']:.3f}  b_away={lam_cal['b_away']:.3f}")

    # ── Meta schreiben (predict_v8-kompatibel) ───────────────────────────────
    meta = {
        "checkpoints": [
            {"path": s["path"], "val_loss": s["cal_loss"], "val_acc": s["val_acc"]}
            for s in res["seeds"]
        ],
        "norm_mean": res["norm_mean"].tolist(),
        "norm_std": res["norm_std"].tolist(),
        "col_medians": res["col_medians"].tolist(),
        "dc_rho": res["dc_rho"],
        "cfg": res["cfg"]["model_cfg"],
        "train_cfg": res["cfg"],
        "ess": res["ess"],
        "lgbm": {"path": f"models/lgbm_{args.tag}.txt", "w_v8": w_v8},
        "lam_cal": lam_cal,
        "model_version": f"v9-ens-{args.tag}",
    }
    meta_path = REPO_ROOT / "data" / "models" / f"v8_latest_{args.tag}.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"   Meta: {meta_path}")
    print(f"\n   Tipps:  python -m scripts.make_tips --model v8 --tag {args.tag}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
