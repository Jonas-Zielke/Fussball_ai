"""
Regenerate the V9 LightGBM member (models/lgbm_{tag}.txt) WITHOUT retraining the
E8Net seeds.

Why: the committed lgbm_v9.txt was corrupt (LightGBM's text-model parser desynced
mid-tree → sometimes a catchable LightGBMError, sometimes a hard segfault, so
`predict_match_v8(tag="v9")` / `make_tips --tag v9` crashed nondeterministically).
This retrains *only* the LightGBM Poisson member with the exact V9 recipe (identical
to scripts/train_v9_final.py, deterministic) and re-saves it with the installed
LightGBM, leaving the E8Net checkpoints + meta (w_v8, lam_cal) untouched.

Usage:
    python -m scripts.regen_lgbm                 # tag v9, default V9 windows
    python -m scripts.regen_lgbm --tag v9
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    ap = argparse.ArgumentParser(description="Regenerate the V9 LightGBM member")
    ap.add_argument("--tag", default="v9")
    ap.add_argument("--half-life", type=float, default=1095.0)
    ap.add_argument("--train-start", default="2010-01-01")
    ap.add_argument("--cal-start", default="2025-06-15")
    ap.add_argument("--val-start", default="2026-06-10")
    ap.add_argument("--anchor", default="2026-06-11")
    args = ap.parse_args()

    import lightgbm as lgb
    from src.features_v8 import load_v8
    from src.train_v8_lib import recency_weights, effective_sample_size
    from scripts.backtest import _swap_static_np

    data = load_v8()
    dates = pd.to_datetime(data["dates"])
    m_tr = np.asarray((dates >= args.train_start) & (dates < args.cal_start))
    m_ca = np.asarray((dates >= args.cal_start) & (dates < args.val_start))
    X = data["X"].astype(np.float64)
    y_hg = data["y_home_goals"].astype(np.float64)
    y_ag = data["y_away_goals"].astype(np.float64)

    w = recency_weights(pd.DatetimeIndex(dates[m_tr]), args.half_life,
                        pd.Timestamp(args.anchor))
    print(f"   LGBM: ESS={effective_sample_size(w):,.0f}/{int(m_tr.sum()):,}")
    w = w / w.mean()

    # Symmetric training: one machine for "goals of the home-side team".
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

    out = REPO_ROOT / "models" / f"lgbm_{args.tag}.txt"
    booster.booster_.save_model(str(out))

    # Verify it reloads cleanly (the whole point).
    reloaded = lgb.Booster(model_file=str(out))
    print(f"   saved + verified: {out} ({booster.best_iteration_} trees, "
          f"{reloaded.num_feature()} features)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
