"""
V8-Inferenz in Python — Tipps direkt aus E8Net-Checkpoints.

Bisher lief die Tipp-Generierung über predict_match_v6 (V7-Pfad); V8 existierte
nur als ONNX im Browser. Dieses Modul macht V8(-Kandidaten) in Python nutzbar:

    from src.predict_v8 import predict_match_v8
    pred = predict_match_v8("Germany", "Ecuador", neutral=True)

- Checkpoints: models/v8_seed{N}[_tag].pt (alle vorhandenen Seeds, Ensemble)
- Meta:        data/models/v8_latest[_tag].json (optional "lam_cal" = affine
               λ-Kalibrierung aus src/calibrate.py)
- Team-State:  data/processed/v8_final_state.json (scripts/export_v8_state.py,
               inkl. letztem Spiel — kein Off-by-one wie v8_team_tensors.json)

Rückgabeformat ist kompatibel zu predict_match_v6 (probabilities,
most_likely_scores, kicktipp_tip, odds_blended, ...) plus V8-Extras
(p_et, p_pen_given_et, p_home_pen, lambdas).
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
STATE_PATH = REPO_ROOT / "data" / "processed" / "v8_final_state.json"

_CACHE: dict = {}


def _load_bundle(tag: str = "") -> dict:
    if tag in _CACHE:
        return _CACHE[tag]
    import torch
    from .model_v8 import E8Config, build_model
    from .train_v8_lib import DEVICE

    suffix = f"_{tag}" if tag else ""
    ckpts = []
    for seed in range(8):
        p = REPO_ROOT / "models" / f"v8_seed{seed}{suffix}.pt"
        if p.exists():
            ckpts.append(torch.load(p, map_location=DEVICE, weights_only=False))
    if not ckpts:
        raise FileNotFoundError(f"Keine Checkpoints models/v8_seed*{suffix}.pt gefunden.")

    models = []
    for ck in ckpts:
        m = build_model(E8Config(**ck["cfg"])).to(DEVICE)
        m.load_state_dict(ck["state_dict"])
        m.eval()
        models.append(m)

    meta_path = REPO_ROOT / "data" / "models" / f"v8_latest{suffix}.json"
    meta = {}
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)

    if not STATE_PATH.exists():
        raise FileNotFoundError(
            f"{STATE_PATH} fehlt — erst `python -m scripts.export_v8_state` ausführen.")
    with open(STATE_PATH) as f:
        team_state = json.load(f)

    # Optionales LGBM-Ensemble-Mitglied (Meta: {"lgbm": {"path", "w_v8"}})
    lgbm_booster = None
    w_v8 = 1.0
    lg = meta.get("lgbm")
    if lg:
        import lightgbm as lgb
        lgbm_path = REPO_ROOT / lg["path"]
        if lgbm_path.exists():
            lgbm_booster = lgb.Booster(model_file=str(lgbm_path))
            w_v8 = float(lg["w_v8"])

    bundle = {
        "models": models,
        "ckpts": ckpts,
        "meta": meta,
        "team_state": team_state,
        "device": DEVICE,
        "n_seeds": len(models),
        "lgbm": lgbm_booster,
        "w_v8": w_v8,
    }
    _CACHE[tag] = bundle
    return bundle


def _team_tensors(bundle: dict, team: str) -> tuple[np.ndarray, np.ndarray, bool]:
    ts = bundle["team_state"].get(team)
    if ts is None:
        return (np.zeros((10, 7), dtype=np.float32),
                np.zeros((15, 3), dtype=np.float32), False)
    return (np.array(ts["seq"], dtype=np.float32),
            np.array(ts["squad"], dtype=np.float32), True)


def predict_match_v8(
    home: str,
    away: str,
    neutral: bool = True,
    tournament: str = "FIFA World Cup",
    tag: str = "",
    market_grid: bool = True,
) -> dict:
    import torch
    from .features_v6 import (
        _build_inference_vector_v6, get_current_team_ratings_v6,
        score_grid, wdl_from_grid, supremacy_blend_grid, _load_odds, _blend_wdl, _ODDS_CACHE,
    )
    from .team_normalize import normalize_team_name
    from .kicktipp import load_scheme, optimal_tip

    bundle = _load_bundle(tag)
    home_n = normalize_team_name(home)
    away_n = normalize_team_name(away)

    # ── Statischer 57-Vektor (gleiche Live-Pipeline wie V7) ─────────────────
    state = get_current_team_ratings_v6()
    X = _build_inference_vector_v6(home_n, away_n, neutral, tournament,
                                   datetime.now(), state).reshape(1, -1)

    # ── Sequenz-/Kader-Tensoren aus dem Final-State ─────────────────────────
    seq_h, squad_h, ok_h = _team_tensors(bundle, home_n)
    seq_a, squad_a, ok_a = _team_tensors(bundle, away_n)

    # ── Ensemble-Forward über alle Seeds ────────────────────────────────────
    log_lams, p_ets, p_pens, p_hpens = [], [], [], []
    dev = bundle["device"]
    for ck, model in zip(bundle["ckpts"], bundle["models"]):
        med = np.array(ck["col_medians"], dtype=np.float64)
        mean = np.array(ck["norm_mean"], dtype=np.float32)
        std = np.array(ck["norm_std"], dtype=np.float32)
        Xi = X.astype(np.float64).copy()
        nans = np.isnan(Xi)
        Xi[nans] = np.take(med, np.where(nans)[1])
        X_n = ((Xi - mean) / std).astype(np.float32)
        ctx = X_n[:, [0, 1]]
        with torch.no_grad():
            out = model(
                torch.from_numpy(X_n).to(dev),
                torch.from_numpy(seq_h[None]).to(dev),
                torch.from_numpy(squad_h[None]).to(dev),
                torch.from_numpy(seq_a[None]).to(dev),
                torch.from_numpy(squad_a[None]).to(dev),
                torch.from_numpy(ctx).to(dev),
            )
        log_lams.append([float(out.log_lam_home[0]), float(out.log_lam_away[0])])
        p_ets.append(float(out.p_et[0]))
        p_pens.append(float(out.p_pen_given_et[0]))
        p_hpens.append(float(out.p_home_pen[0]))

    ll = np.mean(np.array(log_lams), axis=0)

    # ── Optionales LGBM-Blend (log-λ-Mittel, Gewicht aus Meta) ──────────────
    if bundle["lgbm"] is not None:
        from .model_v8 import SWAP_PAIRS, DIFF_IDX
        X_raw = X.astype(np.float64)
        X_swap = X_raw.copy()
        for ai, bi in SWAP_PAIRS:
            X_swap[:, ai] = X_raw[:, bi]
            X_swap[:, bi] = X_raw[:, ai]
        for di in DIFF_IDX:
            X_swap[:, di] = -X_raw[:, di]
        lam_lgb_h = float(np.clip(bundle["lgbm"].predict(X_raw)[0], 0.05, None))
        lam_lgb_a = float(np.clip(bundle["lgbm"].predict(X_swap)[0], 0.05, None))
        w = bundle["w_v8"]
        ll = w * ll + (1 - w) * np.log(np.array([lam_lgb_h, lam_lgb_a]))

    # ── Optionale affine λ-Kalibrierung (aus Meta) ──────────────────────────
    lam_cal = bundle["meta"].get("lam_cal")
    if lam_cal:
        ll = np.array([
            lam_cal["a_home"] + lam_cal["b_home"] * ll[0],
            lam_cal["a_away"] + lam_cal["b_away"] * ll[1],
        ])

    lam_h, lam_a = float(np.exp(ll[0])), float(np.exp(ll[1]))
    dc_rho = float(np.mean([ck["dc_rho"] for ck in bundle["ckpts"]]))
    grid = score_grid(lam_h, lam_a, dc_rho, n=10)
    p_home, p_draw, p_away = wdl_from_grid(grid)

    # ── Markt-Blend ─────────────────────────────────────────────────────────
    # market_grid=True blendet den Markt ins SCORE-GRID (schärft den Tipp, da
    # optimal_tip aus dem Grid wählt) statt nur die Anzeige-W/D/L (Alt-Verhalten).
    _load_odds()
    odds_blended = False
    market_grid_applied = False
    market_probs = None
    lookup_key = f"{home_n}|{away_n}"
    if lookup_key in _ODDS_CACHE.get("matches", {}):
        mo = _ODDS_CACHE["matches"][lookup_key]
        w_blend = _ODDS_CACHE["blend_weight"]
        market_wdl = (float(mo["home"]), float(mo["draw"]), float(mo["away"]))
        odds_blended = True
        market_probs = {"home_win": market_wdl[0], "draw": market_wdl[1],
                        "away_win": market_wdl[2]}
        if market_grid:
            grid = supremacy_blend_grid(lam_h, lam_a, dc_rho, market_wdl, w_blend, n=10)
            p_home, p_draw, p_away = wdl_from_grid(grid)
            market_grid_applied = True
        else:
            p_home, p_draw, p_away = _blend_wdl(
                (p_home, p_draw, p_away), market_wdl, w_blend)

    # ── KickTipp Decision Layer ─────────────────────────────────────────────
    odds_for_kt = (
        {"home": market_probs["home_win"], "draw": market_probs["draw"],
         "away": market_probs["away_win"]}
        if market_probs else {"home": p_home, "draw": p_draw, "away": p_away}
    )
    kt = optimal_tip(grid, odds_for_kt, load_scheme())
    kicktipp_tip = {
        "home": kt["tip"][0], "away": kt["tip"][1],
        "expected_points": kt["expected_points"],
        "alternatives": kt["alternatives"][:3],
    }

    # Top-Scorelines
    flat = [(int(h), int(a), float(grid[h, a])) for h in range(10) for a in range(10)]
    flat.sort(key=lambda x: -x[2])
    most_likely = [{"home": h, "away": a, "prob": p} for h, a, p in flat[:5]]

    n_seeds = bundle["n_seeds"]
    return {
        "home": home_n, "away": away_n,
        "probabilities": {"home_win": p_home, "draw": p_draw, "away_win": p_away},
        "most_likely_scores": most_likely,
        "lambdas": {"home": lam_h, "away": lam_a},
        "p_et": float(np.mean(p_ets)),
        "p_pen_given_et": float(np.mean(p_pens)),
        "p_home_pen": float(np.mean(p_hpens)),
        "odds_blended": odds_blended,
        "market_probs": market_probs,
        "kicktipp_tip": kicktipp_tip,
        "tensors_found": {"home": ok_h, "away": ok_a},
        "model_version": f"v8-e8net-ensemble-{n_seeds}" + (f"-{tag}" if tag else "")
                         + ("+lgbm" if bundle["lgbm"] is not None else "")
                         + ("-cal" if lam_cal else "")
                         + ("+mktgrid" if market_grid_applied else ""),
        "ensemble_size": n_seeds,
    }
