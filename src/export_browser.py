"""
Exportiert das trainierte V2-Modell + Team-States + Aliase als JSON-Dateien
fuer den Browser-lokalen Einsatz in der Web-Applikation.

Output: ../Profilov2/public/data/wm-predictor/
  - model.json         - 7 fold-BatchNorm-Linears + Architektur
  - teams.json         - 336 Teams mit allen Features
  - aliases.json       - Team-Name-Aliase (DE/EN/FR/...)
  - config.json        - Normalisierungs-Stats, Temperatur, Feature-Order

Im Browser: Vanilla-JS-Inference (~5KB), kein ONNX/TF.js noetig.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from .train_v2 import FootballNet, MODELS_DIR
from .features_v2 import get_current_team_ratings, _team_to_continent, ELOG_START

REPO_ROOT = Path(__file__).resolve().parent.parent
# Output ins public/-Verzeichnis der Webseite
# REPO_ROOT ist E:\Projects\Fussball_ai, Ziel ist E:\Profilov2\public\data\wm-predictor
WEBSITE_PUBLIC = Path("E:/Profilov2/public/data/wm-predictor")
WEBSITE_PUBLIC.mkdir(parents=True, exist_ok=True)


def _fold_bn(linear: nn.Linear, bn: nn.BatchNorm1d) -> tuple[np.ndarray, np.ndarray]:
    """Faltet BatchNorm1d in die vorangehende Linear-Schicht.

    Eval-Mode: y = ((x - mu) / sqrt(var + eps)) * gamma + beta
                  = x * (gamma / sqrt(var + eps)) - mu * (gamma / sqrt(var + eps)) + beta
    Mit Linear: y = Wx + b
    Ersatz: W' = W * (gamma / sqrt(var + eps)),  b' = (b - mu) * (gamma / sqrt(var + eps)) + beta
    """
    w = linear.weight.detach().cpu().numpy().astype(np.float32)
    b = linear.bias.detach().cpu().numpy().astype(np.float32) if linear.bias is not None else np.zeros(w.shape[0], dtype=np.float32)
    gamma = bn.weight.detach().cpu().numpy().astype(np.float32)
    beta = bn.bias.detach().cpu().numpy().astype(np.float32)
    mu = bn.running_mean.detach().cpu().numpy().astype(np.float32)
    var = bn.running_var.detach().cpu().numpy().astype(np.float32)
    eps = bn.eps
    scale = gamma / np.sqrt(var + eps)  # shape: (out_dim,)
    w_new = w * scale[:, np.newaxis]  # broadcast: (out, in) * (out, 1) -> (out, in)
    b_new = (b - mu) * scale + beta
    return w_new, b_new


def _fold_block(block: nn.Module, prev_dim: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Faltet einen ResidualBlock."""
    w1, b1 = _fold_bn(block.lin1, block.bn1)
    w2, b2 = _fold_bn(block.lin2, block.bn2)
    return w1, b1, w2, b2


def _serialize_one_model(model: nn.Module) -> dict:
    """Konvertiert ein FootballNet (eval mode) in ein flaches Layer-Dict."""
    state = {}
    # Input projection
    w, b = _fold_bn(model.input_proj[0], model.input_proj[1])
    state["input_w"] = w.tolist()
    state["input_b"] = b.tolist()
    # Dropout wird im eval-mode uebersprungen
    # Residual blocks
    for i, block in enumerate(model.blocks):
        w1, b1, w2, b2 = _fold_block(block, model.hidden)
        state[f"block{i}_w1"] = w1.tolist()
        state[f"block{i}_b1"] = b1.tolist()
        state[f"block{i}_w2"] = w2.tolist()
        state[f"block{i}_b2"] = b2.tolist()
    # Classification head
    cls_lin1 = model.cls_head[0]
    cls_drop = model.cls_head[2]  # noop
    cls_lin2 = model.cls_head[3]
    state["cls_w1"] = cls_lin1.weight.detach().cpu().numpy().astype(np.float32).tolist()
    state["cls_b1"] = cls_lin1.bias.detach().cpu().numpy().astype(np.float32).tolist()
    state["cls_w2"] = cls_lin2.weight.detach().cpu().numpy().astype(np.float32).tolist()
    state["cls_b2"] = cls_lin2.bias.detach().cpu().numpy().astype(np.float32).tolist()
    # Regression head
    reg_lin1 = model.reg_head[0]
    reg_lin2 = model.reg_head[3]
    state["reg_w1"] = reg_lin1.weight.detach().cpu().numpy().astype(np.float32).tolist()
    state["reg_b1"] = reg_lin1.bias.detach().cpu().numpy().astype(np.float32).tolist()
    state["reg_w2"] = reg_lin2.weight.detach().cpu().numpy().astype(np.float32).tolist()
    state["reg_b2"] = reg_lin2.bias.detach().cpu().numpy().astype(np.float32).tolist()
    return state


def export_model() -> dict:
    """Laedt das V2-Ensemble und serialisiert es in JSON."""
    from .train_v2 import _load_latest_v2, DEVICE
    print("=" * 70)
    print(" Export V2-Ensemble -> Browser-JSON")
    print("=" * 70)
    models, bundle = _load_latest_v2()
    print(f"   Ensemble-Groesse: {len(models)}")
    print(f"   Val-Accuracy: {bundle['ensemble_val_acc']:.4f}")

    serialized = []
    for m in models:
        m.eval()
        serialized.append(_serialize_one_model(m))

    arch = {
        "n_models": len(models),
        "in_dim": bundle["in_dim"],
        "hidden": models[0].hidden,
        "n_blocks": models[0].n_blocks,
        "n_classes": 3,
        "hidden_configs": bundle["hidden_configs"],
        "n_blocks_configs": bundle["n_blocks_configs"],
        "dropout_configs": bundle["dropout_configs"],
    }

    out = {
        "architecture": arch,
        "models": serialized,
        "norm_stats": bundle["norm_stats"],
        "goal_stats": bundle["goal_stats"],
        "temperature": float(bundle.get("temperature", 1.0)),
        "ensemble_val_acc": float(bundle["ensemble_val_acc"]),
        "calibrated_val_acc": float(bundle.get("calibrated_val_acc", bundle["ensemble_val_acc"])),
        "feature_names": list(bundle["feature_names"]),
        "n_features": len(bundle["feature_names"]),
    }
    out_path = WEBSITE_PUBLIC / "model.json"
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, separators=(",", ":"))
    size_mb = out_path.stat().st_size / 1e6
    print(f"   geschrieben: {out_path} ({size_mb:.2f} MB)")
    print("=" * 70)
    return out


def _sf(val, default: float) -> float:
    """float() mit NaN-Fallback — verhindert ungültiges JSON."""
    try:
        v = float(val)
        return default if v != v else v  # NaN != NaN
    except (TypeError, ValueError):
        return default


def export_teams() -> dict:
    """Exportiert alle Team-States (V6, mit Kader-Features) als JSON."""
    print(">> Exportiere Team-States (V6)...")
    from .features_v6 import get_current_team_ratings_v6
    from .shootout_features import get_current_shootout_ratings
    state = get_current_team_ratings_v6()
    pen = get_current_shootout_ratings()

    teams_out = {}
    for name, st in state.items():
        ps = pen.get(name, {})
        teams_out[name] = {
            "elo": _sf(st.get("elo"), 1500.0),
            "re_elo": _sf(st.get("re_elo"), 1500.0),
            "vr_elo": _sf(st.get("vr_elo"), 1500.0),
            "form1": _sf(st.get("form1"), 1.0),
            "form2": _sf(st.get("form2"), 1.0),
            "form3": _sf(st.get("form3"), 1.0),
            "form5": _sf(st.get("form5"), 1.0),
            "form10": _sf(st.get("form10"), 1.0),
            "gf5": _sf(st.get("gf5"), 1.0),
            "ga5": _sf(st.get("ga5"), 1.0),
            "gd5": _sf(st.get("gd5"), 0.0),
            "win_streak": int(st.get("win_streak", 0)),
            "unbeaten": int(st.get("unbeaten", 0)),
            "continent": int(st.get("continent", 0)),
            "oppo_elo5": _sf(st.get("oppo_elo5"), 1500.0),
            "w_form": _sf(st.get("w_form"), 1.0),
            "momentum": _sf(st.get("momentum"), 0.0),
            "wins_top10": _sf(st.get("wins_top10"), 0.0),
            "wins_top20": _sf(st.get("wins_top20"), 0.0),
            "stability": _sf(st.get("stability"), 1.0),
            # V6 squad features
            "sq_ovr": _sf(st.get("sq_ovr"), 74.0),
            "sq_att": _sf(st.get("sq_att"), 73.0),
            "sq_def": _sf(st.get("sq_def"), 73.0),
            "sq_age": _sf(st.get("sq_age"), 27.0),
            "sq_depth": _sf(st.get("sq_depth"), 5.0),
            "coach": st.get("coach", ""),
            # Elfmeterschießen (Shootout-V1)
            "pen_skill": _sf(ps.get("pen_skill"), 0.5),
            "pen_games": int(ps.get("pen_games", 0)),
            "last_match": st.get("last_match"),
        }

    items_sorted = sorted(teams_out.items(), key=lambda x: -x[1]["elo"])
    ranking = [(name, float(st["elo"])) for name, st in items_sorted]

    out = {
        "n_teams": len(teams_out),
        "as_of": max((t["last_match"] for t in teams_out.values() if t["last_match"]), default=None),
        "teams": teams_out,
        "ranking_top50": ranking[:50],
        "model_version": "v6-squad-aware",
    }
    out_path = WEBSITE_PUBLIC / "teams.json"
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, separators=(",", ":"))
    size_mb = out_path.stat().st_size / 1e6
    print(f"   geschrieben: {out_path} ({size_mb:.2f} MB) - {len(teams_out)} Teams")
    return out


def export_aliases() -> dict:
    """Exportiert die Team-Name-Aliase als JSON."""
    print(">> Exportiere Aliase...")
    from .team_normalize import TEAM_ALIASES, TOURNAMENT_TIERS
    out = {
        "aliases": TEAM_ALIASES,
        "tournament_weights": TOURNAMENT_TIERS,
    }
    out_path = WEBSITE_PUBLIC / "aliases.json"
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, separators=(",", ":"))
    print(f"   geschrieben: {out_path} ({out_path.stat().st_size/1024:.1f} KB)")
    return out


def export_config() -> dict:
    """Exportiert die V6-Feature-Konfiguration."""
    print(">> Exportiere Config (V6)...")
    out = {
        "feature_order": [
            "neutral", "tournament_w",
            "elo_a", "elo_b", "elo_diff",
            "vr_elo_a", "vr_elo_b", "vr_elo_diff",
            "re_elo_a", "re_elo_b", "re_elo_diff",
            "form1_a", "form1_b", "form2_a", "form2_b", "form3_a", "form3_b",
            "form5_a", "form5_b", "form10_a", "form10_b",
            "gf5_a", "gf5_b", "ga5_a", "ga5_b", "gd5_a", "gd5_b",
            "h2h_a", "h2h_b",
            "rest_a", "rest_b",
            "win_streak_a", "win_streak_b", "unbeaten_a",
            "continent_a", "continent_b", "oppo_elo5_a", "oppo_elo5_b",
            "w_form_a", "w_form_b",
            "momentum_a", "momentum_b",
            "wins_top10_a", "wins_top10_b", "wins_top20_a", "wins_top20_b",
            "stability_a", "stability_b",
            "sq_ovr_a", "sq_ovr_b",
            "sq_att_a", "sq_att_b",
            "sq_def_a", "sq_def_b",
            "sq_diff",
            "sq_age_a", "sq_age_b",
        ],
        "model_version": "v6-squad-aware",
        "n_features": 57,
        "home_advantage_elo": 80.0,
        "rest_default_days": 30,
        "rest_cap_days": 365,
        "dc_rho": -0.13,
    }
    out_path = WEBSITE_PUBLIC / "config.json"
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, separators=(",", ":"))
    print(f"   geschrieben: {out_path}")
    return out


def export_h2h() -> dict:
    """Pre-computed H2H-Werte fuer alle Team-Paare aus den letzten 10 direkten Duellen."""
    print(">> Pre-compute H2H-Map...")
    import pandas as pd
    df = pd.read_csv(REPO_ROOT / "data" / "raw" / "results.csv", parse_dates=["date"])
    df = df.dropna(subset=["home_score", "away_score"])
    # Sample die letzten ~5000 Spiele fuer H2H (genug fuer 99% der Query-Paare)
    df_recent = df.tail(8000)
    h2h = {}
    for _, r in df_recent.iterrows():
        a, b = r["home_team"], r["away_team"]
        key = "::".join(sorted([a, b]))
        if key not in h2h:
            h2h[key] = {"n": 0, "wins_a": 0, "wins_b": 0}
        h2h[key]["n"] += 1
        if r["home_score"] > r["away_score"]:
            if a < b:
                h2h[key]["wins_a"] += 1
            else:
                h2h[key]["wins_b"] += 1
        elif r["home_score"] < r["away_score"]:
            if a < b:
                h2h[key]["wins_b"] += 1
            else:
                h2h[key]["wins_a"] += 1
    out_path = WEBSITE_PUBLIC / "h2h.json"
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(h2h, fh, separators=(",", ":"))
    print(f"   geschrieben: {out_path} ({out_path.stat().st_size/1024:.1f} KB, {len(h2h)} Paare)")
    return h2h


def export_shootout_model() -> dict:
    """Exportiert das kleine Shootout-V1-Modell als JSON für den Browser."""
    print(">> Exportiere Shootout-Modell...")
    import torch
    from .shootout_features import MODEL_PATH
    bundle = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    sd = bundle["state_dict"]
    out = {
        "feature_order": bundle["feature_order"],
        "norm_stats": bundle["norm_stats"],
        "elo_scale": 100.0,
        "k_shrink": 5.0,
        # 1 Hidden-Layer: w0 (hidden x in), b0 (hidden), w1 (1 x hidden), b1 (1)
        "w0": sd["net.0.weight"].cpu().numpy().astype(float).tolist(),
        "b0": sd["net.0.bias"].cpu().numpy().astype(float).tolist(),
        "w1": sd["net.3.weight"].cpu().numpy().astype(float).tolist(),
        "b1": sd["net.3.bias"].cpu().numpy().astype(float).tolist(),
        "model_version": "shootout-v1",
        "val_metrics": bundle.get("val_metrics", {}),
    }
    out_path = WEBSITE_PUBLIC / "shootout_model.json"
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, separators=(",", ":"))
    print(f"   geschrieben: {out_path} ({out_path.stat().st_size/1024:.1f} KB)")
    return out


def export_odds() -> None:
    """Kopiert wm2026_odds.json ins public/-Verzeichnis der Webseite."""
    import shutil
    print(">> Exportiere WM2026-Quoten...")
    src = REPO_ROOT / "data" / "raw" / "wm2026_odds.json"
    if not src.exists():
        print("   SKIP: data/raw/wm2026_odds.json nicht gefunden (run scripts/fetch_wm2026_odds.py first)")
        return
    dst = WEBSITE_PUBLIC / "wm2026_odds.json"
    shutil.copy2(src, dst)
    print(f"   geschrieben: {dst} ({dst.stat().st_size / 1024:.1f} KB)")


def export_all() -> None:
    """Fuehrt alle Exports aus."""
    print(f"Output: {WEBSITE_PUBLIC}")
    WEBSITE_PUBLIC.mkdir(parents=True, exist_ok=True)
    export_model()
    export_teams()
    export_aliases()
    export_config()
    export_h2h()
    export_shootout_model()
    export_odds()
    total = sum((WEBSITE_PUBLIC / f).stat().st_size for f in WEBSITE_PUBLIC.iterdir() if f.is_file())
    print(f"\n   GESAMT: {total/1e6:.2f} MB in {WEBSITE_PUBLIC}")


def main() -> int:
    export_all()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
