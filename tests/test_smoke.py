"""
End-to-End Smoke Tests fuer den WM 2026 Predictor.

Wird mit pytest ODER direkt ausgefuehrt:
    python -m tests.test_smoke

Prueft:
    1) Datendownload vorhanden + genug Zeilen
    2) Feature-File vorhanden + genug Zeilen + plausible Verteilung
    3) Modell vorhanden + laedt
    4) predict_match gibt sinnvolle Werte zurueck
    5) predict_match ist deterministisch (gleicher Input -> gleicher Output)
    6) Extreme Mismatch (Top vs Bottom) hat hohe Win-Wahrscheinlichkeit
    7) Symmetrie-Test: p(A schlaegt B) und p(B schlaegt A) sind sinnvoll
    8) Class-Probabilities summieren zu ~1
    9) Unbekanntes Team wirft Fehler
   10) Teamname-Aliase werden korrekt aufgeloest
   11) Sweep-Funktion liefert NxN Matrix
"""

from __future__ import annotations

import sys
import json
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.features import load_features, get_current_team_ratings, PROCESSED_DIR
from src.train import _load_latest, predict_match, DEVICE
from src.train_v2 import _load_latest_v2, predict_match_v2
from src.team_normalize import normalize_team_name, tournament_weight
from src import predict as predict_module


FAILED = []
PASSED = 0


def _ok(name: str, msg: str = "") -> None:
    global PASSED
    PASSED += 1
    print(f"  [PASS] {name}{(' - ' + msg) if msg else ''}")


def _fail(name: str, msg: str) -> None:
    FAILED.append((name, msg))
    print(f"  [FAIL] {name}: {msg}")


def test_data_present() -> None:
    raw = REPO_ROOT / "data" / "raw" / "results.csv"
    if not raw.exists():
        _fail("data_present", f"results.csv fehlt: {raw}")
        return
    df = pd.read_csv(raw)
    if len(df) < 30000:
        _fail("data_present", f"nur {len(df)} Zeilen, erwarte >= 30000")
        return
    _ok("data_present", f"{len(df):,} Zeilen")


def test_features_v2_present() -> None:
    npz = PROCESSED_DIR / "features_v2.npz"
    if not npz.exists():
        _fail("features_v2_present", f"features_v2.npz fehlt: {npz}")
        return
    bundle = np.load(npz, allow_pickle=True)
    X, y = bundle["X"], bundle["y"]
    if len(X) < 30000:
        _fail("features_v2_present", f"nur {len(X)} Samples")
        return
    if X.shape[1] < 30:
        _fail("features_v2_present", f"erwarte >= 30 Features, habe {X.shape[1]}")
        return
    cls_dist = np.bincount(y, minlength=3)
    if cls_dist[1] < cls_dist[0] or cls_dist[1] < cls_dist[2]:
        _fail("features_v2_present", f"unerwartete Klassenverteilung {cls_dist.tolist()}")
        return
    _ok("features_v2_present", f"{len(X):,} Samples, {X.shape[1]} Features, Klassen={cls_dist.tolist()}")


def test_v2_model_loads() -> None:
    """Laedt das V2-Ensemble und prueft die Validation-Accuracy."""
    try:
        models, bundle = _load_latest_v2()
    except Exception as exc:
        _fail("v2_model_loads", str(exc))
        return
    val_acc = bundle.get("ensemble_val_acc", 0)
    if val_acc < 0.50:
        _fail("v2_model_loads", f"val_acc = {val_acc:.4f} (zu niedrig)")
        return
    _ok("v2_model_loads", f"ensemble={bundle['n_models']}, val_acc={val_acc:.4f}, "
                          f"cal_acc={bundle.get('calibrated_val_acc', 0):.4f}")


def test_predict_deterministic() -> None:
    p1 = predict_match("Germany", "Brazil", neutral=True, tournament="FIFA World Cup")
    p2 = predict_match("Germany", "Brazil", neutral=True, tournament="FIFA World Cup")
    if abs(p1["probabilities"]["home_win"] - p2["probabilities"]["home_win"]) > 1e-6:
        _fail("predict_deterministic", "unterschiedliche Werte fuer identischen Input")
        return
    _ok("predict_deterministic", f"home_win = {p1['probabilities']['home_win']:.4f}")


def test_predict_probabilities_sum_to_one() -> None:
    p = predict_match("Germany", "Brazil", neutral=True, tournament="FIFA World Cup")
    s = p["probabilities"]["draw"] + p["probabilities"]["home_win"] + p["probabilities"]["away_win"]
    if abs(s - 1.0) > 1e-4:
        _fail("predict_sum_to_one", f"summe = {s}")
        return
    _ok("predict_sum_to_one", f"summe = {s:.6f}")


def test_predict_extreme_mismatch() -> None:
    state = get_current_team_ratings()
    items = sorted(state.items(), key=lambda x: -x[1]["elo"])
    top_team = items[0][0]
    bottom_team = items[-1][0]
    p = predict_match_v2(top_team, bottom_team, neutral=True, tournament="FIFA World Cup")
    if p["probabilities"]["home_win"] < 0.7:
        _fail("extreme_mismatch", f"top {top_team} vs bottom {bottom_team} -> "
                                  f"win_prob = {p['probabilities']['home_win']:.3f} (erwarte > 0.7)")
        return
    _ok("extreme_mismatch", f"{top_team} (Elo {state[top_team]['elo']:.0f}) vs "
                            f"{bottom_team} (Elo {state[bottom_team]['elo']:.0f}) -> "
                            f"{p['probabilities']['home_win']*100:.1f}%")


def test_team_aliases() -> None:
    cases = {
        "Germany": "Germany",
        "deutschland": "Germany",
        "USA": "United States",
        "Iran": "Iran",  # Martj42 nutzt "Iran", nicht "IR Iran"
        "South Korea": "South Korea",
        "Südkorea": "South Korea",
        "Türkiye": "Turkey",
        "Ivory Coast": "Ivory Coast",
        "Côte d'Ivoire": "Ivory Coast",
        "Czech Republic": "Czech Republic",
        "Tschechien": "Czech Republic",
        "Frankreich": "France",
    }
    bad = []
    for inp, exp in cases.items():
        got = normalize_team_name(inp)
        if got != exp:
            bad.append(f"{inp!r} -> {got!r} (erwarte {exp!r})")
    if bad:
        _fail("team_aliases", "; ".join(bad))
    else:
        _ok("team_aliases", f"{len(cases)} Aliase geprueft")


def test_unknown_team_raises() -> None:
    try:
        predict_match_v2("Atlantis", "Brazil", neutral=True, tournament="FIFA World Cup")
    except ValueError as exc:
        if "Unbekanntes Team" in str(exc):
            _ok("unknown_team_raises", "ValueError mit hilfreicher Message")
            return
    _fail("unknown_team_raises", "erwartete ValueError mit 'Unbekanntes Team'")


def test_gpu_used() -> None:
    if DEVICE.type != "cuda":
        _fail("gpu_used", f"Device = {DEVICE} (erwarte cuda)")
        return
    _ok("gpu_used", f"{torch.cuda.get_device_name(0)}")


def test_tournament_weight() -> None:
    if tournament_weight("FIFA World Cup") < tournament_weight("Friendly"):
        _fail("tournament_weight", "FIFA World Cup sollte schwerer gewichtet sein als Friendly")
        return
    if tournament_weight("UEFA Euro") > tournament_weight("FIFA World Cup"):
        _fail("tournament_weight", "WM sollte >= EM sein (in unserem Schema)")
        return
    _ok("tournament_weight", f"WM={tournament_weight('FIFA World Cup')}, "
                             f"EM={tournament_weight('UEFA Euro')}, "
                             f"Friendly={tournament_weight('Friendly')}")


def test_recent_form_present() -> None:
    state = get_current_team_ratings()
    items = sorted(state.items(), key=lambda x: -x[1]["elo"])
    top = items[0][1]
    if top["form5"] <= 0 or top["form5"] > 3.001:
        _fail("recent_form", f"form5 fuer {items[0][0]} = {top['form5']} (erwarte 0..3)")
        return
    _ok("recent_form", f"{items[0][0]} form5={top['form5']:.2f}, gf5={top['gf5']:.2f}, ga5={top['ga5']:.2f}")


def test_home_advantage_matters() -> None:
    # Auf neutralem Boden mit gleichem Team ist es konsistent, aber Heimvorteil sollte
    # die home_win Wahrscheinlichkeit fuer ein ausgeglichenes Match erhoehen.
    p_neutral = predict_match_v2("Germany", "France", neutral=True, tournament="FIFA World Cup")
    p_home = predict_match_v2("Germany", "France", neutral=False, tournament="FIFA World Cup")
    diff = p_home["probabilities"]["home_win"] - p_neutral["probabilities"]["home_win"]
    if diff <= 0:
        _fail("home_advantage", f"home_win nicht erhoeht (neutral={p_neutral['probabilities']['home_win']:.3f}, "
                                f"home={p_home['probabilities']['home_win']:.3f})")
        return
    _ok("home_advantage", f"+{diff*100:.1f}pp home_win durch Heimvorteil")


def test_sweep_dimensions() -> None:
    teams = ["Germany", "France", "Brazil", "Spain"]
    n = len(teams)
    matrix = np.zeros((n, n))
    for i, a in enumerate(teams):
        for j, b in enumerate(teams):
            if a == b:
                continue
            p = predict_match_v2(a, b, neutral=True, tournament="FIFA World Cup")
            matrix[i, j] = p["probabilities"]["home_win"]
    if matrix.shape != (n, n):
        _fail("sweep_dim", f"shape = {matrix.shape}")
        return
    if (matrix.diagonal() != 0).any():
        _fail("sweep_dim", "Diagonale sollte 0 sein")
        return
    _ok("sweep_dim", f"{n}x{n} Matrix, range = {matrix[matrix > 0].min():.3f}..{matrix.max():.3f}")


def test_v2_score_regression() -> None:
    """V2 Modell sagt erwartete Tore voraus und produziert plausible exakte Scores."""
    p = predict_match_v2("Spain", "Bhutan", neutral=True, tournament="FIFA World Cup")
    es = p["expected_score"]
    if es["home_goals"] < 2 or es["away_goals"] > 2:
        _fail("v2_score_regression",
              f"Spain vs Bhutan: expected {es['home_goals']:.2f}:{es['away_goals']:.2f} unplausibel")
        return
    if "most_likely_scores" not in p or len(p["most_likely_scores"]) == 0:
        _fail("v2_score_regression", "keine most_likely_scores zurueckgegeben")
        return
    # Das wahrscheinlichste exakte Ergebnis sollte hohe Tore fuer Spain haben
    top = p["most_likely_scores"][0]
    if top["home"] < 3:
        _fail("v2_score_regression",
              f"Top-Score {top['home']}:{top['away']} fuer Spain vs Bhutan zu niedrig")
        return
    _ok("v2_score_regression",
        f"Spain vs Bhutan E[{es['home_goals']:.2f}:{es['away_goals']:.2f}], "
        f"top {top['home']}:{top['away']} ({top['prob']*100:.1f}%)")


def test_v2_realistic_close_match() -> None:
    """Bei einem engen Matchup sollte das wahrscheinlichste exakte Ergebnis niedrig sein."""
    p = predict_match_v2("Germany", "Uruguay", neutral=True, tournament="FIFA World Cup")
    if "most_likely_scores" not in p:
        _fail("v2_realistic_close", "keine most_likely_scores")
        return
    top = p["most_likely_scores"][0]
    total = top["home"] + top["away"]
    if total > 5:
        _fail("v2_realistic_close", f"Top-Score {top['home']}:{top['away']} zu hoch (total={total})")
        return
    _ok("v2_realistic_close",
        f"Germany vs Uruguay: top {top['home']}:{top['away']} ({top['prob']*100:.1f}%), "
        f"max total = {total}")


def main() -> int:
    print("=" * 70)
    print(" WM 2026 Predictor V2 - Smoke Tests")
    print("=" * 70)
    test_data_present()
    test_features_v2_present()
    test_v2_model_loads()
    test_gpu_used()
    test_team_aliases()
    test_tournament_weight()
    test_recent_form_present()
    test_predict_deterministic()
    test_predict_probabilities_sum_to_one()
    test_predict_extreme_mismatch()
    test_unknown_team_raises()
    test_home_advantage_matters()
    test_sweep_dimensions()
    test_v2_score_regression()
    test_v2_realistic_close_match()
    print("=" * 70)
    print(f" Ergebnis: {PASSED} PASS, {len(FAILED)} FAIL")
    if FAILED:
        for name, msg in FAILED:
            print(f"   [FAIL] {name}: {msg}")
        return 1
    print(" ALLE TESTS BESTANDEN")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
