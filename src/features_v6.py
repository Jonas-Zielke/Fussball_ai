"""
Feature-Engineering V6 – V5 + Kader-Stärke (FIFA-Ratings).

Neue Features gegenüber V5 (9 zusätzliche Dimensionen → 59 total):
  sq_ovr_a/b   - Durchschnittliches Overall Top-23 des Kaders
  sq_att_a/b   - Angriffs-Overall (Top Stürmer/Flügel/CAM)
  sq_def_a/b   - Verteidigungs-Overall (Top CB/LB/RB/GK)
  sq_diff      - sq_ovr_a - sq_ovr_b (Kader-Stärkedifferenz)
  sq_age_a/b   - Kader-Durchschnittsalter

Für Spiele außerhalb der FIFA-Datenbasis (vor 2015, nach 2022):
  - Vorwärts-/Rückwärtsfüllung mit dem nächsten verfügbaren Jahr
  - Bei unbekannter Nation: Median-Defaults

Für die aktuelle Vorhersage (Inference) werden squads_2026.json-Werte
statt der historischen FIFA-Daten verwendet (Anpassungs-Schicht).
"""
from __future__ import annotations

import json
from collections import defaultdict, deque
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .team_normalize import tournament_weight
from .features_v3 import _team_to_continent, ELOG_START, HOME_ADVANTAGE_ELO
from .fifa_squad import get_squad_lookup, load_squads_2026, _default_squad

REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_RESULTS = REPO_ROOT / "data" / "raw" / "results.csv"
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Bookmaker-odds blend helpers
# ---------------------------------------------------------------------------
_ODDS_CACHE: dict = {}


def _load_odds() -> None:
    """Lazily load wm2026_odds.json. Populates _ODDS_CACHE in-place."""
    if _ODDS_CACHE:
        return
    odds_path = REPO_ROOT / "data" / "raw" / "wm2026_odds.json"
    try:
        with open(odds_path, encoding="utf-8") as fh:
            raw = json.load(fh)
        _ODDS_CACHE["matches"] = raw.get("matches", {})
        _ODDS_CACHE["blend_weight"] = float(raw.get("blend_weight", 0.45))
    except (FileNotFoundError, json.JSONDecodeError):
        _ODDS_CACHE["matches"] = {}
        _ODDS_CACHE["blend_weight"] = 0.45


def _blend_wdl(
    p_model: tuple[float, float, float],
    p_market: tuple[float, float, float],
    w: float,
) -> tuple[float, float, float]:
    """Linear blend (1-w)*model + w*market, renormalized."""
    ph = (1.0 - w) * p_model[0] + w * p_market[0]
    pd_ = (1.0 - w) * p_model[1] + w * p_market[1]
    pa = (1.0 - w) * p_model[2] + w * p_market[2]
    total = ph + pd_ + pa
    if total > 0:
        ph /= total
        pd_ /= total
        pa /= total
    return ph, pd_, pa

STABILITY_SCALE = 2.0


def expected_score(rating_a: float, rating_b: float) -> float:
    return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))


def update_elo(rating_a: float, rating_b: float, score_a: float, k: int = 30):
    ea = expected_score(rating_a, rating_b)
    new_a = rating_a + k * (score_a - ea)
    new_b = rating_b + k * ((1.0 - score_a) - (1.0 - ea))
    return new_a, new_b


def _safe_avg(history, idx, default):
    if not history:
        return default
    vals = [h[idx] for h in history]
    return sum(vals) / len(vals)


def build_feature_table_v6() -> Path:
    print("=" * 70)
    print(" V6 Features: V5 + Kader-Stärke (FIFA-Ratings)")
    print("=" * 70)

    # Load squad lookup
    squad_get, _ = get_squad_lookup()

    df = pd.read_csv(RAW_RESULTS, parse_dates=["date"])
    df = df.dropna(subset=["home_score", "away_score"])
    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)
    df = df.sort_values("date").reset_index(drop=True)
    print(f"   {len(df):,} Spiele gelesen")

    # V5 state variables
    elo: dict[str, float] = defaultdict(lambda: ELOG_START)
    re_elo: dict[str, float] = defaultdict(lambda: ELOG_START)
    vr_elo: dict[str, float] = defaultdict(lambda: ELOG_START)
    form_history: dict[str, deque] = defaultdict(lambda: deque(maxlen=10))
    last_match_date: dict[str, datetime] = {}
    h2h: dict[tuple[str, str], deque] = defaultdict(lambda: deque(maxlen=10))
    win_streak: dict[str, int] = defaultdict(int)
    unbeaten_streak: dict[str, int] = defaultdict(int)
    oppo_elo_history: dict[str, deque] = defaultdict(lambda: deque(maxlen=10))
    weighted_form_history: dict[str, deque] = defaultdict(lambda: deque(maxlen=10))
    last_year_results: dict[str, deque] = defaultdict(lambda: deque(maxlen=20))

    n = len(df)
    rows = []
    for i, row in enumerate(df.itertuples(index=False)):
        home = row.home_team
        away = row.away_team
        date: datetime = row.date
        hs = int(row.home_score)
        as_ = int(row.away_score)
        neutral = bool(row.neutral)
        tournament = row.tournament
        year = date.year

        # --- V5 features (exact copy) ---
        elo_home = elo[home]
        elo_away = elo[away]
        elo_home_eff = elo_home + (0 if neutral else HOME_ADVANTAGE_ELO)
        re_elo_home = re_elo[home]
        re_elo_away = re_elo[away]
        re_elo_home_eff = re_elo_home + (0 if neutral else HOME_ADVANTAGE_ELO)
        vr_elo_home = vr_elo[home]
        vr_elo_away = vr_elo[away]
        vr_elo_home_eff = vr_elo_home + (0 if neutral else HOME_ADVANTAGE_ELO)

        fh = form_history[home]
        fa = form_history[away]
        fl = list(fh)
        fl_a = list(fa)
        form1_h = fl[-1][3] if fl else 1.0
        form1_a = fl_a[-1][3] if fl_a else 1.0
        form2_h = _safe_avg(fl[-2:], 3, 1.0)
        form2_a = _safe_avg(fl_a[-2:], 3, 1.0)
        form3_h = _safe_avg(fl[-3:], 3, 1.0)
        form3_a = _safe_avg(fl_a[-3:], 3, 1.0)
        form5_h = _safe_avg(fh, 3, 1.0)
        form5_a = _safe_avg(fa, 3, 1.0)
        form10_h = _safe_avg(fl, 3, 1.0)
        form10_a = _safe_avg(fl_a, 3, 1.0)

        gf5_h = _safe_avg(fh, 1, 1.0)
        ga5_h = _safe_avg(fh, 2, 1.0)
        gf5_a = _safe_avg(fa, 1, 1.0)
        ga5_a = _safe_avg(fa, 2, 1.0)
        gd5_h = gf5_h - ga5_h
        gd5_a = gf5_a - ga5_a

        rest_h = (date - last_match_date[home]).days if home in last_match_date else 30
        rest_a = (date - last_match_date[away]).days if away in last_match_date else 30
        rest_h = min(rest_h, 365)
        rest_a = min(rest_a, 365)

        key = tuple(sorted([home, away]))
        h2h_hist = list(h2h[key])
        if h2h_hist:
            h2h_home = sum(1 for w in h2h_hist if w == home) / len(h2h_hist)
            h2h_away = sum(1 for w in h2h_hist if w == away) / len(h2h_hist)
        else:
            h2h_home = 0.5
            h2h_away = 0.5

        gd_history_h = [h[1] - h[2] for h in fl[-5:]]
        gd_history_a = [h[1] - h[2] for h in fl_a[-5:]]
        std_h = float(np.std(gd_history_h)) if len(gd_history_h) >= 2 else 0.0
        std_a = float(np.std(gd_history_a)) if len(gd_history_a) >= 2 else 0.0
        stability_h = 1.0 / (1.0 + std_h / STABILITY_SCALE)
        stability_a = 1.0 / (1.0 + std_a / STABILITY_SCALE)

        oppo_elo5_h = sum(oppo_elo_history[home]) / len(oppo_elo_history[home]) if oppo_elo_history[home] else ELOG_START
        oppo_elo5_a = sum(oppo_elo_history[away]) / len(oppo_elo_history[away]) if oppo_elo_history[away] else ELOG_START

        wfh = list(weighted_form_history[home])
        wfa = list(weighted_form_history[away])
        w_form_h = sum(e[0] for e in wfh) / len(wfh) if wfh else 1.0
        w_form_a = sum(e[0] for e in wfa) / len(wfa) if wfa else 1.0

        momentum_h = form3_h - form10_h
        momentum_a = form3_a - form10_a

        one_year_ago = date - pd.Timedelta(days=365)
        recent_results_h = [r for r in last_year_results[home] if r[0] >= one_year_ago]
        recent_results_a = [r for r in last_year_results[away] if r[0] >= one_year_ago]
        wins_top10_h = sum(1 for r in recent_results_h if r[1] >= 1900 and r[2] == "W") / max(len(recent_results_h), 1)
        wins_top20_h = sum(1 for r in recent_results_h if r[1] >= 1850 and r[2] == "W") / max(len(recent_results_h), 1)
        wins_top10_a = sum(1 for r in recent_results_a if r[1] >= 1900 and r[2] == "W") / max(len(recent_results_a), 1)
        wins_top20_a = sum(1 for r in recent_results_a if r[1] >= 1850 and r[2] == "W") / max(len(recent_results_a), 1)

        cont_h = _team_to_continent(home)
        cont_a = _team_to_continent(away)

        # --- V6 NEW: Squad features ---
        sq_h = squad_get(home, year)
        sq_a = squad_get(away, year)
        sq_ovr_h = sq_h["sq_ovr"]
        sq_ovr_a = sq_a["sq_ovr"]
        sq_att_h = sq_h["sq_att"]
        sq_att_a = sq_a["sq_att"]
        sq_def_h = sq_h["sq_def"]
        sq_def_a = sq_a["sq_def"]
        sq_diff = sq_ovr_h - sq_ovr_a
        sq_age_h = sq_h["sq_age"]
        sq_age_a = sq_a["sq_age"]

        # Target
        if hs > as_:
            target = 1
            score_home = 1.0
        elif hs < as_:
            target = 2
            score_home = 0.0
        else:
            target = 0
            score_home = 0.5

        rows.append({
            "date": date,
            "home_team": home,
            "away_team": away,
            # V5 features
            "neutral": int(neutral),
            "tournament_w": tournament_weight(tournament),
            "elo_a": elo_home, "elo_b": elo_away, "elo_diff": elo_home_eff - elo_away,
            "vr_elo_a": vr_elo_home, "vr_elo_b": vr_elo_away, "vr_elo_diff": vr_elo_home_eff - vr_elo_away,
            "re_elo_a": re_elo_home, "re_elo_b": re_elo_away, "re_elo_diff": re_elo_home_eff - re_elo_away,
            "form1_a": form1_h, "form1_b": form1_a,
            "form2_a": form2_h, "form2_b": form2_a,
            "form3_a": form3_h, "form3_b": form3_a,
            "form5_a": form5_h, "form5_b": form5_a,
            "form10_a": form10_h, "form10_b": form10_a,
            "gf5_a": gf5_h, "gf5_b": gf5_a,
            "ga5_a": ga5_h, "ga5_b": ga5_a,
            "gd5_a": gd5_h, "gd5_b": gd5_a,
            "h2h_a": h2h_home, "h2h_b": h2h_away,
            "rest_a": rest_h, "rest_b": rest_a,
            "win_streak_a": win_streak[home], "win_streak_b": win_streak[away],
            "unbeaten_a": unbeaten_streak[home],
            "continent_a": cont_h, "continent_b": cont_a,
            "oppo_elo5_a": oppo_elo5_h, "oppo_elo5_b": oppo_elo5_a,
            "w_form_a": w_form_h, "w_form_b": w_form_a,
            "momentum_a": momentum_h, "momentum_b": momentum_a,
            "wins_top10_a": wins_top10_h, "wins_top10_b": wins_top10_a,
            "wins_top20_a": wins_top20_h, "wins_top20_b": wins_top20_a,
            "stability_a": stability_h, "stability_b": stability_a,
            # V6 NEW
            "sq_ovr_a": sq_ovr_h, "sq_ovr_b": sq_ovr_a,
            "sq_att_a": sq_att_h, "sq_att_b": sq_att_a,
            "sq_def_a": sq_def_h, "sq_def_b": sq_def_a,
            "sq_diff": sq_diff,
            "sq_age_a": sq_age_h, "sq_age_b": sq_age_a,
            # Targets
            "target": target,
            "home_goals": hs,
            "away_goals": as_,
        })

        # Update state (identical to V5)
        k = tournament_weight(tournament)
        new_home, new_away = update_elo(elo_home_eff, elo[away], score_home, k=k)
        if not neutral:
            new_home -= HOME_ADVANTAGE_ELO
        elo[home] = new_home
        elo[away] = new_away

        re_eff = re_elo[home] + (0 if neutral else HOME_ADVANTAGE_ELO)
        re_new_h, re_new_a = update_elo(re_eff, re_elo[away], score_home, k=max(k * 2, 30))
        if not neutral:
            re_new_h -= HOME_ADVANTAGE_ELO
        re_elo[home] = re_new_h
        re_elo[away] = re_new_a

        vr_eff = vr_elo[home] + (0 if neutral else HOME_ADVANTAGE_ELO)
        vr_k = max(k * 4, 60)
        vr_new_h, vr_new_a = update_elo(vr_eff, vr_elo[away], score_home, k=vr_k)
        if not neutral:
            vr_new_h -= HOME_ADVANTAGE_ELO
        vr_elo[home] = vr_new_h
        vr_elo[away] = vr_new_a

        pts_home = 3 if hs > as_ else (1 if hs == as_ else 0)
        pts_away = 3 if as_ > hs else (1 if hs == as_ else 0)
        form_history[home].append((date, hs, as_, pts_home))
        form_history[away].append((date, as_, hs, pts_away))
        last_match_date[home] = date
        last_match_date[away] = date

        if hs > as_:
            winner = home
            res_h, res_a = "W", "L"
        elif hs < as_:
            winner = away
            res_h, res_a = "L", "W"
        else:
            winner = "draw"
            res_h, res_a = "D", "D"
        h2h[key].append(winner)

        if hs > as_:
            win_streak[home] = max(0, win_streak[home]) + 1
            win_streak[away] = min(0, win_streak[away]) - 1
            unbeaten_streak[home] += 1
            unbeaten_streak[away] = 0
        elif hs < as_:
            win_streak[away] = max(0, win_streak[away]) + 1
            win_streak[home] = min(0, win_streak[home]) - 1
            unbeaten_streak[away] += 1
            unbeaten_streak[home] = 0
        else:
            win_streak[home] = 0
            win_streak[away] = 0
            unbeaten_streak[home] += 1
            unbeaten_streak[away] += 1

        oppo_elo_history[home].append(elo[away])
        oppo_elo_history[away].append(elo[home])
        weighted_form_history[home].append((pts_home, elo[away], hs, as_))
        weighted_form_history[away].append((pts_away, elo[home], as_, hs))
        last_year_results[home].append((date, elo[away], res_h))
        last_year_results[away].append((date, elo[home], res_a))

        if (i + 1) % 5000 == 0:
            print(f"   ... {i+1:,}/{n:,} Spiele verarbeitet")

    feat_df = pd.DataFrame(rows)
    feat_df["date"] = pd.to_datetime(feat_df["date"])
    feat_df = feat_df.sort_values("date").reset_index(drop=True)
    print(f"   {len(feat_df):,} Feature-Zeilen erzeugt.")

    out_parquet = PROCESSED_DIR / "features_v6.parquet"
    feat_df.to_parquet(out_parquet, index=False)
    print(f"   geschrieben: {out_parquet}")

    feature_cols = [
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
        # V6 NEW
        "sq_ovr_a", "sq_ovr_b",
        "sq_att_a", "sq_att_b",
        "sq_def_a", "sq_def_b",
        "sq_diff",
        "sq_age_a", "sq_age_b",
    ]

    X = feat_df[feature_cols].astype("float32").to_numpy()
    y = feat_df["target"].astype("int64").to_numpy()
    y_hg = feat_df["home_goals"].astype("float32").to_numpy()
    y_ag = feat_df["away_goals"].astype("float32").to_numpy()
    dates = feat_df["date"].astype("datetime64[ns]").to_numpy()
    home_teams = feat_df["home_team"].to_numpy()
    away_teams = feat_df["away_team"].to_numpy()

    np.savez_compressed(
        PROCESSED_DIR / "features_v6.npz",
        X=X, y=y, y_home_goals=y_hg, y_away_goals=y_ag,
        dates=dates, home=home_teams, away=away_teams,
        feature_names=np.array(feature_cols),
    )
    print(f"   geschrieben: {PROCESSED_DIR / 'features_v6.npz'}  ({len(feature_cols)} Features)")

    import json as _json
    meta = {
        "n_samples": int(len(feat_df)),
        "feature_columns": feature_cols,
        "n_features": len(feature_cols),
        "class_distribution": {int(k): int(v) for k, v in pd.Series(y).value_counts().to_dict().items()},
        "n_teams": int(len(set(home_teams) | set(away_teams))),
    }
    with open(PROCESSED_DIR / "features_v6_meta.json", "w", encoding="utf-8") as fh:
        _json.dump(meta, fh, indent=2, ensure_ascii=False)
    print(f"   Klassenverteilung: {meta['class_distribution']}")
    print("=" * 70)
    return out_parquet


def load_features_v6(split: str = "all", train_start: str = "2018-01-01", val_start: str = "2024-01-01"):
    bundle = np.load(PROCESSED_DIR / "features_v6.npz", allow_pickle=True)
    X = bundle["X"]
    y = bundle["y"]
    y_hg = bundle["y_home_goals"]
    y_ag = bundle["y_away_goals"]
    dates = pd.to_datetime(bundle["dates"])

    if split == "train":
        mask = (dates >= pd.Timestamp(train_start)) & (dates < pd.Timestamp(val_start))
    elif split == "val":
        mask = dates >= pd.Timestamp(val_start)
    elif split == "all":
        mask = np.ones(len(dates), dtype=bool)
    else:
        raise ValueError(f"unknown split: {split}")
    return (X[mask], y[mask], y_hg[mask], y_ag[mask],
            dates[mask], bundle["home"][mask], bundle["away"][mask], bundle["feature_names"])


def get_current_team_ratings_v6() -> dict[str, dict]:
    """
    Aktuelle Team-States für Inference:
    - V5-Zustände (Elo, Form etc.) aus results.csv
    - Kader-Features aus squads_2026.json (Anpassungs-Schicht)
    """
    from .features_v5 import get_current_team_ratings_v5, _compute_final_state_v5
    import pandas as _pd

    # V5 state
    v5_state = get_current_team_ratings_v5()

    # Squad 2026 override layer
    squads = load_squads_2026()

    # Defaults für Teams ohne 2026-Daten (aus nation_strength, neuestes Jahr)
    sq_lookup, _ = get_squad_lookup()

    def _sq_val(sq, key, default):
        import math as _math
        val = sq.get(key, default)
        try:
            f = float(val)
            return f if not _math.isnan(f) else default
        except (TypeError, ValueError):
            return default

    defaults = _default_squad()
    result = {}
    for team, state in v5_state.items():
        sq = squads.get(team) or sq_lookup(team, 2022)
        result[team] = {
            **state,
            "sq_ovr": _sq_val(sq, "sq_ovr", defaults["sq_ovr"]),
            "sq_att": _sq_val(sq, "sq_att", defaults["sq_att"]),
            "sq_def": _sq_val(sq, "sq_def", defaults["sq_def"]),
            "sq_age": _sq_val(sq, "sq_age", defaults["sq_age"]),
            "sq_depth": _sq_val(sq, "sq_depth", defaults["sq_depth"]),
            "coach": sq.get("coach", "") if isinstance(sq, dict) else "",
        }
    return result


def score_grid(lh: float, la: float, rho: float = -0.13, n: int = 10) -> np.ndarray:
    """Bivariate Poisson score-probability grid P[h][a] with Dixon-Coles correction.

    Returns an (n x n) array where P[h][a] = P(home_goals=h, away_goals=a).
    Rows = home goals, columns = away goals.
    """
    lh = max(float(lh), 0.05)
    la = max(float(la), 0.05)

    k = np.arange(n, dtype=np.float64)
    # log-factorial cache for numerically stable Poisson PMF
    logfac = np.zeros(n, dtype=np.float64)
    for i in range(2, n):
        logfac[i] = logfac[i - 1] + np.log(i)

    ph = np.exp(-lh + k * np.log(lh) - logfac)
    pa = np.exp(-la + k * np.log(la) - logfac)

    grid = np.outer(ph, pa)
    # Dixon-Coles correction for low scores
    grid[0, 0] *= max(1.0 - lh * la * rho, 1e-9)
    grid[1, 0] *= max(1.0 + la * rho, 1e-9)
    grid[0, 1] *= max(1.0 + lh * rho, 1e-9)
    grid[1, 1] *= max(1.0 - rho, 1e-9)

    total = grid.sum()
    if total > 0:
        grid /= total
    return grid


def wdl_from_grid(grid: np.ndarray) -> tuple[float, float, float]:
    """Sum W/D/L probabilities from a score-probability grid."""
    n = grid.shape[0]
    idx = np.arange(n)
    h_idx, a_idx = np.meshgrid(idx, idx, indexing="ij")
    p_home = float(grid[h_idx > a_idx].sum())
    p_draw = float(np.diag(grid).sum())
    p_away = float(grid[h_idx < a_idx].sum())
    # Normalize to handle float rounding
    total = p_home + p_draw + p_away
    if total > 0:
        p_home /= total
        p_draw /= total
        p_away /= total
    return p_home, p_draw, p_away


_V6_MODEL_CACHE: dict = {}


def _load_v6_model():
    """Lädt das V7- oder V6-Modell aus dem PyTorch-Checkpoint (cached)."""
    if _V6_MODEL_CACHE:
        return _V6_MODEL_CACHE["model"], _V6_MODEL_CACHE["bundle"]

    import torch
    from .train_v2 import FootballNet, DEVICE
    # Prefer V7 checkpoint if available
    pt_path = REPO_ROOT / "data" / "models" / "v7_latest.pt"
    if not pt_path.exists():
        pt_path = REPO_ROOT / "data" / "models" / "v6_latest.pt"
    if not pt_path.exists():
        raise FileNotFoundError(
            f"Kein Modell-Checkpoint gefunden. Bitte zuerst 'python scripts/train_v7.py' ausführen."
        )
    bundle = torch.load(pt_path, map_location=DEVICE, weights_only=False)

    # Support ensemble: list of state_dicts or single
    state_dicts = bundle.get("state_dicts", [bundle.get("state_dict")])
    models = []
    for sd in state_dicts:
        m = FootballNet(
            in_dim=bundle["in_dim"],
            hidden=bundle["hidden"],
            n_blocks=bundle["n_blocks"],
            dropout=0.0,
        ).to(DEVICE)
        m.load_state_dict(sd)
        m.eval()
        models.append(m)
    bundle["_models"] = models
    _V6_MODEL_CACHE["model"] = models[0]  # compat alias
    _V6_MODEL_CACHE["bundle"] = bundle
    return models[0], bundle


def _build_inference_vector_v6(
    home: str, away: str, neutral: bool, tournament: str, today, state: dict
) -> np.ndarray:
    """Baut den 57-dimensionalen Feature-Vektor für ein hypothetisches Spiel."""
    import pandas as pd
    from .team_normalize import normalize_team_name, tournament_weight
    from .features_v3 import HOME_ADVANTAGE_ELO, ELOG_START
    from datetime import datetime as dt

    home_n = normalize_team_name(home)
    away_n = normalize_team_name(away)
    h = state.get(home_n)
    a = state.get(away_n)
    if h is None or a is None:
        missing = [t for t, s in [(home_n, h), (away_n, a)] if s is None]
        raise ValueError(
            f"Unbekanntes Team: {missing}. Verfügbar: "
            f"{', '.join(sorted(state.keys(), key=lambda t: -state[t]['elo'])[:20])}"
        )

    # Rest days
    rest_h = 30
    rest_a = 30
    if h.get("last_match"):
        last = dt.fromisoformat(h["last_match"])
        rest_h = min((today - last).days, 365)
    if a.get("last_match"):
        last = dt.fromisoformat(a["last_match"])
        rest_a = min((today - last).days, 365)

    # H2H from raw results
    raw = pd.read_csv(RAW_RESULTS, parse_dates=["date"])
    raw = raw.dropna(subset=["home_score", "away_score"])
    pair = raw[
        ((raw["home_team"] == home_n) & (raw["away_team"] == away_n)) |
        ((raw["home_team"] == away_n) & (raw["away_team"] == home_n))
    ].tail(10)
    h2h_h, h2h_a = 0.5, 0.5
    if len(pair) > 0:
        wins_h, wins_a = 0, 0
        for _, r in pair.iterrows():
            if r["home_team"] == home_n:
                if r["home_score"] > r["away_score"]:
                    wins_h += 1
                elif r["home_score"] < r["away_score"]:
                    wins_a += 1
            else:
                if r["away_score"] > r["home_score"]:
                    wins_h += 1
                elif r["away_score"] < r["home_score"]:
                    wins_a += 1
        n = len(pair)
        h2h_h = wins_h / n
        h2h_a = wins_a / n

    elo_h_eff = h["elo"] + (0 if neutral else HOME_ADVANTAGE_ELO)
    re_elo_h_eff = h["re_elo"] + (0 if neutral else HOME_ADVANTAGE_ELO)
    vr_elo_h_eff = h["vr_elo"] + (0 if neutral else HOME_ADVANTAGE_ELO)
    tw = tournament_weight(tournament)

    # Squad features from V6 state
    sq_ovr_h = h.get("sq_ovr", 74.0)
    sq_ovr_a = a.get("sq_ovr", 74.0)

    feature_cols = [
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
    ]
    d = {
        "neutral": int(neutral),
        "tournament_w": tw,
        "elo_a": h["elo"], "elo_b": a["elo"],
        "elo_diff": elo_h_eff - a["elo"],
        "vr_elo_a": h["vr_elo"], "vr_elo_b": a["vr_elo"],
        "vr_elo_diff": vr_elo_h_eff - a["vr_elo"],
        "re_elo_a": h["re_elo"], "re_elo_b": a["re_elo"],
        "re_elo_diff": re_elo_h_eff - a["re_elo"],
        "form1_a": h.get("form1", 1.0), "form1_b": a.get("form1", 1.0),
        "form2_a": h.get("form2", 1.0), "form2_b": a.get("form2", 1.0),
        "form3_a": h.get("form3", 1.0), "form3_b": a.get("form3", 1.0),
        "form5_a": h.get("form5", 1.0), "form5_b": a.get("form5", 1.0),
        "form10_a": h.get("form10", 1.0), "form10_b": a.get("form10", 1.0),
        "gf5_a": h.get("gf5", 1.0), "gf5_b": a.get("gf5", 1.0),
        "ga5_a": h.get("ga5", 1.0), "ga5_b": a.get("ga5", 1.0),
        "gd5_a": h.get("gd5", 0.0), "gd5_b": a.get("gd5", 0.0),
        "h2h_a": h2h_h, "h2h_b": h2h_a,
        "rest_a": rest_h, "rest_b": rest_a,
        "win_streak_a": h.get("win_streak", 0), "win_streak_b": a.get("win_streak", 0),
        "unbeaten_a": h.get("unbeaten", 0),
        "continent_a": _team_to_continent(home_n), "continent_b": _team_to_continent(away_n),
        "oppo_elo5_a": h.get("oppo_elo5", ELOG_START), "oppo_elo5_b": a.get("oppo_elo5", ELOG_START),
        "w_form_a": h.get("w_form", 1.0), "w_form_b": a.get("w_form", 1.0),
        "momentum_a": h.get("momentum", 0.0), "momentum_b": a.get("momentum", 0.0),
        "wins_top10_a": h.get("wins_top10", 0.0), "wins_top10_b": a.get("wins_top10", 0.0),
        "wins_top20_a": h.get("wins_top20", 0.0), "wins_top20_b": a.get("wins_top20", 0.0),
        "stability_a": h.get("stability", 1.0), "stability_b": a.get("stability", 1.0),
        "sq_ovr_a": sq_ovr_h, "sq_ovr_b": sq_ovr_a,
        "sq_att_a": h.get("sq_att", 73.0), "sq_att_b": a.get("sq_att", 73.0),
        "sq_def_a": h.get("sq_def", 73.0), "sq_def_b": a.get("sq_def", 73.0),
        "sq_diff": sq_ovr_h - sq_ovr_a,
        "sq_age_a": h.get("sq_age", 27.0), "sq_age_b": a.get("sq_age", 27.0),
    }
    return np.array([d[k] for k in feature_cols], dtype=np.float32)


def predict_match_v6(
    home: str,
    away: str,
    neutral: bool = True,
    tournament: str = "FIFA World Cup",
    today=None,
) -> dict:
    """Prognose mit dem aktuellen Modell (V7 Poisson oder V6 squad-aware)."""
    import torch
    import torch.nn.functional as F
    from datetime import datetime as dt
    from .team_normalize import normalize_team_name

    if today is None:
        today = dt.now()

    _, bundle = _load_v6_model()
    norm = bundle["norm_stats"]
    mean = np.array(norm["mean"], dtype=np.float32)
    std = np.array(norm["std"], dtype=np.float32)
    is_v7 = bundle.get("model_version", "").startswith("v7")
    dc_rho = float(bundle.get("dc_rho", -0.13))
    models = bundle.get("_models") or []
    if not models:
        m0, _ = _load_v6_model()
        models = bundle.get("_models") or [m0]

    home_n = normalize_team_name(home)
    away_n = normalize_team_name(away)
    state = get_current_team_ratings_v6()

    vec = _build_inference_vector_v6(home_n, away_n, neutral, tournament, today, state)

    # NaN imputation using stored col_medians from training
    col_medians = bundle.get("col_medians")
    if col_medians is not None:
        medians = np.array(col_medians, dtype=np.float32)
        nan_mask = np.isnan(vec)
        if nan_mask.any():
            vec[nan_mask] = medians[nan_mask]
    else:
        nan_mask = np.isnan(vec)
        if nan_mask.any():
            vec[nan_mask] = 0.0

    vec_n = (vec - mean) / std

    from .train_v2 import DEVICE
    x = torch.from_numpy(vec_n).unsqueeze(0).to(DEVICE)

    # Ensemble: average score grids (V7) or logits (V6)
    if is_v7:
        avg_grid = np.zeros((10, 10), dtype=np.float64)
        for m in models:
            with torch.no_grad():
                _, log_lam = m(x)
                log_lam = log_lam.cpu().numpy()[0]
            lh = float(np.exp(log_lam[0]))
            la = float(np.exp(log_lam[1]))
            avg_grid += score_grid(lh, la, dc_rho, n=10)
        avg_grid /= len(models)
        p_home, p_draw, p_away = wdl_from_grid(avg_grid)
        # Expected goals from grid (expectation)
        idx = np.arange(10)
        h_idx, a_idx = np.meshgrid(idx, idx, indexing="ij")
        pred_hg = float((avg_grid * h_idx).sum())
        pred_ag = float((avg_grid * a_idx).sum())
        # Most likely scores from averaged grid
        scores_flat = [
            {"home": int(h), "away": int(a), "prob": float(avg_grid[h, a])}
            for h in range(10) for a in range(10)
        ]
        scores_flat.sort(key=lambda s: -s["prob"])
        most_likely = scores_flat[:5]
        model_ver = f"v7-poisson-ensemble-{len(models)}"
    else:
        gs = bundle["goal_stats"]
        avg_logits = np.zeros(3, dtype=np.float64)
        avg_goals = np.zeros(2, dtype=np.float64)
        for m in models:
            with torch.no_grad():
                logits, goals = m(x)
                avg_logits += F.softmax(logits, dim=1).cpu().numpy()[0]
                avg_goals += goals.cpu().numpy()[0]
        avg_logits /= len(models)
        avg_goals /= len(models)
        p_draw = float(avg_logits[0])
        p_home = float(avg_logits[1])
        p_away = float(avg_logits[2])
        pred_hg = float(np.log1p(np.exp(avg_goals[0] * gs["home_std"] + gs["home_mean"])))
        pred_ag = float(np.log1p(np.exp(avg_goals[1] * gs["away_std"] + gs["away_mean"])))
        from math import exp as _exp, factorial as _fac
        def _pois(k, lam): return _exp(-max(lam, 0.1)) * (max(lam, 0.1) ** k) / _fac(k)
        scores_flat = [{"home": h, "away": a, "prob": _pois(h, pred_hg) * _pois(a, pred_ag)}
                       for h in range(8) for a in range(8)]
        scores_flat.sort(key=lambda s: -s["prob"])
        most_likely = scores_flat[:5]
        model_ver = f"v6-squad-aware-ensemble-{len(models)}"

    # --- Bookmaker-Quoten blend (Gruppenphase only) ---
    odds_blended = False
    market_probs: dict | None = None
    _load_odds()
    lookup_key = f"{home_n}|{away_n}"
    if lookup_key in _ODDS_CACHE.get("matches", {}):
        mo = _ODDS_CACHE["matches"][lookup_key]
        bw = _ODDS_CACHE["blend_weight"]
        p_home, p_draw, p_away = _blend_wdl(
            (p_home, p_draw, p_away),
            (float(mo["home"]), float(mo["draw"]), float(mo["away"])),
            bw,
        )
        odds_blended = True
        market_probs = {"home_win": float(mo["home"]), "draw": float(mo["draw"]), "away_win": float(mo["away"])}

    # --- KickTipp Decision Layer ---
    _odds_for_kt = (
        {"home": float(market_probs["home_win"]),
         "draw": float(market_probs["draw"]),
         "away": float(market_probs["away_win"])}
        if market_probs is not None
        else {"home": p_home, "draw": p_draw, "away": p_away}
    )
    if not is_v7:
        from math import exp as _ex, factorial as _fc
        def _p(k, lam): return _ex(-max(lam, 0.1)) * (max(lam, 0.1) ** k) / _fc(k)
        avg_grid = np.array(
            [[_p(h, pred_hg) * _p(a, pred_ag) for a in range(10)] for h in range(10)],
            dtype=np.float64,
        )
    kicktipp_tip: dict | None = None
    try:
        from .kicktipp import load_scheme as _kt_load_scheme, optimal_tip as _kt_optimal
        _kt = _kt_optimal(avg_grid, _odds_for_kt, _kt_load_scheme())
        kicktipp_tip = {
            "home": _kt["tip"][0],
            "away": _kt["tip"][1],
            "expected_points": _kt["expected_points"],
            "alternatives": _kt["alternatives"][:3],
        }
    except Exception:
        pass

    best_score = most_likely[0]
    german_sentence = (
        f"{home_n} gewinnt zu {p_home*100:.0f}%, "
        f"Unentschieden {p_draw*100:.0f}%, "
        f"{away_n} {p_away*100:.0f}%, "
        f"wahrscheinlichstes Ergebnis {best_score['home']}:{best_score['away']} "
        f"({best_score['prob']*100:.0f}%)"
    )

    h_state = state.get(home_n, {})
    a_state = state.get(away_n, {})

    shootout = None
    try:
        from .shootout_features import predict_shootout
        sp = predict_shootout(home_n, away_n)
        shootout = {
            "home_win_prob": sp["home_win_prob"],
            "away_win_prob": sp["away_win_prob"],
            "home_pen_skill": sp["home_pen_skill"],
            "away_pen_skill": sp["away_pen_skill"],
        }
    except Exception:
        pass

    return {
        "home": home_n,
        "away": away_n,
        "neutral": neutral,
        "tournament": tournament,
        "as_of": today.isoformat(),
        "probabilities": {"draw": p_draw, "home_win": p_home, "away_win": p_away},
        "argmax_label": (
            "Unentschieden" if p_draw >= p_home and p_draw >= p_away
            else (f"Sieg {home_n}" if p_home >= p_away else f"Sieg {away_n}")
        ),
        "expected_score": {
            "home_goals": pred_hg,
            "away_goals": pred_ag,
            "display": f"{pred_hg:.2f} : {pred_ag:.2f}",
        },
        "most_likely_scores": most_likely,
        "german_sentence": german_sentence,
        "elo_home": h_state.get("elo", 1500.0),
        "elo_away": a_state.get("elo", 1500.0),
        "re_elo_home": h_state.get("re_elo", h_state.get("elo", 1500.0)),
        "re_elo_away": a_state.get("re_elo", a_state.get("elo", 1500.0)),
        "sq_ovr_home": h_state.get("sq_ovr", 74.0),
        "sq_ovr_away": a_state.get("sq_ovr", 74.0),
        "coach_home": h_state.get("coach", ""),
        "coach_away": a_state.get("coach", ""),
        "shootout": shootout,
        "model_version": model_ver,
        "ensemble_size": len(models),
        "odds_blended": odds_blended,
        "market_probs": market_probs,
        "kicktipp_tip": kicktipp_tip,
    }


def main() -> int:
    build_feature_table_v6()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
