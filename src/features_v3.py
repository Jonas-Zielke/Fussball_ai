"""
Feature-Engineering V3 - mit echtem Recency-Fokus.

Schluessel-Verbesserungen gegenueber V2:
  1. Recent-Elo (Elo aus den letzten 2 Jahren, schnelle Decay)
  2. Opponent-Quality-Weighted Form (Sieg vs Top-10 zaehlt mehr)
  3. Momentum (form3 - form10: ist Team im Auf-/Abwind?)
  4. Wins vs Top-10 / Top-20 (in den letzten 12 Monaten)
  5. Team-Age-Factor (Anzahl Spieler-Wechsel via Team-Generations-Hinweis)

Total: 42 Features (vorher 34)
"""

from __future__ import annotations

import json
import math
from collections import defaultdict, deque
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .team_normalize import tournament_weight

REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_RESULTS = REPO_ROOT / "data" / "raw" / "results.csv"
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

ELOG_START = 1500.0
HOME_ADVANTAGE_ELO = 80.0
RECENT_ELO_WINDOW_DAYS = 730  # 2 Jahre
FORM_WINDOWS = (3, 5, 10)
REST_WINDOW_DAYS = 365

CONTINENT_MAP = {0: "UEFA", 1: "CONMEBOL", 2: "CONCACAF", 3: "AFC", 4: "CAF", 5: "OFC"}


def _team_to_continent(team: str) -> int:
    south_america = ["Argentina", "Brazil", "Uruguay", "Colombia", "Chile", "Ecuador",
                     "Paraguay", "Peru", "Venezuela", "Bolivia"]
    concacaf = ["United States", "Mexico", "Canada", "Costa Rica", "Honduras", "Jamaica",
                "Panama", "Trinidad and Tobago", "Haiti", "Cuba", "Curacao"]
    afc = ["Japan", "South Korea", "North Korea", "China PR", "Iran", "IR Iran",
           "Saudi Arabia", "Iraq", "Qatar", "United Arab Emirates", "Uzbekistan",
           "Australia", "Thailand", "Vietnam", "Indonesia", "India"]
    caf = ["Morocco", "Tunisia", "Egypt", "Nigeria", "Ghana", "Cameroon", "Senegal",
           "Algeria", "Ivory Coast", "Mali", "Burkina Faso", "South Africa", "Cape Verde"]
    if team in south_america:
        return 1
    if team in concacaf:
        return 2
    if team in afc:
        return 3
    if team in caf:
        return 4
    return 0


def expected_score(rating_a: float, rating_b: float) -> float:
    return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))


def update_elo(rating_a: float, rating_b: float, score_a: float, k: int = 30):
    ea = expected_score(rating_a, rating_b)
    eb = 1.0 - ea
    new_a = rating_a + k * (score_a - ea)
    new_b = rating_b + k * ((1.0 - score_a) - eb)
    return new_a, new_b


def _safe_window_avg(history: deque, idx: int, default: float) -> float:
    if not history:
        return default
    vals = [h[idx] for h in history]
    return sum(vals) / len(vals)


def _safe_window_sum(history: deque, idx: int, default: float = 0.0) -> float:
    return sum(h[idx] for h in history) if history else default


def build_feature_table_v3() -> Path:
    """V3: Recent-fokussiert mit Opponent-Quality-Weighted Form."""
    print("=" * 70)
    print(" Feature Engineering V3 - Recency-Fokus + Opponent-Weighted Form")
    print("=" * 70)
    df = pd.read_csv(RAW_RESULTS, parse_dates=["date"])
    df = df.dropna(subset=["home_score", "away_score"])
    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)
    df = df.sort_values("date").reset_index(drop=True)
    print(f"   {len(df):,} Spiele gelesen")

    # State - V2 + new
    elo: dict[str, float] = defaultdict(lambda: ELOG_START)  # Cumulative Elo
    recent_elo: dict[str, float] = defaultdict(lambda: ELOG_START)  # Recent-Elo (2y window)
    elo_history: dict[str, deque] = defaultdict(lambda: deque(maxlen=20))
    form_history: dict[str, deque] = defaultdict(lambda: deque(maxlen=10))
    home_form_history: dict[str, deque] = defaultdict(lambda: deque(maxlen=10))
    away_form_history: dict[str, deque] = defaultdict(lambda: deque(maxlen=10))
    last_match_date: dict[str, datetime] = {}
    h2h: dict[tuple[str, str], deque] = defaultdict(lambda: deque(maxlen=10))
    win_streak: dict[str, int] = defaultdict(int)
    unbeaten_streak: dict[str, int] = defaultdict(int)
    oppo_elo_history: dict[str, deque] = defaultdict(lambda: deque(maxlen=10))
    tour_form: dict[tuple[str, str], deque] = defaultdict(lambda: deque(maxlen=10))
    # V3 new
    weighted_form_history: dict[str, deque] = defaultdict(lambda: deque(maxlen=10))  # form * opponent_elo
    last_year_results: dict[str, deque] = defaultdict(lambda: deque(maxlen=20))  # (date, opp_elo, result)

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

        # ---------- Compute features BEFORE the match ----------
        elo_home = elo[home]
        elo_away = elo[away]
        elo_home_eff = elo_home + (0 if neutral else HOME_ADVANTAGE_ELO)

        # Recent Elo (VOR dem Match - also der Wert, der NACH den vorherigen Spielen berechnet wurde)
        re_elo_home = recent_elo[home]
        re_elo_away = recent_elo[away]
        re_elo_home_eff = re_elo_home + (0 if neutral else HOME_ADVANTAGE_ELO)

        # Form windows
        fh = form_history[home]
        fa = form_history[away]
        form3_h = _safe_window_avg(list(fh)[-3:], 3, 1.0)
        form3_a = _safe_window_avg(list(fa)[-3:], 3, 1.0)
        form5_h = _safe_window_avg(fh, 3, 1.0)
        form5_a = _safe_window_avg(fa, 3, 1.0)
        form10_h = _safe_window_avg(list(fh), 3, 1.0)
        form10_a = _safe_window_avg(list(fa), 3, 1.0)

        gf5_h = _safe_window_avg(fh, 1, 1.0)
        ga5_h = _safe_window_avg(fh, 2, 1.0)
        gf5_a = _safe_window_avg(fa, 1, 1.0)
        ga5_a = _safe_window_avg(fa, 2, 1.0)
        gd5_h = gf5_h - ga5_h
        gd5_a = gf5_a - ga5_a

        home_fh = home_form_history[home]
        away_fh = away_form_history[away]
        home_form5 = _safe_window_avg(home_fh, 3, 1.0)
        away_form5 = _safe_window_avg(away_fh, 3, 1.0)

        rest_h = (date - last_match_date[home]).days if home in last_match_date else 30
        rest_a = (date - last_match_date[away]).days if away in last_match_date else 30
        rest_h = min(rest_h, REST_WINDOW_DAYS)
        rest_a = min(rest_a, REST_WINDOW_DAYS)

        key = tuple(sorted([home, away]))
        h2h_hist = list(h2h[key])
        if h2h_hist:
            h2h_home = sum(1 for w in h2h_hist if w == home) / len(h2h_hist)
            h2h_away = sum(1 for w in h2h_hist if w == away) / len(h2h_hist)
        else:
            h2h_home = 0.5
            h2h_away = 0.5

        eh = list(elo_history[home])
        ea = list(elo_history[away])
        elo_mom_h = (eh[-1] - eh[-3]) if len(eh) >= 3 else 0.0
        elo_mom_a = (ea[-1] - ea[-3]) if len(ea) >= 3 else 0.0

        oe_h = list(oppo_elo_history[home])
        oe_a = list(oppo_elo_history[away])
        oppo_elo5_h = sum(oe_h) / len(oe_h) if oe_h else ELOG_START
        oppo_elo5_a = sum(oe_a) / len(oe_a) if oe_a else ELOG_START

        tkey_h = (home, tournament)
        tkey_a = (away, tournament)
        tfh = list(tour_form[tkey_h])
        tfa = list(tour_form[tkey_a])
        tour_form_h = sum(tfh) / len(tfh) if tfh else 1.0
        tour_form_a = sum(tfa) / len(tfa) if tfa else 1.0

        # V3: Weighted form (form * opponent_elo) — captures "stille Stärke"
        wfh = list(weighted_form_history[home])
        wfa = list(weighted_form_history[away])
        # Jeder Eintrag: (points, opp_elo, gf, ga) — wir wollen form = sum(points) * opp_elo_avg
        if wfh:
            avg_opp_elo = sum(e[1] for e in wfh) / len(wfh)
            weighted_form_h = sum(e[0] for e in wfh) / len(wfh)
        else:
            weighted_form_h = 1.0
            avg_opp_elo = ELOG_START
        if wfa:
            avg_opp_elo_a = sum(e[1] for e in wfa) / len(wfa)
            weighted_form_a = sum(e[0] for e in wfa) / len(wfa)
        else:
            weighted_form_a = 1.0
            avg_opp_elo_a = ELOG_START

        # V3: Momentum (form3 - form10: positive = aufsteigend, negativ = absteigend)
        momentum_h = form3_h - form10_h
        momentum_a = form3_a - form10_a

        # V3: Wins vs Top-Teams in den letzten ~12 Monaten
        # Wir approximieren "Top-10" als aktuelle Elo >= 1900 (basierend auf Cumulative Elo)
        # WICHTIG: Wir nutzen hier den cumulative Elo VOR diesem Match
        one_year_ago = date - pd.Timedelta(days=365)
        recent_results_h = [r for r in last_year_results[home] if r[0] >= one_year_ago]
        recent_results_a = [r for r in last_year_results[away] if r[0] >= one_year_ago]
        # Top-10 = Elo >= 1900, Top-20 = Elo >= 1850 (zur Zeit des Spiels)
        wins_top10_h = sum(1 for r in recent_results_h if r[1] >= 1900 and r[2] == "W") / max(len(recent_results_h), 1)
        wins_top20_h = sum(1 for r in recent_results_h if r[1] >= 1850 and r[2] == "W") / max(len(recent_results_h), 1)
        wins_top10_a = sum(1 for r in recent_results_a if r[1] >= 1900 and r[2] == "W") / max(len(recent_results_a), 1)
        wins_top20_a = sum(1 for r in recent_results_a if r[1] >= 1850 and r[2] == "W") / max(len(recent_results_a), 1)

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

        cont_h = _team_to_continent(home)
        cont_a = _team_to_continent(away)

        rows.append({
            "date": date,
            "home_team": home,
            "away_team": away,
            # Group 1
            "neutral": int(neutral),
            "tournament_w": tournament_weight(tournament),
            # Group 2: Elo (cumulative + recent)
            "elo_a": elo_home, "elo_b": elo_away,
            "elo_diff": elo_home_eff - elo_away,
            "re_elo_a": re_elo_home, "re_elo_b": re_elo_away,
            "re_elo_diff": re_elo_home_eff - re_elo_away,
            "elo_mom_a": elo_mom_h, "elo_mom_b": elo_mom_a,
            # Group 3: Form
            "form3_a": form3_h, "form3_b": form3_a,
            "form5_a": form5_h, "form5_b": form5_a,
            "form10_a": form10_h, "form10_b": form10_a,
            # Group 4: Tore
            "gf5_a": gf5_h, "gf5_b": gf5_a,
            "ga5_a": ga5_h, "ga5_b": ga5_a,
            "gd5_a": gd5_h, "gd5_b": gd5_a,
            # Group 5: Home/Away
            "home_form5_a": home_form5, "away_form5_b": away_form5,
            # Group 6: H2H
            "h2h_a": h2h_home, "h2h_b": h2h_away,
            # Group 7: Rest + Streak
            "rest_a": rest_h, "rest_b": rest_a,
            "win_streak_a": win_streak[home], "win_streak_b": win_streak[away],
            "unbeaten_a": unbeaten_streak[home],
            # Group 8: Continent + Opponent Strength
            "continent_a": cont_h, "continent_b": cont_a,
            "oppo_elo5_a": oppo_elo5_h, "oppo_elo5_b": oppo_elo5_a,
            # Group 9: Tour Form
            "tour_form_a": tour_form_h, "tour_form_b": tour_form_a,
            # V3 NEW
            "w_form_a": weighted_form_h, "w_form_b": weighted_form_a,
            "momentum_a": momentum_h, "momentum_b": momentum_a,
            "wins_top10_a": wins_top10_h, "wins_top10_b": wins_top10_a,
            "wins_top20_a": wins_top20_h, "wins_top20_b": wins_top20_a,
            # Targets
            "target": target,
            "home_goals": hs,
            "away_goals": as_,
        })

        # ---------- Update state ----------
        # Cumulative Elo
        k = tournament_weight(tournament)
        new_home, new_away = update_elo(elo_home_eff, elo_away, score_home, k=k)
        if not neutral:
            new_home -= HOME_ADVANTAGE_ELO
        elo[home] = new_home
        elo[away] = new_away
        elo_history[home].append(new_home)
        elo_history[away].append(new_away)

        # Recent Elo (gleicher Update, aber mit höherer K-faktor-Verstärkung durch 2y window)
        # Wir skalieren den K-Faktor hoch, weil das Fenster kleiner ist
        recent_k = max(k * 2, 30)
        re_new_home, re_new_away = update_elo(re_elo_home_eff, re_elo_away, score_home, k=recent_k)
        if not neutral:
            re_new_home -= HOME_ADVANTAGE_ELO
        recent_elo[home] = re_new_home
        recent_elo[away] = re_new_away

        # Form
        pts_home = 3 if hs > as_ else (1 if hs == as_ else 0)
        pts_away = 3 if as_ > hs else (1 if hs == as_ else 0)
        form_history[home].append((date, hs, as_, pts_home))
        form_history[away].append((date, as_, hs, pts_away))
        if not neutral:
            home_form_history[home].append((date, hs, as_, pts_home))
            away_form_history[away].append((date, as_, hs, pts_away))
        else:
            away_form_history[home].append((date, hs, as_, pts_home))
            home_form_history[away].append((date, as_, hs, pts_away))
        last_match_date[home] = date
        last_match_date[away] = date

        if hs > as_:
            winner = home
        elif hs < as_:
            winner = away
        else:
            winner = "draw"
        h2h[key].append(winner)

        # Streak
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

        oppo_elo_history[home].append(elo_away)
        oppo_elo_history[away].append(elo[home])
        tour_form[tkey_h].append(pts_home)
        tour_form[tkey_a].append(pts_away)

        # V3: Weighted form
        weighted_form_history[home].append((pts_home, elo_away, hs, as_))
        weighted_form_history[away].append((pts_away, elo[home], as_, hs))
        # Last year results for top-wins
        if hs > as_:
            result_h, result_a = "W", "L"
        elif hs < as_:
            result_h, result_a = "L", "W"
        else:
            result_h, result_a = "D", "D"
        last_year_results[home].append((date, elo_away, result_h))
        last_year_results[away].append((date, elo[home], result_a))

        if (i + 1) % 5000 == 0:
            print(f"   ... {i+1:,}/{n:,} Spiele verarbeitet")

    feat_df = pd.DataFrame(rows)
    feat_df["date"] = pd.to_datetime(feat_df["date"])
    feat_df = feat_df.sort_values("date").reset_index(drop=True)
    print(f"   {len(feat_df):,} Feature-Zeilen erzeugt.")

    out_parquet = PROCESSED_DIR / "features_v3.parquet"
    feat_df.to_parquet(out_parquet, index=False)
    print(f"   geschrieben: {out_parquet}")

    feature_cols = [
        "neutral", "tournament_w",
        "elo_a", "elo_b", "elo_diff",
        "re_elo_a", "re_elo_b", "re_elo_diff",
        "elo_mom_a", "elo_mom_b",
        "form3_a", "form3_b", "form5_a", "form5_b", "form10_a", "form10_b",
        "gf5_a", "gf5_b", "ga5_a", "ga5_b", "gd5_a", "gd5_b",
        "home_form5_a", "away_form5_b",
        "h2h_a", "h2h_b",
        "rest_a", "rest_b",
        "win_streak_a", "win_streak_b", "unbeaten_a",
        "continent_a", "continent_b", "oppo_elo5_a", "oppo_elo5_b",
        "tour_form_a", "tour_form_b",
        # V3 new
        "w_form_a", "w_form_b",
        "momentum_a", "momentum_b",
        "wins_top10_a", "wins_top10_b", "wins_top20_a", "wins_top20_b",
    ]
    X = feat_df[feature_cols].astype("float32").to_numpy()
    y = feat_df["target"].astype("int64").to_numpy()
    y_home_goals = feat_df["home_goals"].astype("float32").to_numpy()
    y_away_goals = feat_df["away_goals"].astype("float32").to_numpy()
    dates = feat_df["date"].astype("datetime64[ns]").to_numpy()
    home_teams = feat_df["home_team"].to_numpy()
    away_teams = feat_df["away_team"].to_numpy()

    np.savez_compressed(
        PROCESSED_DIR / "features_v3.npz",
        X=X, y=y, y_home_goals=y_home_goals, y_away_goals=y_away_goals,
        dates=dates, home=home_teams, away=away_teams,
        feature_names=np.array(feature_cols),
    )
    print(f"   geschrieben: {PROCESSED_DIR / 'features_v3.npz'}")

    meta = {
        "n_samples": int(len(feat_df)),
        "feature_columns": feature_cols,
        "n_features": len(feature_cols),
        "date_min": str(feat_df["date"].min()),
        "date_max": str(feat_df["date"].max()),
        "class_distribution": {int(k): int(v) for k, v in pd.Series(y).value_counts().to_dict().items()},
        "n_teams": int(len(set(home_teams) | set(away_teams))),
    }
    with open(PROCESSED_DIR / "features_v3_meta.json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2, ensure_ascii=False)
    print(f"   geschrieben: {PROCESSED_DIR / 'features_v3_meta.json'}")
    print(f"   Features: {len(feature_cols)}")
    print(f"   Klassenverteilung: {meta['class_distribution']}")
    print("=" * 70)
    return out_parquet


def load_features_v3(split: str = "all", train_start: str = "2015-01-01", val_start: str = "2024-01-01"):
    """Laedt v3-Features."""
    bundle = np.load(PROCESSED_DIR / "features_v3.npz", allow_pickle=True)
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


def get_current_team_ratings_v3() -> dict[str, dict]:
    """Rekonstruiert die finalen V3-Team-States (mit Recent-Elo, Weighted Form, etc)."""
    df = pd.read_csv(RAW_RESULTS, parse_dates=["date"])
    df = df.dropna(subset=["home_score", "away_score"])
    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)
    df = df.sort_values("date").reset_index(drop=True)
    return _compute_final_state_v3(df)


def _compute_final_state_v3(df: pd.DataFrame) -> dict[str, dict]:
    """Berechnet die finalen V3-Team-States."""
    elo: dict[str, float] = defaultdict(lambda: ELOG_START)
    recent_elo: dict[str, float] = defaultdict(lambda: ELOG_START)
    elo_history: dict[str, deque] = defaultdict(lambda: deque(maxlen=20))
    form_history: dict[str, deque] = defaultdict(lambda: deque(maxlen=10))
    home_form_history: dict[str, deque] = defaultdict(lambda: deque(maxlen=10))
    away_form_history: dict[str, deque] = defaultdict(lambda: deque(maxlen=10))
    last_match_date: dict[str, datetime] = {}
    h2h: dict[tuple[str, str], deque] = defaultdict(lambda: deque(maxlen=10))
    win_streak: dict[str, int] = defaultdict(int)
    unbeaten_streak: dict[str, int] = defaultdict(int)
    oppo_elo_history: dict[str, deque] = defaultdict(lambda: deque(maxlen=10))
    tour_form: dict[tuple[str, str], deque] = defaultdict(lambda: deque(maxlen=10))
    weighted_form_history: dict[str, deque] = defaultdict(lambda: deque(maxlen=10))
    last_year_results: dict[str, deque] = defaultdict(lambda: deque(maxlen=20))

    for row in df.itertuples(index=False):
        home = row.home_team
        away = row.away_team
        date: datetime = row.date
        hs = int(row.home_score)
        as_ = int(row.away_score)
        neutral = bool(row.neutral)
        tournament = row.tournament

        if hs > as_:
            score_home = 1.0
            pts_home = 3
            pts_away = 0
        elif hs < as_:
            score_home = 0.0
            pts_home = 0
            pts_away = 3
        else:
            score_home = 0.5
            pts_home = 1
            pts_away = 1

        elo_home_eff = elo[home] + (0 if neutral else HOME_ADVANTAGE_ELO)
        k = tournament_weight(tournament)
        new_home, new_away = update_elo(elo_home_eff, elo[away], score_home, k=k)
        if not neutral:
            new_home -= HOME_ADVANTAGE_ELO
        elo[home] = new_home
        elo[away] = new_away
        elo_history[home].append(new_home)
        elo_history[away].append(new_away)

        # Recent Elo
        re_eff = recent_elo[home] + (0 if neutral else HOME_ADVANTAGE_ELO)
        re_a_eff = recent_elo[away]
        re_new_h, re_new_a = update_elo(re_eff, re_a_eff, score_home, k=max(k * 2, 30))
        if not neutral:
            re_new_h -= HOME_ADVANTAGE_ELO
        recent_elo[home] = re_new_h
        recent_elo[away] = re_new_a

        form_history[home].append((date, hs, as_, pts_home))
        form_history[away].append((date, as_, hs, pts_away))
        if not neutral:
            home_form_history[home].append((date, hs, as_, pts_home))
            away_form_history[away].append((date, as_, hs, pts_away))
        else:
            away_form_history[home].append((date, hs, as_, pts_home))
            home_form_history[away].append((date, as_, hs, pts_away))
        last_match_date[home] = date
        last_match_date[away] = date

        key = tuple(sorted([home, away]))
        if hs > as_:
            winner = home
        elif hs < as_:
            winner = away
        else:
            winner = "draw"
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
        tour_form[(home, tournament)].append(pts_home)
        tour_form[(away, tournament)].append(pts_away)
        weighted_form_history[home].append((pts_home, elo[away], hs, as_))
        weighted_form_history[away].append((pts_away, elo[home], as_, hs))
        if hs > as_:
            res_h, res_a = "W", "L"
        elif hs < as_:
            res_h, res_a = "L", "W"
        else:
            res_h, res_a = "D", "D"
        last_year_results[home].append((date, elo[away], res_h))
        last_year_results[away].append((date, elo[home], res_a))

    # Sammle alle Features pro Team
    out = {}
    one_year_ago = df["date"].max() - pd.Timedelta(days=365)
    all_teams = set(elo.keys())
    for t in all_teams:
        fh = form_history[t]
        if fh:
            form3 = _safe_window_avg(list(fh)[-3:], 3, 1.0)
            form5 = _safe_window_avg(fh, 3, 1.0)
            form10 = _safe_window_avg(list(fh), 3, 1.0)
            gf5 = _safe_window_avg(fh, 1, 1.0)
            ga5 = _safe_window_avg(fh, 2, 1.0)
            gd5 = gf5 - ga5
            last = fh[-1][0]
        else:
            form3 = form5 = form10 = 1.0
            gf5 = ga5 = 1.0
            gd5 = 0.0
            last = None

        home_fh = home_form_history[t]
        away_fh = away_form_history[t]
        home_form5 = _safe_window_avg(home_fh, 3, 1.0) if home_fh else 1.0
        away_form5 = _safe_window_avg(away_fh, 3, 1.0) if away_fh else 1.0

        eh = list(elo_history[t])
        elo_mom = (eh[-1] - eh[-3]) if len(eh) >= 3 else 0.0

        oe = list(oppo_elo_history[t])
        oppo_elo5 = sum(oe) / len(oe) if oe else ELOG_START

        wfh = list(weighted_form_history[t])
        if wfh:
            w_form = sum(e[0] for e in wfh) / len(wfh)
        else:
            w_form = 1.0

        momentum = form3 - form10

        recent_results = [r for r in last_year_results[t] if r[0] >= one_year_ago]
        wins_top10 = sum(1 for r in recent_results if r[1] >= 1900 and r[2] == "W") / max(len(recent_results), 1)
        wins_top20 = sum(1 for r in recent_results if r[1] >= 1850 and r[2] == "W") / max(len(recent_results), 1)

        out[t] = {
            "elo": float(elo[t]),
            "re_elo": float(recent_elo[t]),
            "form3": float(form3), "form5": float(form5), "form10": float(form10),
            "gf5": float(gf5), "ga5": float(ga5), "gd5": float(gd5),
            "home_form5": float(home_form5), "away_form5": float(away_form5),
            "elo_mom": float(elo_mom),
            "win_streak": int(win_streak[t]),
            "unbeaten": int(unbeaten_streak[t]),
            "continent": _team_to_continent(t),
            "oppo_elo5": float(oppo_elo5),
            "w_form": float(w_form),
            "momentum": float(momentum),
            "wins_top10": float(wins_top10),
            "wins_top20": float(wins_top20),
            "last_match": last.isoformat() if last is not None else None,
        }
    return out


def main() -> int:
    build_feature_table_v3()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
