"""
Feature-Engineering V2 - deutlich mehr Features fuer bessere Accuracy.

Erweiterte Feature-Liste (34 Dimensionen pro Match):

Gruppe 1: Match-Kontext
  0  neutral             - 0/1 neutraler Boden
  1  tournament_w        - K-Faktor: WM=60, EM=50, Friendly=20

Gruppe 2: Elo & Stärke
  2  elo_a               - Elo Team A
  3  elo_b               - Elo Team B
  4  elo_diff            - elo_a - elo_b (mit +80 Heimvorteil)
  5  elo_mom_a           - Elo-Differenz letzte 3 Spiele
  6  elo_mom_b           - Elo-Differenz letzte 3 Spiele

Gruppe 3: Form (3 verschiedene Fenster)
  7  form3_a, form3_b    - Punkte/Spiel letzte 3
  8  form5_a, form5_b    - Punkte/Spiel letzte 5
  9  form10_a, form10_b  - Punkte/Spiel letzte 10

Gruppe 4: Tore
  10 gf5_a, gf5_b        - Tore/Schnitt letzte 5
  11 ga5_a, ga5_b        - Gegentore/Schnitt letzte 5
  12 gd5_a, gd5_b        - Tordifferenz letzte 5

Gruppe 5: Home/Away-Split
  13 home_form5_a        - Form Team A wenn HEIM (letzte 5 Heimspiele)
  14 away_form5_b        - Form Team B wenn AUSWÄRTS (letzte 5 Auswaertsspiele)

Gruppe 6: H2H
  15 h2h_a               - Win-Rate A in letzten direkten Duellen
  16 h2h_b               - Win-Rate B in letzten direkten Duellen

Gruppe 7: Rest & Momentum
  17 rest_a, rest_b      - Tage seit letztem Spiel
  18 win_streak_a        - aktuelle Siegesserie
  19 win_streak_b        - aktuelle Siegesserie
  20 unbeaten_a          - aktuelle Ungeschlagen-Serie

Gruppe 8: Kontinent & Gegnerstärke
  21 continent_a         - 0=UEFA, 1=CONMEBOL, 2=CONCACAF, 3=AFC, 4=CAF, 5=OFC
  22 continent_b         - dito
  23 oppo_elo5_a         - durchschnittlicher Gegner-Elo letzte 5 Spiele A
  24 oppo_elo5_b         - dito B

Gruppe 9: Tournament-Spezifisch
  25 tour_form_a         - Form in gleichem Turnier-Typ (WM-Spiele etc.)
  26 tour_form_b         - dito

Targets:
  - classification: 0=Draw, 1=HomeWin, 2=AwayWin
  - regression: home_goals, away_goals (fuer exakte Score-Prognose)
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
FORM_WINDOWS = (3, 5, 10)
REST_WINDOW_DAYS = 365

# Kontinent-Codes (heuristisch nach haeufigsten Konfoederationen)
CONTINENT_MAP = {
    "UEFA": 0, "CONMEBOL": 1, "CONCACAF": 2, "AFC": 3, "CAF": 4, "OFC": 5,
}


def _team_to_continent(team: str) -> int:
    """Heuristische Zuordnung Team -> Kontinent (0-5)."""
    # Die uebliche Konfoederation der grossen Teams:
    south_america = ["Argentina", "Brazil", "Uruguay", "Colombia", "Chile", "Ecuador",
                     "Paraguay", "Peru", "Venezuela", "Bolivia"]
    concacaf = ["United States", "Mexico", "Canada", "Costa Rica", "Honduras", "Jamaica",
                "Panama", "Trinidad and Tobago", "Haiti", "Cuba", "Curacao"]
    afc = ["Japan", "South Korea", "North Korea", "China PR", "Iran", "IR Iran",
           "Saudi Arabia", "Iraq", "Qatar", "United Arab Emirates", "Uzbekistan",
           "Australia", "Thailand", "Vietnam", "Indonesia", "India"]
    caf = ["Morocco", "Tunisia", "Egypt", "Nigeria", "Ghana", "Cameroon", "Senegal",
           "Algeria", "Ivory Coast", "Mali", "Burkina Faso", "South Africa", "Cape Verde"]
    ifc = ["India"]  # India spielt in AFC
    if team in south_america:
        return 1
    if team in concacaf:
        return 2
    if team in afc:
        return 3
    if team in caf:
        return 4
    # Default: UEFA
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


def build_feature_table() -> Path:
    """Liest results.csv, berechnet Features, schreibt Parquet + Numpy + Meta."""
    print("=" * 70)
    print(" Feature Engineering V2 (32 Features + Score Targets)")
    print("=" * 70)
    print(f">> Lade {RAW_RESULTS}")
    df = pd.read_csv(RAW_RESULTS, parse_dates=["date"])
    df = df.dropna(subset=["home_score", "away_score"])
    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)
    df = df.sort_values("date").reset_index(drop=True)
    dropped_orig = len(df)
    print(f"   {len(df):,} Spiele gelesen ({df['date'].min():%Y-%m-%d} .. {df['date'].max():%Y-%m-%d})")

    # State
    elo: dict[str, float] = defaultdict(lambda: ELOG_START)
    elo_history: dict[str, deque] = defaultdict(lambda: deque(maxlen=20))  # fuer momentum
    form_history: dict[str, deque] = defaultdict(lambda: deque(maxlen=10))
    home_form_history: dict[str, deque] = defaultdict(lambda: deque(maxlen=10))
    away_form_history: dict[str, deque] = defaultdict(lambda: deque(maxlen=10))
    last_match_date: dict[str, datetime] = {}
    h2h: dict[tuple[str, str], deque] = defaultdict(lambda: deque(maxlen=10))
    win_streak: dict[str, int] = defaultdict(int)  # positive = win streak, negative = loss
    unbeaten_streak: dict[str, int] = defaultdict(int)
    last_outcome: dict[str, str] = defaultdict(lambda: "none")
    oppo_elo_history: dict[str, deque] = defaultdict(lambda: deque(maxlen=10))
    tour_form: dict[tuple[str, str], deque] = defaultdict(lambda: deque(maxlen=10))

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
        if not neutral:
            elo_home_eff = elo_home + HOME_ADVANTAGE_ELO
        else:
            elo_home_eff = elo_home

        # Form windows
        fh = form_history[home]
        fa = form_history[away]
        form3_h = _safe_window_avg(list(fh)[-3:], 3, 1.0) if fh else 1.0
        form3_a = _safe_window_avg(list(fa)[-3:], 3, 1.0) if fa else 1.0
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
        away_fa = away_form_history[away]
        home_form5 = _safe_window_avg(home_fh, 3, 1.0)
        away_form5 = _safe_window_avg(away_fa, 3, 1.0)

        # Rest days
        rest_h = (date - last_match_date[home]).days if home in last_match_date else 30
        rest_a = (date - last_match_date[away]).days if away in last_match_date else 30
        rest_h = min(rest_h, REST_WINDOW_DAYS)
        rest_a = min(rest_a, REST_WINDOW_DAYS)

        # H2H
        key = tuple(sorted([home, away]))
        h2h_hist = list(h2h[key])
        if h2h_hist:
            h2h_home = sum(1 for w in h2h_hist if w == home) / len(h2h_hist)
            h2h_away = sum(1 for w in h2h_hist if w == away) / len(h2h_hist)
        else:
            h2h_home = 0.5
            h2h_away = 0.5

        # Elo momentum (Elo-Diff letzte 3)
        eh = list(elo_history[home])
        ea = list(elo_history[away])
        elo_mom_h = (eh[-1] - eh[-3]) if len(eh) >= 3 else 0.0
        elo_mom_a = (ea[-1] - ea[-3]) if len(ea) >= 3 else 0.0

        # Opponent strength
        oe_h = list(oppo_elo_history[home])
        oe_a = list(oppo_elo_history[away])
        oppo_elo5_h = sum(oe_h) / len(oe_h) if oe_h else ELOG_START
        oppo_elo5_a = sum(oe_a) / len(oe_a) if oe_a else ELOG_START

        # Tournament form
        tkey_h = (home, tournament)
        tkey_a = (away, tournament)
        tfh = list(tour_form[tkey_h])
        tfa = list(tour_form[tkey_a])
        tour_form_h = sum(tfh) / len(tfh) if tfh else 1.0
        tour_form_a = sum(tfa) / len(tfa) if tfa else 1.0

        # Target: classification
        if hs > as_:
            target = 1
            score_home = 1.0
        elif hs < as_:
            target = 2
            score_home = 0.0
        else:
            target = 0
            score_home = 0.5

        # Continent
        cont_h = _team_to_continent(home)
        cont_a = _team_to_continent(away)

        rows.append({
            "date": date,
            "home_team": home,
            "away_team": away,
            # Group 1
            "neutral": int(neutral),
            "tournament_w": tournament_weight(tournament),
            # Group 2
            "elo_a": elo_home, "elo_b": elo_away,
            "elo_diff": elo_home_eff - elo_away,
            "elo_mom_a": elo_mom_h, "elo_mom_b": elo_mom_a,
            # Group 3
            "form3_a": form3_h, "form3_b": form3_a,
            "form5_a": form5_h, "form5_b": form5_a,
            "form10_a": form10_h, "form10_b": form10_a,
            # Group 4
            "gf5_a": gf5_h, "gf5_b": gf5_a,
            "ga5_a": ga5_h, "ga5_b": ga5_a,
            "gd5_a": gd5_h, "gd5_b": gd5_a,
            # Group 5
            "home_form5_a": home_form5, "away_form5_b": away_form5,
            # Group 6
            "h2h_a": h2h_home, "h2h_b": h2h_away,
            # Group 7
            "rest_a": rest_h, "rest_b": rest_a,
            "win_streak_a": win_streak[home], "win_streak_b": win_streak[away],
            "unbeaten_a": unbeaten_streak[home],
            # Group 8
            "continent_a": cont_h, "continent_b": cont_a,
            "oppo_elo5_a": oppo_elo5_h, "oppo_elo5_b": oppo_elo5_a,
            # Group 9
            "tour_form_a": tour_form_h, "tour_form_b": tour_form_a,
            # Targets
            "target": target,
            "home_goals": hs,
            "away_goals": as_,
        })

        # ---------- Update state ----------
        # Elo
        k = tournament_weight(tournament)
        new_home, new_away = update_elo(elo_home_eff, elo_away, score_home, k=k)
        if not neutral:
            new_home -= HOME_ADVANTAGE_ELO
        elo[home] = new_home
        elo[away] = new_away
        elo_history[home].append(new_home)
        elo_history[away].append(new_away)

        # Form
        pts_home = 3 if hs > as_ else (1 if hs == as_ else 0)
        pts_away = 3 if as_ > hs else (1 if hs == as_ else 0)
        form_history[home].append((date, hs, as_, pts_home))
        form_history[away].append((date, as_, hs, pts_away))
        if not neutral:
            home_form_history[home].append((date, hs, as_, pts_home))
        else:
            away_form_history[home].append((date, hs, as_, pts_home))  # away appearance for home team
        if not neutral:
            away_form_history[away].append((date, as_, hs, pts_away))  # away appearance for away team
        else:
            home_form_history[away].append((date, as_, hs, pts_away))
        last_match_date[home] = date
        last_match_date[away] = date

        # H2H
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

        # Opponent elo
        oppo_elo_history[home].append(elo_away)
        oppo_elo_history[away].append(elo_home)

        # Tour form
        tour_form[tkey_h].append(pts_home)
        tour_form[tkey_a].append(pts_away)

        if (i + 1) % 5000 == 0:
            print(f"   ... {i+1:,}/{n:,} Spiele verarbeitet")

    feat_df = pd.DataFrame(rows)
    feat_df["date"] = pd.to_datetime(feat_df["date"])
    feat_df = feat_df.sort_values("date").reset_index(drop=True)
    print(f"   {len(feat_df):,} Feature-Zeilen erzeugt.")

    out_parquet = PROCESSED_DIR / "features_v2.parquet"
    feat_df.to_parquet(out_parquet, index=False)
    print(f"   geschrieben: {out_parquet}")

    # Feature columns (ohne Targets und Teamnamen)
    feature_cols = [
        "neutral", "tournament_w",
        "elo_a", "elo_b", "elo_diff", "elo_mom_a", "elo_mom_b",
        "form3_a", "form3_b", "form5_a", "form5_b", "form10_a", "form10_b",
        "gf5_a", "gf5_b", "ga5_a", "ga5_b", "gd5_a", "gd5_b",
        "home_form5_a", "away_form5_b",
        "h2h_a", "h2h_b",
        "rest_a", "rest_b",
        "win_streak_a", "win_streak_b", "unbeaten_a",
        "continent_a", "continent_b", "oppo_elo5_a", "oppo_elo5_b",
        "tour_form_a", "tour_form_b",
    ]
    X = feat_df[feature_cols].astype("float32").to_numpy()
    y = feat_df["target"].astype("int64").to_numpy()
    y_home_goals = feat_df["home_goals"].astype("float32").to_numpy()
    y_away_goals = feat_df["away_goals"].astype("float32").to_numpy()
    dates = feat_df["date"].astype("datetime64[ns]").to_numpy()
    home_teams = feat_df["home_team"].to_numpy()
    away_teams = feat_df["away_team"].to_numpy()

    np.savez_compressed(
        PROCESSED_DIR / "features_v2.npz",
        X=X, y=y, y_home_goals=y_home_goals, y_away_goals=y_away_goals,
        dates=dates, home=home_teams, away=away_teams,
        feature_names=np.array(feature_cols),
    )
    print(f"   geschrieben: {PROCESSED_DIR / 'features_v2.npz'}")

    meta = {
        "n_samples": int(len(feat_df)),
        "feature_columns": feature_cols,
        "date_min": str(feat_df["date"].min()),
        "date_max": str(feat_df["date"].max()),
        "elo_start": ELOG_START,
        "home_advantage_elo": HOME_ADVANTAGE_ELO,
        "form_windows": list(FORM_WINDOWS),
        "class_distribution": {int(k): int(v) for k, v in pd.Series(y).value_counts().to_dict().items()},
        "n_teams": int(len(set(home_teams) | set(away_teams))),
        "home_goals_mean": float(y_home_goals.mean()),
        "away_goals_mean": float(y_away_goals.mean()),
    }
    with open(PROCESSED_DIR / "features_v2_meta.json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2, ensure_ascii=False)
    print(f"   geschrieben: {PROCESSED_DIR / 'features_v2_meta.json'}")
    print(f"   Klassenverteilung: {meta['class_distribution']} (0=Draw, 1=HomeWin, 2=AwayWin)")
    print(f"   Tore: Heim {meta['home_goals_mean']:.2f}, Auswaerts {meta['away_goals_mean']:.2f}")
    print("=" * 70)
    return out_parquet


def load_features_v2(split: str = "all", train_start: str = "1990-01-01", val_start: str = "2024-01-01"):
    """Laedt v2-Features. Liefert X, y_cls, y_home_goals, y_away_goals, dates, teams, names."""
    bundle = np.load(PROCESSED_DIR / "features_v2.npz", allow_pickle=True)
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


def get_current_team_ratings() -> dict[str, dict]:
    """Rekonstruiert die aktuellen Team-States aus den Rohdaten."""
    df = pd.read_csv(RAW_RESULTS, parse_dates=["date"])
    df = df.dropna(subset=["home_score", "away_score"])
    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)
    df = df.sort_values("date").reset_index(drop=True)
    return _compute_final_state(df)


def _compute_final_state(df: pd.DataFrame) -> dict[str, dict]:
    """Berechnet die finalen Team-States (Elo + alle 32 Features)."""
    elo: dict[str, float] = defaultdict(lambda: ELOG_START)
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

    out = {}
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

        out[t] = {
            "elo": float(elo[t]),
            "form3": float(form3), "form5": float(form5), "form10": float(form10),
            "gf5": float(gf5), "ga5": float(ga5), "gd5": float(gd5),
            "home_form5": float(home_form5), "away_form5": float(away_form5),
            "elo_mom": float(elo_mom),
            "win_streak": int(win_streak[t]),
            "unbeaten": int(unbeaten_streak[t]),
            "continent": _team_to_continent(t),
            "oppo_elo5": float(oppo_elo5),
            "last_match": last.isoformat() if last is not None else None,
        }
    return out


def main() -> int:
    build_feature_table()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
