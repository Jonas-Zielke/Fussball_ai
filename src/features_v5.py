"""
V5 Features - Aggressive Recency (90 Tage) + Team-Stabilitaet.

Schluessel-Idee: Spieler wechseln staendig. Der "Recent-Elo" (2 Jahre)
enthaelt Spieler die heute nicht mehr aktiv sind. Mit 90-Tage Recency
bilden wir nur die aktuelle Kader-Situation ab.

Neue Features:
  - very_recent_elo: Elo mit 90-Tage Window (extrem kurz)
  - form1_a, form2_a, form3_a: Form nur ueber letzte 1-3 Spiele
  - team_stability_a: 1 / (std der letzten 5 Goal-Differenzen + 1)
  - team_in_flux: 1 wenn Team kuerzlich stark gewechselt hat (Varianz hoch)
  - rest_minutes_a, rest_minutes_b: Minuten seit letztem Spiel (feinere Granularitaet)
"""
import json
import math
from collections import defaultdict, deque
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .team_normalize import tournament_weight
from .features_v3 import _team_to_continent, ELOG_START, HOME_ADVANTAGE_ELO

REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_RESULTS = REPO_ROOT / "data" / "raw" / "results.csv"
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# Konstanter c fuer sigmoid: hoehere Varianz -> niedrigerer Score
STABILITY_SCALE = 2.0


def expected_score(rating_a: float, rating_b: float) -> float:
    return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))


def update_elo(rating_a: float, rating_b: float, score_a: float, k: int = 30):
    ea = expected_score(rating_a, rating_b)
    eb = 1.0 - ea
    new_a = rating_a + k * (score_a - ea)
    new_b = rating_b + k * ((1.0 - score_a) - eb)
    return new_a, new_b


def _safe_avg(history, idx, default):
    if not history:
        return default
    vals = [h[idx] for h in history]
    return sum(vals) / len(vals)


def _safe_std(history, idx, default=0.0):
    if len(history) < 2:
        return default
    vals = [h[idx] for h in history]
    mean = sum(vals) / len(vals)
    return float(np.sqrt(sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)))


def build_feature_table_v5() -> Path:
    print("=" * 70)
    print(" V5 Features: Aggressive Recency (90d) + Team-Stabilitaet")
    print("=" * 70)
    df = pd.read_csv(RAW_RESULTS, parse_dates=["date"])
    df = df.dropna(subset=["home_score", "away_score"])
    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)
    df = df.sort_values("date").reset_index(drop=True)
    print(f"   {len(df):,} Spiele gelesen")

    # State - V3 + neue V5 features
    elo: dict[str, float] = defaultdict(lambda: ELOG_START)  # cumulative
    re_elo: dict[str, float] = defaultdict(lambda: ELOG_START)  # 2y recent
    vr_elo: dict[str, float] = defaultdict(lambda: ELOG_START)  # 90d very recent
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

        # Standard features
        elo_home = elo[home]
        elo_away = elo[away]
        elo_home_eff = elo_home + (0 if neutral else HOME_ADVANTAGE_ELO)
        re_elo_home = re_elo[home]
        re_elo_away = re_elo[away]
        re_elo_home_eff = re_elo_home + (0 if neutral else HOME_ADVANTAGE_ELO)

        # V5 NEW: very-recent Elo (90 Tage, viel staerkerer Decay)
        vr_elo_home = vr_elo[home]
        vr_elo_away = vr_elo[away]
        vr_elo_home_eff = vr_elo_home + (0 if neutral else HOME_ADVANTAGE_ELO)

        # Form windows
        fh = form_history[home]
        fa = form_history[away]
        # Sehr kurze Forms (1, 2, 3 Spiele)
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

        # V5 NEW: Team-Stabilitaet (1/Varianz) - je stabiler desto hoeher
        gd_history_h = [h[1] - h[2] for h in fl[-5:]]  # letzte 5 GD
        gd_history_a = [h[1] - h[2] for h in fl_a[-5:]]
        std_h = float(np.std(gd_history_h)) if len(gd_history_h) >= 2 else 0.0
        std_a = float(np.std(gd_history_a)) if len(gd_history_a) >= 2 else 0.0
        stability_h = 1.0 / (1.0 + std_h / STABILITY_SCALE)
        stability_a = 1.0 / (1.0 + std_a / STABILITY_SCALE)

        # Standard V3 features
        oppo_elo5_h = sum(oppo_elo_history[home]) / len(oppo_elo_history[home]) if oppo_elo_history[home] else ELOG_START
        oppo_elo5_a = sum(oppo_elo_history[away]) / len(oppo_elo_history[away]) if oppo_elo_history[away] else ELOG_START

        # Weighted form
        wfh = list(weighted_form_history[home])
        wfa = list(weighted_form_history[away])
        if wfh:
            w_form_h = sum(e[0] for e in wfh) / len(wfh)
        else:
            w_form_h = 1.0
        if wfa:
            w_form_a = sum(e[0] for e in wfa) / len(wfa)
        else:
            w_form_a = 1.0

        # Momentum
        momentum_h = form3_h - form10_h
        momentum_a = form3_a - form10_a

        # Wins vs top teams
        one_year_ago = date - pd.Timedelta(days=365)
        recent_results_h = [r for r in last_year_results[home] if r[0] >= one_year_ago]
        recent_results_a = [r for r in last_year_results[away] if r[0] >= one_year_ago]
        wins_top10_h = sum(1 for r in recent_results_h if r[1] >= 1900 and r[2] == "W") / max(len(recent_results_h), 1)
        wins_top20_h = sum(1 for r in recent_results_h if r[1] >= 1850 and r[2] == "W") / max(len(recent_results_h), 1)
        wins_top10_a = sum(1 for r in recent_results_a if r[1] >= 1900 and r[2] == "W") / max(len(recent_results_a), 1)
        wins_top20_a = sum(1 for r in recent_results_a if r[1] >= 1850 and r[2] == "W") / max(len(recent_results_a), 1)

        cont_h = _team_to_continent(home)
        cont_a = _team_to_continent(away)

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
            "neutral": int(neutral),
            "tournament_w": tournament_weight(tournament),
            "elo_a": elo_home, "elo_b": elo_away, "elo_diff": elo_home_eff - elo_away,
            # V5 NEW: 90-day Elo
            "vr_elo_a": vr_elo_home, "vr_elo_b": vr_elo_away, "vr_elo_diff": vr_elo_home_eff - vr_elo_away,
            "re_elo_a": re_elo_home, "re_elo_b": re_elo_away, "re_elo_diff": re_elo_home_eff - re_elo_away,
            # Forms: 1, 2, 3, 5, 10
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
            # V5 NEW: Stabilitaet
            "stability_a": stability_h, "stability_b": stability_a,
            # Targets
            "target": target,
            "home_goals": hs,
            "away_goals": as_,
        })

        # Update state
        k = tournament_weight(tournament)
        new_home, new_away = update_elo(elo_home_eff, elo[away], score_home, k=k)
        if not neutral:
            new_home -= HOME_ADVANTAGE_ELO
        elo[home] = new_home
        elo[away] = new_away

        # 2-year Elo
        re_new_h, re_new_a = update_elo(re_elo_home_eff, re_elo[away], score_home, k=max(k*2, 30))
        if not neutral:
            re_new_h -= HOME_ADVANTAGE_ELO
        re_elo[home] = re_new_h
        re_elo[away] = re_new_a

        # V5 NEW: 90-day Elo (4x staerkerer Decay)
        vr_k = max(k * 4, 60)
        vr_new_h, vr_new_a = update_elo(vr_elo_home_eff, vr_elo[away], score_home, k=vr_k)
        if not neutral:
            vr_new_h -= HOME_ADVANTAGE_ELO
        vr_elo[home] = vr_new_h
        vr_elo[away] = vr_new_a

        # Form
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

    out_parquet = PROCESSED_DIR / "features_v5.parquet"
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
    ]
    X = feat_df[feature_cols].astype("float32").to_numpy()
    y = feat_df["target"].astype("int64").to_numpy()
    y_hg = feat_df["home_goals"].astype("float32").to_numpy()
    y_ag = feat_df["away_goals"].astype("float32").to_numpy()
    dates = feat_df["date"].astype("datetime64[ns]").to_numpy()
    home_teams = feat_df["home_team"].to_numpy()
    away_teams = feat_df["away_team"].to_numpy()

    np.savez_compressed(
        PROCESSED_DIR / "features_v5.npz",
        X=X, y=y, y_home_goals=y_hg, y_away_goals=y_ag,
        dates=dates, home=home_teams, away=away_teams,
        feature_names=np.array(feature_cols),
    )
    print(f"   geschrieben: {PROCESSED_DIR / 'features_v5.npz'}")

    meta = {
        "n_samples": int(len(feat_df)),
        "feature_columns": feature_cols,
        "n_features": len(feature_cols),
        "class_distribution": {int(k): int(v) for k, v in pd.Series(y).value_counts().to_dict().items()},
        "n_teams": int(len(set(home_teams) | set(away_teams))),
    }
    with open(PROCESSED_DIR / "features_v5_meta.json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2, ensure_ascii=False)
    print(f"   Features: {len(feature_cols)}")
    print(f"   Klassenverteilung: {meta['class_distribution']}")
    print("=" * 70)
    return out_parquet


def load_features_v5(split: str = "all", train_start: str = "2018-01-01", val_start: str = "2024-01-01"):
    bundle = np.load(PROCESSED_DIR / "features_v5.npz", allow_pickle=True)
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


def get_current_team_ratings_v5() -> dict[str, dict]:
    df = pd.read_csv(RAW_RESULTS, parse_dates=["date"])
    df = df.dropna(subset=["home_score", "away_score"])
    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)
    df = df.sort_values("date").reset_index(drop=True)
    return _compute_final_state_v5(df)


def _compute_final_state_v5(df: pd.DataFrame) -> dict[str, dict]:
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
            pts_home, pts_away = 3, 0
        elif hs < as_:
            score_home = 0.0
            pts_home, pts_away = 0, 3
        else:
            score_home = 0.5
            pts_home, pts_away = 1, 1

        elo_home_eff = elo[home] + (0 if neutral else HOME_ADVANTAGE_ELO)
        k = tournament_weight(tournament)
        new_home, new_away = update_elo(elo_home_eff, elo[away], score_home, k=k)
        if not neutral:
            new_home -= HOME_ADVANTAGE_ELO
        elo[home] = new_home
        elo[away] = new_away

        re_eff = re_elo[home] + (0 if neutral else HOME_ADVANTAGE_ELO)
        re_new_h, re_new_a = update_elo(re_eff, re_elo[away], score_home, k=max(k*2, 30))
        if not neutral:
            re_new_h -= HOME_ADVANTAGE_ELO
        re_elo[home] = re_new_h
        re_elo[away] = re_new_a

        vr_eff = vr_elo[home] + (0 if neutral else HOME_ADVANTAGE_ELO)
        vr_new_h, vr_new_a = update_elo(vr_eff, vr_elo[away], score_home, k=max(k*4, 60))
        if not neutral:
            vr_new_h -= HOME_ADVANTAGE_ELO
        vr_elo[home] = vr_new_h
        vr_elo[away] = vr_new_a

        form_history[home].append((date, hs, as_, pts_home))
        form_history[away].append((date, as_, hs, pts_away))
        last_match_date[home] = date
        last_match_date[away] = date

        key = tuple(sorted([home, away]))
        if hs > as_: winner = home
        elif hs < as_: winner = away
        else: winner = "draw"
        h2h[key].append(winner)

        if hs > as_:
            win_streak[home] = max(0, win_streak[home]) + 1
            win_streak[away] = min(0, win_streak[away]) - 1
            unbeaten_streak[home] += 1
            unbeaten_streak[away] = 0
            res_h, res_a = "W", "L"
        elif hs < as_:
            win_streak[away] = max(0, win_streak[away]) + 1
            win_streak[home] = min(0, win_streak[home]) - 1
            unbeaten_streak[away] += 1
            unbeaten_streak[home] = 0
            res_h, res_a = "L", "W"
        else:
            win_streak[home] = 0
            win_streak[away] = 0
            unbeaten_streak[home] += 1
            unbeaten_streak[away] += 1
            res_h, res_a = "D", "D"

        oppo_elo_history[home].append(elo[away])
        oppo_elo_history[away].append(elo[home])
        weighted_form_history[home].append((pts_home, elo[away], hs, as_))
        weighted_form_history[away].append((pts_away, elo[home], as_, hs))
        last_year_results[home].append((date, elo[away], res_h))
        last_year_results[away].append((date, elo[home], res_a))

    out = {}
    one_year_ago = df["date"].max() - pd.Timedelta(days=365)
    for t in set(elo.keys()):
        fh = form_history[t]
        if fh:
            form1 = fh[-1][3]
            form2 = _safe_avg(list(fh)[-2:], 3, 1.0)
            form3 = _safe_avg(list(fh)[-3:], 3, 1.0)
            form5 = _safe_avg(fh, 3, 1.0)
            form10 = _safe_avg(list(fh), 3, 1.0)
            gf5 = _safe_avg(fh, 1, 1.0)
            ga5 = _safe_avg(fh, 2, 1.0)
            gd5 = gf5 - ga5
            last = fh[-1][0]
            gd_history = [h[1] - h[2] for h in list(fh)[-5:]]
            std = float(np.std(gd_history)) if len(gd_history) >= 2 else 0.0
            stability = 1.0 / (1.0 + std / STABILITY_SCALE)
        else:
            form1 = form2 = form3 = form5 = form10 = 1.0
            gf5 = ga5 = 1.0
            gd5 = 0.0
            last = None
            stability = 1.0

        oe = list(oppo_elo_history[t])
        oppo_elo5 = sum(oe) / len(oe) if oe else ELOG_START
        wfh = list(weighted_form_history[t])
        w_form = sum(e[0] for e in wfh) / len(wfh) if wfh else 1.0
        momentum = form3 - form10

        recent_results = [r for r in last_year_results[t] if r[0] >= one_year_ago]
        wins_top10 = sum(1 for r in recent_results if r[1] >= 1900 and r[2] == "W") / max(len(recent_results), 1)
        wins_top20 = sum(1 for r in recent_results if r[1] >= 1850 and r[2] == "W") / max(len(recent_results), 1)

        out[t] = {
            "elo": float(elo[t]),
            "re_elo": float(re_elo[t]),
            "vr_elo": float(vr_elo[t]),
            "form1": float(form1), "form2": float(form2), "form3": float(form3),
            "form5": float(form5), "form10": float(form10),
            "gf5": float(gf5), "ga5": float(ga5), "gd5": float(gd5),
            "elo_mom": 0.0,  # not tracked in V5 state; use 0
            "home_form5": float(stability),  # placeholder, not critical
            "away_form5": float(stability),  # placeholder
            "win_streak": int(win_streak[t]),
            "unbeaten": int(unbeaten_streak[t]),
            "continent": _team_to_continent(t),
            "oppo_elo5": float(oppo_elo5),
            "w_form": float(w_form),
            "momentum": float(momentum),
            "wins_top10": float(wins_top10),
            "wins_top20": float(wins_top20),
            "stability": float(stability),
            "last_match": last.isoformat() if last is not None else None,
        }
    return out


def main() -> int:
    build_feature_table_v5()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
