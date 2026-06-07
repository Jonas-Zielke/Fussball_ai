"""
Feature-Engineering fuer den WM 2026 Predictor.

Pro Spiel berechnen wir einen Feature-Vektor aus Sicht von Team A (home_team):

Numerische Features:
    1. elo_a              - Elo-Rating Team A VOR dem Spiel
    2. elo_b              - Elo-Rating Team B VOR dem Spiel
    3. elo_diff           - elo_a - elo_b
    4. form5_a            - Punkte pro Spiel der letzten 5 Spiele von Team A
    5. form5_b            - ... Team B
    6. gf5_a, ga5_a       - Tore geschossen/kassiert Schnitt letzte 5 Spiele A
    7. gf5_b, ga5_b       - ... Team B
    8. rest_a, rest_b     - Tage seit letztem Spiel A bzw. B
    9. neutral            - 0/1 (1 = neutraler Boden)
   10. tournament_w       - Wichtigkeit des Spiels (K-Faktor)
   11. h2h_a, h2h_b       - Win-Rate Team A bzw. B in den letzten 5 direkten Duellen
   12. age_a, age_b       - Durchschnittsalter der letzten Startelf (verwenden wir als konstante 27 falls unbek.)

Target: 0 = Unentschieden, 1 = Sieg Team A, 2 = Niederlage Team A
        (Reihenfolge so gewaehlt, dass Klasse 0 (Unentschieden) in der Mitte liegt)

Zeitbasierter Split:
    train: 2000-01-01 .. 2023-12-31
    val:   2024-01-01 .. 2025-12-31
    test:  2026-01-01 .. (zur Modell-Evaluation gedacht, aber wir trainieren final auf allen Daten
                          bis vor dem Turnier und nutzen 2024-2025 als Validierung)

Output: data/processed/features.parquet (effizient) + .npz (numpy) + meta.json
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
HOME_ADVANTAGE_ELO = 80.0  # ~ 80 Elo-Punkte fuer Heimbonus
FORM_WINDOW = 5
REST_WINDOW_DAYS = 365  # Reset wenn aelter


def expected_score(rating_a: float, rating_b: float) -> float:
    """Elo-Erwartungswert fuer Team A gegen Team B (0..1)."""
    return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))


def update_elo(
    rating_a: float,
    rating_b: float,
    score_a: float,  # 1.0 win, 0.5 draw, 0.0 loss
    k: int = 30,
) -> tuple[float, float]:
    """Elo-Update nach einem Spiel. Liefert neue Ratings (a, b)."""
    ea = expected_score(rating_a, rating_b)
    eb = 1.0 - ea
    new_a = rating_a + k * (score_a - ea)
    new_b = rating_b + k * ((1.0 - score_a) - eb)
    return new_a, new_b


def build_feature_table() -> Path:
    """Liest results.csv, berechnet Features, schreibt Parquet + Numpy-Bundles + Meta."""
    print("=" * 70)
    print(" Feature-Engineering")
    print("=" * 70)
    print(f">> Lade {RAW_RESULTS}")
    df = pd.read_csv(RAW_RESULTS, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    print(f"   {len(df):,} Spiele gelesen ({df['date'].min():%Y-%m-%d} .. {df['date'].max():%Y-%m-%d})")

    # State-Dictionaries
    elo: dict[str, float] = defaultdict(lambda: ELOG_START)
    # Pro Team: deque der letzten FORM_WINDOW Spiele
    # Wir speichern (date, gf, ga, points) -> points = 3/1/0
    form_history: dict[str, deque] = defaultdict(lambda: deque(maxlen=FORM_WINDOW))
    last_match_date: dict[str, datetime] = {}
    # Head-to-head: pro Paar (a,b sortiert) die letzten 5 Punkte aus Sicht von a
    h2h: dict[tuple[str, str], deque] = defaultdict(lambda: deque(maxlen=FORM_WINDOW))

    n = len(df)
    # Zeilen mit fehlenden Scores (z.B. abgebrochene Spiele) rauswerfen
    before = len(df)
    df = df.dropna(subset=["home_score", "away_score"]).reset_index(drop=True)
    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)
    dropped = before - len(df)
    if dropped:
        print(f"   {dropped:,} Spiele mit fehlenden Scores entfernt.")
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

        # Aktuelle Ratings VOR dem Spiel
        elo_home = elo[home]
        elo_away = elo[away]
        if not neutral:
            # Heimbonus zur Berechnung der expected score, aber raw rating behalten
            elo_home_eff = elo_home + HOME_ADVANTAGE_ELO
        else:
            elo_home_eff = elo_home

        # Feature: Form (Punkte pro Spiel)
        fh = form_history[home]
        fa = form_history[away]
        form5_home = (sum(x[3] for x in fh) / len(fh)) if fh else 1.0
        form5_away = (sum(x[3] for x in fa) / len(fa)) if fa else 1.0
        gf5_home = (sum(x[1] for x in fh) / len(fh)) if fh else 1.0
        ga5_home = (sum(x[2] for x in fh) / len(fh)) if fh else 1.0
        gf5_away = (sum(x[1] for x in fa) / len(fa)) if fa else 1.0
        ga5_away = (sum(x[2] for x in fa) / len(fa)) if fa else 1.0

        # Rest days
        rest_home = (date - last_match_date[home]).days if home in last_match_date else 30
        rest_away = (date - last_match_date[away]).days if away in last_match_date else 30
        rest_home = min(rest_home, REST_WINDOW_DAYS)
        rest_away = min(rest_away, REST_WINDOW_DAYS)

        # Head-to-head (aus Sicht home)
        key = tuple(sorted([home, away]))
        h2h_hist = list(h2h[key])
        if h2h_hist:
            # h2h_hist speichert (winner: 'home'|'away'|'draw')
            h2h_home = sum(1 for w in h2h_hist if w == home) / len(h2h_hist)
            h2h_away = sum(1 for w in h2h_hist if w == away) / len(h2h_hist)
        else:
            h2h_home = 0.5
            h2h_away = 0.5

        # Target (One-Hot-artig in einer Spalte: 0=draw, 1=home win, 2=away win)
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
            "elo_home": elo_home,
            "elo_away": elo_away,
            "elo_home_eff": elo_home_eff,
            "elo_diff": elo_home_eff - elo_away,
            "form5_home": form5_home,
            "form5_away": form5_away,
            "gf5_home": gf5_home,
            "ga5_home": ga5_home,
            "gf5_away": gf5_away,
            "ga5_away": ga5_away,
            "rest_home": rest_home,
            "rest_away": rest_away,
            "neutral": int(neutral),
            "tournament": tournament,
            "tournament_w": tournament_weight(tournament),
            "h2h_home": h2h_home,
            "h2h_away": h2h_away,
            "target": target,
        })

        # ---------- State Updates (fuer ZUKUENFTIGE Features) ----------
        # Elo
        k = tournament_weight(tournament)
        new_home, new_away = update_elo(elo_home_eff, elo_away, score_home, k=k)
        # Wenn Heimbonus drauf war, ziehen wir ihn wieder ab
        if not neutral:
            new_home -= HOME_ADVANTAGE_ELO
        elo[home] = new_home
        elo[away] = new_away

        # Form
        pts_home = 3 if hs > as_ else (1 if hs == as_ else 0)
        pts_away = 3 if as_ > hs else (1 if hs == as_ else 0)
        form_history[home].append((date, hs, as_, pts_home))
        form_history[away].append((date, as_, hs, pts_away))
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

        if (i + 1) % 5000 == 0:
            print(f"   ... {i+1:,}/{n:,} Spiele verarbeitet")

    feat_df = pd.DataFrame(rows)
    feat_df["date"] = pd.to_datetime(feat_df["date"])
    feat_df = feat_df.sort_values("date").reset_index(drop=True)
    print(f"   {len(feat_df):,} Feature-Zeilen erzeugt.")

    # Speichern
    out_parquet = PROCESSED_DIR / "features.parquet"
    feat_df.to_parquet(out_parquet, index=False)
    print(f"   geschrieben: {out_parquet}")

    # X / y numpy Bundles
    feature_cols = [
        "elo_home", "elo_away", "elo_diff",
        "form5_home", "form5_away",
        "gf5_home", "ga5_home", "gf5_away", "ga5_away",
        "rest_home", "rest_away",
        "neutral", "tournament_w",
        "h2h_home", "h2h_away",
    ]
    X = feat_df[feature_cols].astype("float32").to_numpy()
    y = feat_df["target"].astype("int64").to_numpy()
    dates = feat_df["date"].astype("datetime64[ns]").to_numpy()
    home_teams = feat_df["home_team"].to_numpy()
    away_teams = feat_df["away_team"].to_numpy()

    np.savez_compressed(
        PROCESSED_DIR / "features.npz",
        X=X, y=y, dates=dates, home=home_teams, away=away_teams,
        feature_names=np.array(feature_cols),
    )
    print(f"   geschrieben: {PROCESSED_DIR / 'features.npz'}")

    # Meta (Feature-Namen + Stats)
    meta = {
        "n_samples": int(len(feat_df)),
        "feature_columns": feature_cols,
        "date_min": str(feat_df["date"].min()),
        "date_max": str(feat_df["date"].max()),
        "elo_start": ELOG_START,
        "home_advantage_elo": HOME_ADVANTAGE_ELO,
        "form_window": FORM_WINDOW,
        "class_distribution": {int(k): int(v) for k, v in pd.Series(y).value_counts().to_dict().items()},
        "n_teams": int(len(set(home_teams) | set(away_teams))),
    }
    with open(PROCESSED_DIR / "features_meta.json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2, ensure_ascii=False)
    print(f"   geschrieben: {PROCESSED_DIR / 'features_meta.json'}")
    print(f"   Klassenverteilung: {meta['class_distribution']} (0=Draw, 1=HomeWin, 2=AwayWin)")
    print(f"   Anzahl Teams: {meta['n_teams']}")
    print("=" * 70)
    return out_parquet


def load_features(split: str = "all", train_start: str = "2000-01-01", val_start: str = "2024-01-01"):
    """Laedt die Features als numpy arrays. Optional mit Zeit-Split.

    split: 'train' | 'val' | 'all'
    """
    bundle = np.load(PROCESSED_DIR / "features.npz", allow_pickle=True)
    X = bundle["X"]
    y = bundle["y"]
    dates = pd.to_datetime(bundle["dates"])

    if split == "train":
        mask = (dates >= pd.Timestamp(train_start)) & (dates < pd.Timestamp(val_start))
    elif split == "val":
        mask = dates >= pd.Timestamp(val_start)
    elif split == "all":
        mask = np.ones(len(dates), dtype=bool)
    else:
        raise ValueError(f"unknown split: {split}")
    return X[mask], y[mask], dates[mask], bundle["home"][mask], bundle["away"][mask], bundle["feature_names"]


def get_current_team_ratings() -> dict[str, dict]:
    """Liest die finalen Elo-Werte + Form aller Teams aus dem letzten Spiel.

    Wird im Inference-Modus gebraucht.
    """
    # Wir muessen die Rohdaten lesen, weil das Parquet-File keine Scores enthaelt
    df = pd.read_csv(RAW_RESULTS, parse_dates=["date"])
    df = df.dropna(subset=["home_score", "away_score"])
    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)
    df = df.sort_values("date").reset_index(drop=True)
    return _compute_final_state(df)


def _compute_final_state(df: pd.DataFrame) -> dict[str, dict]:
    """Rekonstruiert die finalen Elo + Form-Stats aus dem DataFrame."""
    elo: dict[str, float] = defaultdict(lambda: ELOG_START)
    form_history: dict[str, deque] = defaultdict(lambda: deque(maxlen=FORM_WINDOW))
    last_match_date: dict[str, datetime] = {}

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

        form_history[home].append((date, hs, as_, pts_home))
        form_history[away].append((date, as_, hs, pts_away))
        last_match_date[home] = date
        last_match_date[away] = date

    # Sammle alles
    out = {}
    all_teams = set(elo.keys())
    for t in all_teams:
        fh = form_history[t]
        if fh:
            form5 = sum(x[3] for x in fh) / len(fh)
            gf5 = sum(x[1] for x in fh) / len(fh)
            ga5 = sum(x[2] for x in fh) / len(fh)
            last = fh[-1][0]
        else:
            form5 = 1.0
            gf5 = 1.0
            ga5 = 1.0
            last = None
        out[t] = {
            "elo": float(elo[t]),
            "form5": float(form5),
            "gf5": float(gf5),
            "ga5": float(ga5),
            "last_match": last.isoformat() if last is not None else None,
        }
    return out


def main() -> int:
    build_feature_table()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
