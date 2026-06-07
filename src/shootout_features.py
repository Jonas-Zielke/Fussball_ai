"""
Elfmeterschießen-Modell (Shootout-V1) – Feature-Engineering & Inferenz.

Elfmeter-Stärke ist teamspezifisch und weitgehend unabhängig von der Spielstärke:
manche Teams spielen stark, schießen aber schlecht (und umgekehrt). Dieses Modul
baut aus `data/raw/shootouts.csv` (677 historische Shootouts) leakage-sichere
Team-Features und stellt ein trainiertes, kleines Modell für die Vorhersage des
Shootout-Siegers bereit.

Features pro Shootout (nur aus der Vergangenheit, kein Leakage):
  pen_skill_x  - Bayesian-geschrumpfte Siegquote (wins + K*0.5)/(games + K), Prior 0.5
  pen_exp_x    - log(1+games), Erfahrung/Sicherheit der Schätzung
  str_diff     - Punkt-in-Zeit Elo-Differenz (home - away), skaliert; lässt das Modell
                 lernen, *wie wenig* allgemeine Spielstärke fürs Elfmeterschießen zählt
"""
from __future__ import annotations

from collections import defaultdict, deque  # noqa: F401  (deque evtl. später)
from math import log
from pathlib import Path

import numpy as np
import pandas as pd

from .team_normalize import normalize_team_name
from .features_v3 import ELOG_START, HOME_ADVANTAGE_ELO
from .features_v6 import update_elo

REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_RESULTS = REPO_ROOT / "data" / "raw" / "results.csv"
RAW_SHOOTOUTS = REPO_ROOT / "data" / "raw" / "shootouts.csv"
MODEL_PATH = REPO_ROOT / "data" / "models" / "shootout_v1.pt"

# Bayesian-Shrinkage-Stärke der Penalty-Quote (Prior = 0.5).
K_SHRINK = 5.0
# Skalierung der Elo-Differenz, damit das Feature in ähnlicher Größenordnung liegt.
ELO_SCALE = 100.0

FEATURE_ORDER = ["pen_skill_a", "pen_skill_b", "pen_exp_a", "pen_exp_b", "str_diff"]


def _pen_skill(wins: float, games: float) -> float:
    return (wins + K_SHRINK * 0.5) / (games + K_SHRINK)


def _pen_exp(games: float) -> float:
    return log(1.0 + games)


# ---------------------------------------------------------------------------
# Elo-Snapshots: Punkt-in-Zeit Spielstärke an den Shootout-Terminen
# ---------------------------------------------------------------------------
def _elo_snapshots(keys: set[tuple]) -> dict[tuple, tuple[float, float]]:
    """Replayt eine einfache Elo über results.csv und schnappt für jedes
    gesuchte (date, home, away) die Pre-Game-Elo beider Teams.

    keys: Set aus (pd.Timestamp(normalisiert auf Tag), home_norm, away_norm).
    Rückgabe: dict key -> (elo_home, elo_away).
    """
    df = pd.read_csv(RAW_RESULTS, parse_dates=["date"])
    df = df.dropna(subset=["home_score", "away_score"])
    df = df.sort_values("date").reset_index(drop=True)

    elo: dict[str, float] = defaultdict(lambda: ELOG_START)
    from .team_normalize import tournament_weight

    snaps: dict[tuple, tuple[float, float]] = {}
    for row in df.itertuples(index=False):
        home = normalize_team_name(row.home_team)
        away = normalize_team_name(row.away_team)
        day = pd.Timestamp(row.date).normalize()
        neutral = bool(row.neutral)

        elo_home = elo[home]
        elo_away = elo[away]

        # Pre-Game-Snapshot, falls dieses Spiel ein gesuchtes Shootout ist
        for k in ((day, home, away), (day, away, home)):
            if k in snaps:
                continue
            if k in keys:
                if k[1] == home:
                    snaps[k] = (elo_home, elo_away)
                else:
                    snaps[k] = (elo_away, elo_home)

        # Elo-Update
        hs = int(row.home_score)
        as_ = int(row.away_score)
        score_home = 1.0 if hs > as_ else (0.0 if hs < as_ else 0.5)
        k_w = tournament_weight(row.tournament)
        elo_h_eff = elo_home + (0 if neutral else HOME_ADVANTAGE_ELO)
        new_h, new_a = update_elo(elo_h_eff, elo_away, score_home, k=k_w)
        if not neutral:
            new_h -= HOME_ADVANTAGE_ELO
        elo[home] = new_h
        elo[away] = new_a

    return snaps, dict(elo)


def _load_shootouts() -> pd.DataFrame:
    s = pd.read_csv(RAW_SHOOTOUTS, parse_dates=["date"])
    s = s.dropna(subset=["home_team", "away_team", "winner"])
    s["home_n"] = s["home_team"].map(normalize_team_name)
    s["away_n"] = s["away_team"].map(normalize_team_name)
    s["winner_n"] = s["winner"].map(normalize_team_name)
    s = s.sort_values("date").reset_index(drop=True)
    return s


def build_shootout_dataset():
    """Baut Trainingsdaten aus shootouts.csv (leakage-sicher, chronologisch).

    Rückgabe:
      X        (n, 5) float32 – Features in FEATURE_ORDER
      y        (n,)   int64   – 1 wenn home_n das Shootout gewinnt, sonst 0
      dates    (n,)   datetime64
      teams    list[(home_n, away_n)]
    """
    s = _load_shootouts()

    # Elo-Snapshots vorbereiten
    keys = {(pd.Timestamp(r.date).normalize(), r.home_n, r.away_n) for r in s.itertuples(index=False)}
    snaps, _final_elo = _elo_snapshots(keys)

    wins: dict[str, float] = defaultdict(float)
    games: dict[str, float] = defaultdict(float)

    X, y, dates, teams = [], [], [], []
    for r in s.itertuples(index=False):
        home, away = r.home_n, r.away_n
        day = pd.Timestamp(r.date).normalize()

        # Features NUR aus Vergangenheit
        ps_a = _pen_skill(wins[home], games[home])
        ps_b = _pen_skill(wins[away], games[away])
        pe_a = _pen_exp(games[home])
        pe_b = _pen_exp(games[away])

        elo_h, elo_a = snaps.get((day, home, away), (ELOG_START, ELOG_START))
        str_diff = (elo_h - elo_a) / ELO_SCALE

        X.append([ps_a, ps_b, pe_a, pe_b, str_diff])
        y.append(1 if r.winner_n == home else 0)
        dates.append(r.date)
        teams.append((home, away))

        # State NACH dem Shootout aktualisieren
        games[home] += 1
        games[away] += 1
        if r.winner_n == home:
            wins[home] += 1
        elif r.winner_n == away:
            wins[away] += 1
        # (unklarer Gewinner: keine Win-Gutschrift, aber Spiel zählt)

    return (
        np.array(X, dtype=np.float32),
        np.array(y, dtype=np.int64),
        np.array(dates, dtype="datetime64[ns]"),
        teams,
    )


def get_current_shootout_ratings() -> dict[str, dict]:
    """Aktuelle Penalty-Stats pro Team (über die gesamte Historie)."""
    s = _load_shootouts()
    wins: dict[str, float] = defaultdict(float)
    games: dict[str, float] = defaultdict(float)
    for r in s.itertuples(index=False):
        games[r.home_n] += 1
        games[r.away_n] += 1
        if r.winner_n == r.home_n:
            wins[r.home_n] += 1
        elif r.winner_n == r.away_n:
            wins[r.away_n] += 1

    out: dict[str, dict] = {}
    for team in games:
        out[team] = {
            "pen_skill": float(_pen_skill(wins[team], games[team])),
            "pen_games": int(games[team]),
            "pen_wins": int(wins[team]),
        }
    return out


# ---------------------------------------------------------------------------
# Inferenz
# ---------------------------------------------------------------------------
_MODEL_CACHE: dict = {}


def _load_model():
    if _MODEL_CACHE:
        return _MODEL_CACHE["model"], _MODEL_CACHE["bundle"]
    import torch
    from .train_v2 import DEVICE
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Shootout-Checkpoint nicht gefunden: {MODEL_PATH}\n"
            "Bitte zuerst 'python -m scripts.train_shootout' ausführen."
        )
    bundle = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    model = ShootoutNet(bundle["in_dim"], bundle["arch"]["hidden"])
    model.load_state_dict(bundle["state_dict"])
    model.eval().to(DEVICE)
    _MODEL_CACHE["model"] = model
    _MODEL_CACHE["bundle"] = bundle
    # Aktuelle Ratings + Elo cachen
    _MODEL_CACHE["ratings"] = get_current_shootout_ratings()
    _MODEL_CACHE["elo"] = _current_elo()
    return model, bundle


def _current_elo() -> dict[str, float]:
    """Finale Elo aller Teams (für str_diff zur Inferenzzeit)."""
    snaps, final_elo = _elo_snapshots(set())
    return final_elo


def _feature_vector(home: str, away: str, ratings: dict, elo: dict) -> np.ndarray:
    rh = ratings.get(home, {"pen_skill": 0.5, "pen_games": 0})
    ra = ratings.get(away, {"pen_skill": 0.5, "pen_games": 0})
    str_diff = (elo.get(home, ELOG_START) - elo.get(away, ELOG_START)) / ELO_SCALE
    return np.array([
        rh["pen_skill"], ra["pen_skill"],
        _pen_exp(rh["pen_games"]), _pen_exp(ra["pen_games"]),
        str_diff,
    ], dtype=np.float32)


def predict_shootout(home: str, away: str) -> dict:
    """Wahrscheinlichkeit, dass `home` das Elfmeterschießen gewinnt (symmetrisiert)."""
    import torch
    from .train_v2 import DEVICE
    model, bundle = _load_model()
    ratings = _MODEL_CACHE["ratings"]
    elo = _MODEL_CACHE["elo"]
    mean = np.array(bundle["norm_stats"]["mean"], dtype=np.float32)
    std = np.array(bundle["norm_stats"]["std"], dtype=np.float32)

    home_n = normalize_team_name(home)
    away_n = normalize_team_name(away)

    def _raw(a, b):
        vec = (_feature_vector(a, b, ratings, elo) - mean) / std
        with torch.no_grad():
            logit = model(torch.from_numpy(vec).unsqueeze(0).to(DEVICE))
            return float(torch.sigmoid(logit).cpu().item())

    p = 0.5 * (_raw(home_n, away_n) + (1.0 - _raw(away_n, home_n)))
    p = min(max(p, 0.001), 0.999)
    return {
        "home": home_n,
        "away": away_n,
        "home_win_prob": p,
        "away_win_prob": 1.0 - p,
        "home_pen_skill": ratings.get(home_n, {}).get("pen_skill", 0.5),
        "away_pen_skill": ratings.get(away_n, {}).get("pen_skill", 0.5),
        "model_version": "shootout-v1",
    }


# ShootoutNet hier definiert (klein), damit Inferenz ohne scripts-Import läuft.
def _make_shootout_net():
    import torch.nn as nn

    class _ShootoutNet(nn.Module):
        """Kleines, stark regularisiertes Modell (1 Hidden-Layer)."""

        def __init__(self, in_dim: int, hidden: int = 8):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),
                nn.Linear(hidden, 1),
            )

        def forward(self, x):
            return self.net(x).squeeze(-1)

    return _ShootoutNet


# Lazy-Klassen-Handle (vermeidet harten torch-Import beim Modul-Load)
class ShootoutNet:  # pragma: no cover - dünner Proxy
    def __new__(cls, in_dim: int, hidden: int = 8):
        return _make_shootout_net()(in_dim, hidden)
