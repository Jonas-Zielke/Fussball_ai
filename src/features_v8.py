"""
Feature-Engineering V8 — V6 Static + Match-Sequences + Player Tensors.

Builds three complementary input modalities for E8Net:
  (A) Static vector     (49k × 57)  — reused from features_v6
  (B) Match-sequence    (49k × K × SEQ_DIM) — last K matches per team
  (C) Squad tensor      (49k × N_PLAYERS × PLAYER_DIM) — Top-N players per team

All modalities are strictly chronological (no leakage):
  - Sequence buffers are populated BEFORE the current match is processed
  - FIFA player data uses the most recent year available BEFORE the match date

Output: data/processed/features_v8.npz
  seq_home, seq_away       — float32 (n_matches, SEQ_LEN, SEQ_DIM)
  squad_home, squad_away   — float32 (n_matches, N_PLAYERS, PLAYER_DIM)
  X, y, y_home_goals, y_away_goals, dates, home, away, feature_names
  — identical to features_v6.npz (reloaded from disk)

Sequence features (SEQ_DIM=7):
  [0] opp_elo_norm       = opponent Elo / 1500
  [1] is_home            = 1.0 home / 0.5 neutral / 0.0 away
  [2] goals_for_norm     = min(goals_scored / 5, 1.0)
  [3] goals_against_norm = min(goals_conceded / 5, 1.0)
  [4] result             = 1.0 win / 0.5 draw / 0.0 loss
  [5] rest_days_norm     = log1p(rest_days) / log1p(200)
  [6] tourn_weight_norm  = tournament_weight / 1.5

Player features (PLAYER_DIM=3):
  [0] overall_norm  = (overall - 60) / 40  →  60=0, 100=1
  [1] age_norm      = (age - 18) / 22      →  18=0, 40=1
  [2] pos_category  = 0.0 GK / 0.33 DEF / 0.67 MID / 1.0 ATT
"""
from __future__ import annotations

import math
import warnings
from collections import defaultdict, deque
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .team_normalize import tournament_weight
from .features_v3 import ELOG_START, HOME_ADVANTAGE_ELO
from .fifa_squad import _normalize_nat, _player_positions

REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_RESULTS = REPO_ROOT / "data" / "raw" / "results.csv"
FIFA_DIR = REPO_ROOT / "data" / "raw" / "fifa"
PROCESSED_DIR = REPO_ROOT / "data" / "processed"

# ── Hyperparameters ──────────────────────────────────────────────────────────
SEQ_LEN = 10        # last N matches per team in the sequence
N_PLAYERS = 15      # Top-N players per national team in squad tensor
SEQ_DIM = 7         # features per match in sequence
PLAYER_DIM = 3      # features per player
FIFA_YEAR_MIN = 15  # players_15.csv is the earliest available
FIFA_YEAR_MAX = 22  # players_22.csv is the latest available

# Position categories for squad tensor
_GK = {"GK"}
_DEF = {"CB", "LCB", "RCB", "LB", "RB", "LWB", "RWB"}
_MID = {"CM", "CDM", "CAM", "LM", "RM", "DM", "ACM", "LCM", "RCM"}
_ATT = {"ST", "CF", "LW", "RW", "LF", "RF", "LS", "RS", "SS"}


def _pos_category(pos_str: str) -> float:
    if not isinstance(pos_str, str):
        return 0.67  # default MID
    positions = _player_positions(pos_str)
    if positions & _GK:
        return 0.0
    if positions & _DEF:
        return 0.33
    if positions & _ATT:
        return 1.0
    return 0.67


def _fifa_year(match_year: int) -> int:
    """Map match year to FIFA data year (strictly not after the match)."""
    return max(FIFA_YEAR_MIN, min(FIFA_YEAR_MAX, match_year - 2000))


# ── FIFA Player Tensor Precomputation ────────────────────────────────────────

def _load_player_tensors() -> dict[tuple[str, int], np.ndarray]:
    """
    Returns {(nation, fifa_year): np.ndarray shape (N_PLAYERS, PLAYER_DIM)}.
    Rows are sorted descending by overall. Padded with zeros if < N_PLAYERS.
    Missing nations get an all-zero tensor.
    """
    print("   Loading FIFA player tensors ...", flush=True)
    tensors: dict[tuple[str, int], np.ndarray] = {}

    for year_suffix in range(FIFA_YEAR_MIN, FIFA_YEAR_MAX + 1):
        csv_path = FIFA_DIR / f"players_{year_suffix:02d}.csv"
        if not csv_path.exists():
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df = pd.read_csv(csv_path, low_memory=False, encoding="utf-8-sig")

        # Normalise column names
        nat_col = "nationality_name" if "nationality_name" in df.columns else "nationality"
        if nat_col not in df.columns:
            nat_col = next((c for c in df.columns if "nation" in c.lower() and "name" in c.lower()), None)
        if nat_col is None:
            continue
        pos_col = "player_positions" if "player_positions" in df.columns else "team_position"
        if pos_col not in df.columns:
            df["player_positions"] = ""
            pos_col = "player_positions"

        for col in ["overall", "age"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.rename(columns={nat_col: "_nation", pos_col: "_pos"})
        df["_nation"] = df["_nation"].apply(_normalize_nat)
        df = df.dropna(subset=["_nation", "overall"])

        for nation, grp in df.groupby("_nation"):
            grp_sorted = grp.sort_values("overall", ascending=False).head(N_PLAYERS)
            rows = []
            for _, p in grp_sorted.iterrows():
                ovr = float(p["overall"])
                age = float(p.get("age", 27.0)) if not pd.isna(p.get("age", float("nan"))) else 27.0
                pos = str(p["_pos"]) if "_pos" in p.index else ""
                rows.append([
                    (ovr - 60.0) / 40.0,             # [0] overall_norm
                    (age - 18.0) / 22.0,              # [1] age_norm
                    _pos_category(pos),               # [2] pos_category
                ])
            tensor = np.zeros((N_PLAYERS, PLAYER_DIM), dtype=np.float32)
            n = min(len(rows), N_PLAYERS)
            if n > 0:
                tensor[:n] = np.array(rows[:n], dtype=np.float32)
            tensors[(str(nation), year_suffix)] = tensor

    print(f"   Loaded player tensors for {len(tensors)} (nation, year) pairs.", flush=True)
    return tensors


def _get_squad_tensor(
    nation: str,
    match_year: int,
    tensors: dict[tuple[str, int], np.ndarray],
) -> np.ndarray:
    """Return player tensor for nation at match_year (with backward fill).

    Matches before 2015 (= before players_15 data existed) get the all-zero
    "missing" tensor — the clamp in _fifa_year would otherwise leak future
    FIFA-15 data backward into earlier matches.
    """
    if match_year < 2000 + FIFA_YEAR_MIN:
        return np.zeros((N_PLAYERS, PLAYER_DIM), dtype=np.float32)
    fy = _fifa_year(match_year)
    # Try exact year first, then search backward
    for y in range(fy, FIFA_YEAR_MIN - 1, -1):
        key = (nation, y)
        if key in tensors:
            return tensors[key]
    return np.zeros((N_PLAYERS, PLAYER_DIM), dtype=np.float32)


# ── Sequence Buffer Helpers ──────────────────────────────────────────────────

_LOG_200 = math.log(201.0)  # normalisation constant for rest days


def _make_seq_entry(
    opp_elo: float,
    is_home: float,
    gf: int,
    ga: int,
    rest_days: int,
    result: float,  # 1.0/0.5/0.0
    tourn_weight: float,
) -> list[float]:
    return [
        opp_elo / 1500.0,
        is_home,
        min(gf / 5.0, 1.0),
        min(ga / 5.0, 1.0),
        result,
        math.log1p(min(rest_days, 200)) / _LOG_200,
        tourn_weight / 60.0,  # 60 = FIFA World Cup K-factor (max)
    ]


def _seq_to_tensor(buf: deque) -> np.ndarray:
    """Convert sequence buffer to padded (SEQ_LEN, SEQ_DIM) tensor."""
    t = np.zeros((SEQ_LEN, SEQ_DIM), dtype=np.float32)
    entries = list(buf)[-SEQ_LEN:]
    for i, entry in enumerate(entries):
        t[i] = entry
    return t


# ── Main Builder ─────────────────────────────────────────────────────────────

def build_feature_table_v8(force: bool = False) -> Path:
    out_path = PROCESSED_DIR / "features_v8.npz"
    if out_path.exists() and not force:
        print(f"   features_v8.npz already exists. Use force=True to rebuild.")
        return out_path

    print("=" * 70)
    print(" V8 Features: V6 Static + Match-Sequences + Player Tensors")
    print("=" * 70)

    # ── Load base V6 data (already computed) ────────────────────────────────
    v6_path = PROCESSED_DIR / "features_v6.npz"
    if not v6_path.exists():
        from .features_v6 import build_feature_table_v6
        build_feature_table_v6()
    v6 = np.load(v6_path, allow_pickle=True)
    X = v6["X"]
    y = v6["y"]
    y_hg = v6["y_home_goals"]
    y_ag = v6["y_away_goals"]
    dates_v6 = v6["dates"]
    home_names = v6["home"]
    away_names = v6["away"]
    feature_names = v6["feature_names"]
    n_matches = len(X)
    print(f"   V6: {n_matches:,} matches loaded.")

    # ── Preload FIFA player tensors ──────────────────────────────────────────
    player_tensors = _load_player_tensors()

    # ── Load results.csv for sequence replay ────────────────────────────────
    df = pd.read_csv(RAW_RESULTS, parse_dates=["date"])
    df = df.dropna(subset=["home_score", "away_score"]).copy()
    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)
    df = df.sort_values("date").reset_index(drop=True)
    print(f"   results.csv: {len(df):,} completed matches.")

    # ── Chronological sequence replay ────────────────────────────────────────
    # Elo state for sequence features (independent from V6, same init)
    elo: dict[str, float] = defaultdict(lambda: ELOG_START)
    last_date: dict[str, datetime] = {}
    seq_buf: dict[str, deque] = defaultdict(lambda: deque(maxlen=SEQ_LEN))

    # We need to align replay rows to V6 rows (same filter: has scores, sorted by date).
    # Build a (date, home, away) → v6_index map for O(1) lookup.
    v6_index: dict[tuple, int] = {}
    for i in range(n_matches):
        d = pd.Timestamp(dates_v6[i]).strftime("%Y-%m-%d")
        key = (d, str(home_names[i]), str(away_names[i]))
        v6_index[key] = i

    seq_home_arr = np.zeros((n_matches, SEQ_LEN, SEQ_DIM), dtype=np.float32)
    seq_away_arr = np.zeros((n_matches, SEQ_LEN, SEQ_DIM), dtype=np.float32)
    squad_home_arr = np.zeros((n_matches, N_PLAYERS, PLAYER_DIM), dtype=np.float32)
    squad_away_arr = np.zeros((n_matches, N_PLAYERS, PLAYER_DIM), dtype=np.float32)

    hits = 0
    for _, row in df.iterrows():
        home = str(row["home_team"])
        away = str(row["away_team"])
        date: datetime = row["date"]
        hs = int(row["home_score"])
        as_ = int(row["away_score"])
        neutral = bool(row.get("neutral", True))
        tourn = str(row.get("tournament", "Friendly"))
        year = date.year

        elo_home = elo[home]
        elo_away = elo[away]

        rest_h = (date - last_date[home]).days if home in last_date else 30
        rest_a = (date - last_date[away]).days if away in last_date else 30
        rest_h = min(rest_h, 200)
        rest_a = min(rest_a, 200)

        tw = tournament_weight(tourn)

        # ── BEFORE updating state: capture pre-match features ────────────────
        key = (date.strftime("%Y-%m-%d"), home, away)
        idx = v6_index.get(key)
        if idx is not None:
            seq_home_arr[idx] = _seq_to_tensor(seq_buf[home])
            seq_away_arr[idx] = _seq_to_tensor(seq_buf[away])
            squad_home_arr[idx] = _get_squad_tensor(home, year, player_tensors)
            squad_away_arr[idx] = _get_squad_tensor(away, year, player_tensors)
            hits += 1

        # ── Update state AFTER capturing ────────────────────────────────────
        result_h = 1.0 if hs > as_ else (0.5 if hs == as_ else 0.0)
        result_a = 1.0 - result_h if hs != as_ else 0.5

        is_home_val = 0.5 if neutral else 1.0
        is_away_val = 0.5 if neutral else 0.0

        seq_buf[home].append(_make_seq_entry(elo_away, is_home_val, hs, as_, rest_h, result_h, tw))
        seq_buf[away].append(_make_seq_entry(elo_home, is_away_val, as_, hs, rest_a, result_a, tw))

        # Elo update — tw is already the K-factor (20/40/60) as in features_v3
        k_factor = tw
        exp_h = 1.0 / (1.0 + 10 ** ((elo_away - elo_home) / 400.0))
        exp_a = 1.0 - exp_h
        elo[home] += k_factor * (result_h - exp_h)
        elo[away] += k_factor * (result_a - exp_a)

        last_date[home] = date
        last_date[away] = date

    print(f"   Sequence replay: {hits:,}/{n_matches:,} rows matched.")

    # ── Save ─────────────────────────────────────────────────────────────────
    np.savez_compressed(
        out_path,
        X=X,
        y=y,
        y_home_goals=y_hg,
        y_away_goals=y_ag,
        dates=dates_v6,
        home=home_names,
        away=away_names,
        feature_names=feature_names,
        seq_home=seq_home_arr,
        seq_away=seq_away_arr,
        squad_home=squad_home_arr,
        squad_away=squad_away_arr,
        seq_len=np.array(SEQ_LEN),
        n_players=np.array(N_PLAYERS),
        seq_dim=np.array(SEQ_DIM),
        player_dim=np.array(PLAYER_DIM),
    )
    size_mb = out_path.stat().st_size / 1e6
    print(f"   Saved: {out_path}  ({size_mb:.1f} MB)")
    print(f"   seq_home:    {seq_home_arr.shape}")
    print(f"   seq_away:    {seq_away_arr.shape}")
    print(f"   squad_home:  {squad_home_arr.shape}")
    print(f"   squad_away:  {squad_away_arr.shape}")
    print("=" * 70)
    return out_path


def load_v8(force: bool = False) -> dict:
    """Load (or build) features_v8.npz and return as dict of numpy arrays."""
    p = PROCESSED_DIR / "features_v8.npz"
    if not p.exists() or force:
        build_feature_table_v8(force=force)
    d = np.load(p, allow_pickle=True)
    return {k: d[k] for k in d.keys()}


def main() -> int:
    build_feature_table_v8(force=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
