"""
Exportiert den AKTUELLEN V8-Zustand (Sequenz + Kader) pro Team.

Unterschied zu export_v8_tensors.py: dort wird der Pre-Match-Buffer des
jeweils letzten Spiels exportiert — das letzte Spiel selbst fehlt also im
Sequenz-Tensor (Off-by-one). Hier wird die Replay-Schleife bis HEUTE
durchlaufen und der Endzustand exportiert (inkl. letztem Spiel).

Output: data/processed/v8_final_state.json
  { team: {"seq": 10x7, "squad": 15x3, "elo": float, "last_match": iso} }

Usage:
  .\\venv\\Scripts\\python.exe -m scripts.export_v8_state
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict, deque
from datetime import datetime
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.features_v3 import ELOG_START
from src.team_normalize import tournament_weight
from src.features_v8 import (
    RAW_RESULTS, SEQ_LEN, _load_player_tensors, _get_squad_tensor,
    _make_seq_entry, _seq_to_tensor,
)

OUT_PATH = REPO_ROOT / "data" / "processed" / "v8_final_state.json"


def main() -> int:
    df = pd.read_csv(RAW_RESULTS, parse_dates=["date"])
    df = df.dropna(subset=["home_score", "away_score"]).copy()
    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)
    df = df.sort_values("date").reset_index(drop=True)
    print(f"Replay über {len(df):,} Spiele ...")

    player_tensors = _load_player_tensors()
    elo: dict[str, float] = defaultdict(lambda: ELOG_START)
    last_date: dict[str, datetime] = {}
    seq_buf: dict[str, deque] = defaultdict(lambda: deque(maxlen=SEQ_LEN))

    for row in df.itertuples(index=False):
        home, away = str(row.home_team), str(row.away_team)
        date = row.date
        hs, as_ = int(row.home_score), int(row.away_score)
        neutral = bool(row.neutral)
        tw = tournament_weight(str(row.tournament))

        elo_home, elo_away = elo[home], elo[away]
        rest_h = min((date - last_date[home]).days, 200) if home in last_date else 30
        rest_a = min((date - last_date[away]).days, 200) if away in last_date else 30

        result_h = 1.0 if hs > as_ else (0.5 if hs == as_ else 0.0)
        result_a = 1.0 - result_h if hs != as_ else 0.5
        is_home_val = 0.5 if neutral else 1.0
        is_away_val = 0.5 if neutral else 0.0

        seq_buf[home].append(_make_seq_entry(elo_away, is_home_val, hs, as_, rest_h, result_h, tw))
        seq_buf[away].append(_make_seq_entry(elo_home, is_away_val, as_, hs, rest_a, result_a, tw))

        exp_h = 1.0 / (1.0 + 10 ** ((elo_away - elo_home) / 400.0))
        elo[home] += tw * (result_h - exp_h)
        elo[away] += tw * (result_a - (1.0 - exp_h))
        last_date[home] = date
        last_date[away] = date

    year_now = datetime.now().year
    out = {}
    for team in sorted(seq_buf.keys()):
        out[team] = {
            "seq": _seq_to_tensor(seq_buf[team]).tolist(),
            "squad": _get_squad_tensor(team, year_now, player_tensors).tolist(),
            "elo": round(float(elo[team]), 1),
            "last_match": str(last_date[team].date()),
        }

    with open(OUT_PATH, "w") as f:
        json.dump(out, f, separators=(",", ":"))
    print(f"Gespeichert: {OUT_PATH}  ({OUT_PATH.stat().st_size / 1e6:.1f} MB, {len(out)} Teams)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
