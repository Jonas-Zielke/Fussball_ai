"""
Export pre-computed V8 team tensors for browser inference.

For each WM 2026 team, extracts the latest seq and squad tensors from
features_v8.npz. These allow the browser to do ORT inference without
recomputing sequence/squad features.

The browser workflow:
  1. Compute V6 static features (existing JS code)
  2. Normalize with norm_mean / norm_std from v8_preproc.json
  3. Look up seq_home, squad_home, seq_away, squad_away by team name
  4. Build context = [is_neutral, tournament_weight / 60.0]
  5. Run ORT session with v8_e8net.onnx
  6. Apply KickTipp decision layer

Output:
  data/processed/v8_team_tensors.json   — seq + squad per team
  (preprocessing params already in data/models/v8_preproc.json)

Usage:
  cd E:/Projects/Fussball_ai
  .\\venv\\Scripts\\python.exe -m scripts.export_v8_tensors
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.features_v8 import load_v8

OUT_PATH   = REPO_ROOT / "data" / "processed" / "v8_team_tensors.json"
ODDS_PATH  = REPO_ROOT / "data" / "raw" / "wm2026_odds.json"


def main():
    print("Loading features_v8 ...")
    d = load_v8()
    home_names = d["home"]
    away_names = d["away"]
    dates      = pd.to_datetime(d["dates"])
    seq_home   = d["seq_home"]    # (N, 10, 7)
    seq_away   = d["seq_away"]
    squad_home = d["squad_home"]  # (N, 15, 3)
    squad_away = d["squad_away"]

    # For each team, find last occurrence (home or away) and store tensors
    # Process chronologically so the last write wins (most recent match)
    team_seq:   dict[str, list] = {}
    team_squad: dict[str, list] = {}
    team_last:  dict[str, str]  = {}

    for i in range(len(home_names)):
        ht = home_names[i]
        at = away_names[i]
        dt = str(dates[i].date())

        team_seq[ht]   = seq_home[i].tolist()
        team_squad[ht] = squad_home[i].tolist()
        team_last[ht]  = dt

        team_seq[at]   = seq_away[i].tolist()
        team_squad[at] = squad_away[i].tolist()
        team_last[at]  = dt

    print(f"   Total teams with tensors: {len(team_seq)}")

    # Load WM 2026 team list
    with open(ODDS_PATH) as f:
        odds_data = json.load(f)
    wm_teams = set()
    for key in odds_data["matches"]:
        parts = key.split("|")
        if len(parts) == 2:
            wm_teams.update(parts)

    print(f"   WM 2026 teams: {len(wm_teams)}")
    missing = wm_teams - set(team_seq.keys())
    if missing:
        print(f"   WARNING: Missing tensors for: {sorted(missing)}")

    # Build output: only WM teams (keep file small)
    output = {}
    for team in sorted(wm_teams):
        if team in team_seq:
            output[team] = {
                "seq":   team_seq[team],
                "squad": team_squad[team],
                "last_match": team_last[team],
            }

    # Add ALL teams for completeness (needed for non-WM matches)
    # Separate key to allow selective loading
    all_teams_output = {}
    for team in sorted(team_seq.keys()):
        all_teams_output[team] = {
            "seq":   team_seq[team],
            "squad": team_squad[team],
            "last_match": team_last[team],
        }

    result = {
        "wm2026": output,
        "all": all_teams_output,
        "meta": {
            "seq_shape": [10, 7],
            "squad_shape": [15, 3],
            "n_wm_teams": len(output),
            "n_all_teams": len(all_teams_output),
        }
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(result, f, separators=(",", ":"))  # compact

    size_mb = OUT_PATH.stat().st_size / 1e6
    print(f"   Saved: {OUT_PATH}  ({size_mb:.1f} MB)")
    print(f"   WM teams: {len(output)}, All teams: {len(all_teams_output)}")

    # Print sample to verify
    first_team = sorted(output.keys())[0]
    first = output[first_team]
    print(f"\n   Sample ({first_team}, last={first['last_match']}):")
    print(f"     seq:   {len(first['seq'])} x {len(first['seq'][0])}")
    print(f"     squad: {len(first['squad'])} x {len(first['squad'][0])}")


if __name__ == "__main__":
    main()
