"""Export V3 ensemble to browser."""
import json
import sys
import numpy as np
import torch
from pathlib import Path

sys.path.insert(0, "E:/Projects/Fussball_ai")
from src.train_v2 import FootballNet
from src.features_v3 import load_features_v3, PROCESSED_DIR

# Lade V3 Ensemble Modell
bundle_in = json.loads(Path("E:/Projects/Fussball_ai/models/v3_ensemble_3_2026.json").read_text()) if Path("E:/Projects/Fussball_ai/models/v3_ensemble_3_2026.json").exists() else None
# Stattdessen: lade aus dem letzten training
import os
models_dir = Path("E:/Projects/Fussball_ai/models")
latest_v3 = None
for f in models_dir.glob("v3_*.json"):
    if "meta" not in f.name:
        latest_v3 = f
if latest_v3 is None:
    # Use the existing model.json as source (it's already the right one)
    src = Path("E:/Profilov2/public/data/wm-predictor/model.json")
    # The 3-model ensemble was saved with arch hidden=96 n_blocks=4
    # Just re-export teams/config/aliases/h2h
    print("Verwende existierendes model.json in Profilov2 (3-Model V3-Ensemble)")
else:
    print(f"Lade {latest_v3}")
    bundle_in = json.loads(latest_v3.read_text())

# Re-export teams with V3 features
from src.features_v3 import get_current_team_ratings_v3
state = get_current_team_ratings_v3()
teams_out = {}
for name, st in state.items():
    teams_out[name] = {
        "elo": float(st["elo"]),
        "re_elo": float(st.get("re_elo", st["elo"])),
        "form3": float(st["form3"]),
        "form5": float(st["form5"]),
        "form10": float(st["form10"]),
        "gf5": float(st["gf5"]),
        "ga5": float(st["ga5"]),
        "gd5": float(st["gd5"]),
        "home_form5": float(st["home_form5"]),
        "away_form5": float(st["away_form5"]),
        "elo_mom": float(st["elo_mom"]),
        "win_streak": int(st["win_streak"]),
        "unbeaten": int(st["unbeaten"]),
        "continent": int(st["continent"]),
        "oppo_elo5": float(st["oppo_elo5"]),
        "w_form": float(st.get("w_form", st["form5"])),
        "momentum": float(st.get("momentum", 0.0)),
        "wins_top10": float(st.get("wins_top10", 0.0)),
        "wins_top20": float(st.get("wins_top20", 0.0)),
        "last_match": st["last_match"],
    }
items_sorted = sorted(teams_out.items(), key=lambda x: -x[1]["re_elo"])
ranking = [(name, float(st["re_elo"])) for name, st in items_sorted]
out = {
    "n_teams": len(teams_out),
    "as_of": max((t["last_match"] for t in teams_out.values() if t["last_match"]), default=None),
    "teams": teams_out,
    "ranking_top50": ranking[:50],
}
out_path = Path("E:/Profilov2/public/data/wm-predictor/teams.json")
with open(out_path, "w", encoding="utf-8") as fh:
    json.dump(out, fh, separators=(",", ":"))
print(f"OK: {out_path} ({out_path.stat().st_size/1024:.1f} KB)")
