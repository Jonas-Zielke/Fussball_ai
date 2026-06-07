"""Export V5 teams data (mit vr_elo, form1, form2, stability)."""
import json
from pathlib import Path

import sys
sys.path.insert(0, "E:/Projects/Fussball_ai")
from src.features_v5 import get_current_team_ratings_v5

OUT = Path("E:/Profilov2/public/data/wm-predictor/teams.json")
state = get_current_team_ratings_v5()

teams_out = {}
for name, st in state.items():
    teams_out[name] = {
        "elo": float(st["elo"]),
        "re_elo": float(st.get("re_elo", st["elo"])),
        "vr_elo": float(st.get("vr_elo", st["re_elo"])),
        "form1": float(st.get("form1", 1.0)),
        "form2": float(st.get("form2", st["form3"])),
        "form3": float(st["form3"]),
        "form5": float(st["form5"]),
        "form10": float(st["form10"]),
        "gf5": float(st["gf5"]),
        "ga5": float(st["ga5"]),
        "gd5": float(st["gd5"]),
        "home_form5": float(st.get("home_form5", 1.0)),
        "away_form5": float(st.get("away_form5", 1.0)),
        "elo_mom": float(st["elo_mom"]),
        "win_streak": int(st["win_streak"]),
        "unbeaten": int(st["unbeaten"]),
        "continent": int(st["continent"]),
        "oppo_elo5": float(st["oppo_elo5"]),
        "w_form": float(st.get("w_form", st["form5"])),
        "momentum": float(st.get("momentum", 0.0)),
        "wins_top10": float(st.get("wins_top10", 0.0)),
        "wins_top20": float(st.get("wins_top20", 0.0)),
        "stability": float(st.get("stability", 1.0)),
        "last_match": st["last_match"],
    }

items_sorted = sorted(teams_out.items(), key=lambda x: -x[1]["vr_elo"])
ranking = [(name, float(st["vr_elo"])) for name, st in items_sorted]
out = {
    "n_teams": len(teams_out),
    "as_of": max((t["last_match"] for t in teams_out.values() if t["last_match"]), default=None),
    "teams": teams_out,
    "ranking_top50": ranking[:50],
}
with open(OUT, "w", encoding="utf-8") as fh:
    json.dump(out, fh, separators=(",", ":"))
print(f"OK: {OUT} ({OUT.stat().st_size/1024:.1f} KB, {len(teams_out)} Teams)")
