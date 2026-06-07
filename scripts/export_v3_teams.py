"""Export V3 teams data to JSON for browser."""
import json
from pathlib import Path
from src.features_v3 import get_current_team_ratings_v3, PROCESSED_DIR

OUT = Path("E:/Profilov2/public/data/wm-predictor/teams.json")

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

items_sorted = sorted(teams_out.items(), key=lambda x: -x[1]["re_elo"])  # Sort by recent-elo!
ranking = [(name, float(st["re_elo"])) for name, st in items_sorted]

out = {
    "n_teams": len(teams_out),
    "as_of": max((t["last_match"] for t in teams_out.values() if t["last_match"]), default=None),
    "teams": teams_out,
    "ranking_top50": ranking[:50],
}
OUT.parent.mkdir(parents=True, exist_ok=True)
with open(OUT, "w", encoding="utf-8") as fh:
    json.dump(out, fh, separators=(",", ":"))
print(f"OK: {OUT} ({OUT.stat().st_size/1024:.1f} KB, {len(teams_out)} Teams, ranking by recent_elo)")
