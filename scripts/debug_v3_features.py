"""Compare V3 features Germany vs Uruguay."""
from src.features_v3 import get_current_team_ratings_v3

state = get_current_team_ratings_v3()
g = state["Germany"]
u = state["Uruguay"]

print("Germany:")
print(f"  Elo (cumulative): {g['elo']:.0f}")
print(f"  Recent-Elo (2y):  {g['re_elo']:.0f}")
print(f"  Form5:            {g['form5']:.2f}")
print(f"  Weighted Form:    {g['w_form']:.2f}")
print(f"  Momentum:         {g['momentum']:+.2f}")
print(f"  Wins vs Top-10:   {g['wins_top10']:.2f}")
print(f"  Wins vs Top-20:   {g['wins_top20']:.2f}")
print(f"  Win Streak:       {g['win_streak']}")
print()
print("Uruguay:")
print(f"  Elo (cumulative): {u['elo']:.0f}")
print(f"  Recent-Elo (2y):  {u['re_elo']:.0f}")
print(f"  Form5:            {u['form5']:.2f}")
print(f"  Weighted Form:    {u['w_form']:.2f}")
print(f"  Momentum:         {u['momentum']:+.2f}")
print(f"  Wins vs Top-10:   {u['wins_top10']:.2f}")
print(f"  Wins vs Top-20:   {u['wins_top20']:.2f}")
print(f"  Win Streak:       {u['win_streak']}")

print()
print("Differenzen (Germany - Uruguay):")
print(f"  Recent-Elo: {g['re_elo'] - u['re_elo']:+.0f}")
print(f"  Form5:      {g['form5'] - u['form5']:+.2f}")
print(f"  Momentum:   {g['momentum'] - u['momentum']:+.2f}")
print(f"  W-Form:     {g['w_form'] - u['w_form']:+.2f}")
print(f"  Top-20 W:   {g['wins_top20'] - u['wins_top20']:+.2f}")
