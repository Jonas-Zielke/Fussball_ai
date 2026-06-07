"""Quick Elo sanity check."""
from src.features import get_current_team_ratings

state = get_current_team_ratings()
items = sorted(state.items(), key=lambda x: x[1]["elo"], reverse=True)

print("Top 25 Teams nach Elo:")
print("Team                       Elo     Form5   GF5   GA5   LastMatch")
for name, st in items[:25]:
    print(f"{name:<25} {st['elo']:>8.1f}  {st['form5']:>5.2f}  {st['gf5']:>4.2f}  {st['ga5']:>4.2f}  {(st['last_match'] or '-')[:10]}")
print()
print(f"Anzahl Teams mit State: {len(state)}")
print(f"Beispiele: {list(state.keys())[:5]}")
