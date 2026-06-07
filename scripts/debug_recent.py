"""Show recent Germany matches to understand data."""
import pandas as pd

df = pd.read_csv("data/raw/results.csv", parse_dates=["date"])
df = df.dropna(subset=["home_score", "away_score"])
df = df.sort_values("date").reset_index(drop=True)
germany = df[(df["home_team"] == "Germany") | (df["away_team"] == "Germany")].tail(30)
print("Deutschland letzte 30 Spiele:")
for _, r in germany.iterrows():
    is_home = r["home_team"] == "Germany"
    score = f"{int(r['home_score'])}:{int(r['away_score'])}"
    opp = r["away_team"] if is_home else r["home_team"]
    if is_home:
        res = "W" if r["home_score"] > r["away_score"] else "D" if r["home_score"] == r["away_score"] else "L"
    else:
        res = "W" if r["away_score"] > r["home_score"] else "D" if r["home_score"] == r["away_score"] else "L"
    print(f'  {r["date"].date()}  {r["tournament"][:25]:<25}  vs {opp:<22}  {score:>5}  {res}')

print()
uruguay = df[(df["home_team"] == "Uruguay") | (df["away_team"] == "Uruguay")].tail(15)
print("Uruguay letzte 15 Spiele:")
for _, r in uruguay.iterrows():
    is_home = r["home_team"] == "Uruguay"
    score = f"{int(r['home_score'])}:{int(r['away_score'])}"
    opp = r["away_team"] if is_home else r["home_team"]
    if is_home:
        res = "W" if r["home_score"] > r["away_score"] else "D" if r["home_score"] == r["away_score"] else "L"
    else:
        res = "W" if r["away_score"] > r["home_score"] else "D" if r["home_score"] == r["away_score"] else "L"
    print(f'  {r["date"].date()}  {r["tournament"][:25]:<25}  vs {opp:<22}  {score:>5}  {res}')
