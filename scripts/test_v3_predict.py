"""Test V3 prediction on Germany vs Uruguay."""
import json
from src.train_v3 import train_v3
from src.features_v3 import get_current_team_ratings_v3
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path

# Load V3 model
bundle = json.loads(Path("E:/Profilov2/public/data/wm-predictor/model.json").read_text())

# Build feature vector for Germany vs Uruguay
state = get_current_team_ratings_v3()
home = state["Germany"]
away = state["Uruguay"]

neutral = True
tournament = "FIFA World Cup"
tournamentW = 60
eloHEff = home["elo"] + (0 if neutral else 80)
re_eloHEff = home["re_elo"] + (0 if neutral else 80)

# H2H - need to compute
import pandas as pd
df = pd.read_csv("data/raw/results.csv", parse_dates=["date"])
df = df.dropna(subset=["home_score", "away_score"])
key = "::".join(sorted(["Germany", "Uruguay"]))
pair = df[((df["home_team"] == "Germany") & (df["away_team"] == "Uruguay")) |
         ((df["home_team"] == "Uruguay") & (df["away_team"] == "Germany"))].tail(10)
h2hH, h2hA = 0.5, 0.5
if len(pair) > 0:
    wins_h, wins_a = 0, 0
    for _, r in pair.iterrows():
        if r["home_team"] == "Germany":
            if r["home_score"] > r["away_score"]:
                wins_h += 1
            elif r["home_score"] < r["away_score"]:
                wins_a += 1
        else:
            if r["away_score"] > r["home_score"]:
                wins_h += 1
            elif r["away_score"] < r["home_score"]:
                wins_a += 1
    h2hH = wins_h / len(pair)
    h2hA = wins_a / len(pair)

from datetime import datetime
today = datetime.now()
restH = 30
if home["last_match"]:
    last = datetime.fromisoformat(home["last_match"])
    restH = min((today - last).days, 365)
restA = 30
if away["last_match"]:
    last = datetime.fromisoformat(away["last_match"])
    restA = min((today - last).days, 365)

from src.features_v3 import _team_to_continent
contH = _team_to_continent("Germany")
contA = _team_to_continent("Uruguay")

# Build 45-dim vector matching the order in features_v3
feat = [
    int(neutral), tournamentW,                                          # 0, 1
    home["elo"], away["elo"], eloHEff - away["elo"],                     # 2, 3, 4
    home["re_elo"], away["re_elo"], re_eloHEff - away["re_elo"],         # 5, 6, 7
    home["elo_mom"], away["elo_mom"],                                    # 8, 9
    home["form3"], away["form3"], home["form5"], away["form5"],          # 10-13
    home["form10"], away["form10"],                                      # 14, 15
    home["gf5"], away["gf5"], home["ga5"], away["ga5"],                  # 16-19
    home["gd5"], away["gd5"],                                            # 20, 21
    home["home_form5"], away["away_form5"],                              # 22, 23
    h2hH, h2hA,                                                          # 24, 25
    restH, restA,                                                        # 26, 27
    home["win_streak"], away["win_streak"], home["unbeaten"],             # 28, 29, 30
    contH, contA, home["oppo_elo5"], away["oppo_elo5"],                  # 31-34
    1.0, 1.0,                                                            # 35, 36 tour_form_a/b
    home["w_form"], away["w_form"],                                      # 37, 38
    home["momentum"], away["momentum"],                                  # 39, 40
    home["wins_top10"], away["wins_top10"],                              # 41, 42
    home["wins_top20"], away["wins_top20"],                              # 43, 44
]

print(f"Feature vector length: {len(feat)} (expected 45)")

# Normalize using model stats
norm = bundle["norm_stats"]
mean = np.array(norm["mean"], dtype=np.float32)
std = np.array(norm["std"], dtype=np.float32)
feat_n = (np.array(feat, dtype=np.float32) - mean) / std

# Forward pass - we need to load the model
from src.train_v2 import FootballNet
arch = bundle["architecture"]
state_dict = bundle["models"][0]
model = FootballNet(
    in_dim=arch["in_dim"], hidden=arch["hidden"], n_blocks=arch["n_blocks"],
    dropout=arch.get("dropout", 0.2),
).cuda()
# Reconstruct state dict from list
sd = {k: torch.tensor(v) for k, v in state_dict.items()}
model.load_state_dict(sd)
model.eval()

with torch.no_grad():
    logits, goals = model(torch.from_numpy(feat_n).cuda().unsqueeze(0))
    probs = F.softmax(logits, dim=1).cpu().numpy()[0]
    goals_n = goals.cpu().numpy()[0]
gs = bundle["goal_stats"]
pred_hg = float(np.log1p(np.exp(goals_n[0] * gs["home_std"] + gs["home_mean"])))
pred_ag = float(np.log1p(np.exp(goals_n[1] * gs["away_std"] + gs["away_mean"])))

print()
print("=" * 50)
print(" V3 PROGNOSE: Germany vs Uruguay (WM, neutral)")
print("=" * 50)
print(f"  P(Draw):      {probs[0]*100:.1f}%")
print(f"  P(Sieg DE):   {probs[1]*100:.1f}%")
print(f"  P(Sieg UY):   {probs[2]*100:.1f}%")
print(f"  Expected:     {pred_hg:.2f} : {pred_ag:.2f}")
print()
if probs[1] > probs[2]:
    print("  >> Modellauswahl: SIEG DEUTSCHLAND ✓")
else:
    print(f"  >> Modellauswahl: {['Draw', 'Germany', 'Uruguay'][probs.argmax()]}")
