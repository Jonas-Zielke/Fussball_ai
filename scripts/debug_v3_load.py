"""Debug V3 prediction in Python by loading the JSON model directly."""
import json
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path

# Load the V3 model JSON
bundle = json.loads(Path("E:/Profilov2/public/data/wm-predictor/model.json").read_text())
print("Architecture:", bundle["architecture"])
print("N features:", bundle["n_features"])

# Load V3 team state
import sys
sys.path.insert(0, "E:/Projects/Fussball_ai")
from src.features_v3 import get_current_team_ratings_v3
state = get_current_team_ratings_v3()
g = state["Germany"]
u = state["Uruguay"]
print(f"Germany re_elo: {g['re_elo']:.0f}, Uruguay re_elo: {u['re_elo']:.0f}, diff: {g['re_elo']-u['re_elo']:+.0f}")

# Build the model from folded weights
# We need a model that uses input_w, input_b, blockN_w1/b1/w2/b2, cls_w1/b1/w2/b2, reg_w1/b1/w2/b2
arch = bundle["architecture"]
hidden = arch["hidden"]
n_blocks = arch["n_blocks"]

# Build a class that takes the folded weights
class FoldedFootballNet(torch.nn.Module):
    def __init__(self, hidden, n_blocks, n_classes=3):
        super().__init__()
        # We construct the model exactly like FootballNet but use the folded weights
        # We do this by creating the SAME architecture, then loading the folded weights
        # into the corresponding locations (BN identity)
        # Simpler: just use the unfolded FootballNet with BN = identity
        from src.train_v2 import FootballNet, ResidualBlock
        self.model = FootballNet(in_dim=bundle["n_features"], hidden=hidden, n_blocks=n_blocks, dropout=0.0)
        # Manually fold BNs
        self.fold_bns()
    def fold_bns(self):
        # Make all BNs identity-like
        for m in self.model.modules():
            if isinstance(m, torch.nn.BatchNorm1d):
                m.weight.data.fill_(1.0)
                m.bias.data.fill_(0.0)
                m.running_mean.fill_(0.0)
                m.running_var.fill_(1.0)
    def forward(self, x):
        return self.model(x)

m = FoldedFootballNet(hidden, n_blocks)
m.eval()

# Load folded weights
sd = bundle["models"][0]
new_sd = {}
# Map folded to unfolded
new_sd["model.input_proj.0.weight"] = torch.tensor(sd["input_w"])
new_sd["model.input_proj.0.bias"] = torch.tensor(sd["input_b"])
for i in range(n_blocks):
    new_sd[f"model.blocks.{i}.lin1.weight"] = torch.tensor(sd[f"block{i}_w1"])
    new_sd[f"model.blocks.{i}.lin1.bias"] = torch.tensor(sd[f"block{i}_b1"])
    new_sd[f"model.blocks.{i}.lin2.weight"] = torch.tensor(sd[f"block{i}_w2"])
    new_sd[f"model.blocks.{i}.lin2.bias"] = torch.tensor(sd[f"block{i}_b2"])
# Heads
new_sd["model.cls_head.0.weight"] = torch.tensor(sd["cls_w1"])
new_sd["model.cls_head.0.bias"] = torch.tensor(sd["cls_b1"])
new_sd["model.cls_head.3.weight"] = torch.tensor(sd["cls_w2"])
new_sd["model.cls_head.3.bias"] = torch.tensor(sd["cls_b2"])
new_sd["model.reg_head.0.weight"] = torch.tensor(sd["reg_w1"])
new_sd["model.reg_head.0.bias"] = torch.tensor(sd["reg_b1"])
new_sd["model.reg_head.3.weight"] = torch.tensor(sd["reg_w2"])
new_sd["model.reg_head.3.bias"] = torch.tensor(sd["reg_b2"])

m.load_state_dict(new_sd, strict=False)
m.cuda()
m.eval()

# Build feature vector (45-dim) - same as JS
neutral = True
tournament = "FIFA World Cup"
tournamentW = 60
eloHEff = g["elo"] + (0 if neutral else 80)
reEloHEff = g["re_elo"] + (0 if neutral else 80)

import pandas as pd
from datetime import datetime
df = pd.read_csv("E:/Projects/Fussball_ai/data/raw/results.csv", parse_dates=["date"])
df = df.dropna(subset=["home_score", "away_score"])
pair = df[((df["home_team"] == "Germany") & (df["away_team"] == "Uruguay")) |
         ((df["home_team"] == "Uruguay") & (df["away_team"] == "Germany"))].tail(10)
h2hH, h2hA = 0.5, 0.5
if len(pair) > 0:
    wins_h, wins_a = 0, 0
    for _, r in pair.iterrows():
        if r["home_team"] == "Germany":
            if r["home_score"] > r["away_score"]: wins_h += 1
            elif r["home_score"] < r["away_score"]: wins_a += 1
        else:
            if r["away_score"] > r["home_score"]: wins_h += 1
            elif r["away_score"] < r["home_score"]: wins_a += 1
    h2hH = wins_h / len(pair)
    h2hA = wins_a / len(pair)
today = datetime.now()
restH = 30
if g["last_match"]:
    last = datetime.fromisoformat(g["last_match"])
    restH = min((today - last).days, 365)
restA = 30
if u["last_match"]:
    last = datetime.fromisoformat(u["last_match"])
    restA = min((today - last).days, 365)
from src.features_v3 import _team_to_continent
contH = _team_to_continent("Germany")
contA = _team_to_continent("Uruguay")

feat = [
    int(neutral), tournamentW,
    g["elo"], u["elo"], eloHEff - u["elo"],
    g["re_elo"], u["re_elo"], reEloHEff - u["re_elo"],
    g["elo_mom"], u["elo_mom"],
    g["form3"], u["form3"], g["form5"], u["form5"], g["form10"], u["form10"],
    g["gf5"], u["gf5"], g["ga5"], u["ga5"], g["gd5"], u["gd5"],
    g["home_form5"], u["away_form5"],
    h2hH, h2hA,
    restH, restA,
    g["win_streak"], u["win_streak"], g["unbeaten"],
    contH, contA, g["oppo_elo5"], u["oppo_elo5"],
    1.0, 1.0,
    g["w_form"], u["w_form"],
    g["momentum"], u["momentum"],
    g["wins_top10"], u["wins_top10"],
    g["wins_top20"], u["wins_top20"],
]
print(f"Feature vector: {len(feat)} dims")

norm = bundle["norm_stats"]
mean = np.array(norm["mean"], dtype=np.float32)
std = np.array(norm["std"], dtype=np.float32)
feat_n = (np.array(feat, dtype=np.float32) - mean) / std

# Show some normalized values
print(f"re_elo_diff (idx 7): raw={feat[7]:.1f}, normalized={feat_n[7]:.2f}")
print(f"re_elo_a (idx 5): raw={feat[5]:.1f}, normalized={feat_n[5]:.2f}")
print(f"re_elo_b (idx 6): raw={feat[6]:.1f}, normalized={feat_n[6]:.2f}")
print(f"momentum_a (idx 39): raw={feat[39]:.2f}, normalized={feat_n[39]:.2f}")
print(f"momentum_b (idx 40): raw={feat[40]:.2f}, normalized={feat_n[40]:.2f}")
print(f"w_form_a (idx 37): raw={feat[37]:.2f}, normalized={feat_n[37]:.2f}")

with torch.no_grad():
    logits, goals = m(torch.from_numpy(feat_n).cuda().unsqueeze(0))
    probs = F.softmax(logits, dim=1).cpu().numpy()[0]
print()
print(f"Raw logits: {logits.cpu().numpy()[0]}")
print(f"Probabilities: Draw={probs[0]:.3f}, Home={probs[1]:.3f}, Away={probs[2]:.3f}")
