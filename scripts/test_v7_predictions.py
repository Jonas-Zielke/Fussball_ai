"""Quick sanity test for V7 model predictions."""
import sys
sys.path.insert(0, "E:/Projects/Fussball_ai")
from src.features_v6 import predict_match_v6

tests = [
    ("Germany", "San Marino", False, "expect >98% home"),
    ("France", "San Marino", True, "expect >99% home"),
    ("Brazil", "Germany", True, "expect spread"),
    ("Argentina", "France", True, "expect spread"),
    ("Germany", "Morocco", True, "expect ~50-60% home"),
    ("USA", "Mexico", True, "expect ~30-40% home"),
    ("Spain", "Panama", True, "expect >90% home"),
]

print("=" * 70)
print(" V7 Model Prediction Tests")
print("=" * 70)

all_ok = True
for h, a, neutral, note in tests:
    try:
        r = predict_match_v6(h, a, neutral=neutral, tournament="FIFA World Cup")
        ph = r["probabilities"]["home_win"] * 100
        pd = r["probabilities"]["draw"] * 100
        pa = r["probabilities"]["away_win"] * 100
        best = r["most_likely_scores"][0]
        mv = r.get("model_version", "unknown")
        # Check coherence: winner of best score should match highest prob
        best_is_home_win = best["home"] > best["away"]
        best_is_draw = best["home"] == best["away"]
        prob_winner = "home" if ph > pd and ph > pa else ("draw" if pd > ph and pd > pa else "away")
        score_winner = "home" if best_is_home_win else ("draw" if best_is_draw else "away")
        coherent = (prob_winner == score_winner)
        flag = "OK" if coherent else "INCOHERENT!"
        if not coherent:
            all_ok = False
        print(f"{h:12s} vs {a:12s} [{note}]")
        print(f"  home={ph:.1f}%  draw={pd:.1f}%  away={pa:.1f}%  best={best['home']}:{best['away']}({best['prob']*100:.0f}%)  [{flag}]")
        print(f"  model: {mv}")
    except Exception as e:
        print(f"{h} vs {a}: ERROR - {e}")
        all_ok = False

print()
print("=" * 70)
print(f" Result: {'ALL COHERENT' if all_ok else 'SOME INCOHERENT - CHECK ABOVE'}")
print("=" * 70)
