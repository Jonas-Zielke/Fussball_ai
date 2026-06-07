"""Quick test of V2 model on Germany vs Uruguay and extreme mismatch."""
import json
from src.train_v2 import predict_match_v2

p = predict_match_v2("Germany", "Uruguay", neutral=True, tournament="FIFA World Cup")
print("Germany vs Uruguay:")
print(json.dumps(p, indent=2, ensure_ascii=False))
print()
p2 = predict_match_v2("Spain", "Bhutan", neutral=True, tournament="FIFA World Cup")
print("Spain vs Bhutan:")
print(f"  Spain {p2['probabilities']['home_win']*100:.1f}% | "
      f"Draw {p2['probabilities']['draw']*100:.1f}% | "
      f"Bhutan {p2['probabilities']['away_win']*100:.1f}%")
print(f"  Expected score: {p2['expected_score']['display']}")
print()
p3 = predict_match_v2("Spain", "Argentina", neutral=True, tournament="FIFA World Cup")
print("Spain vs Argentina:")
print(f"  Spain {p3['probabilities']['home_win']*100:.1f}% | "
      f"Draw {p3['probabilities']['draw']*100:.1f}% | "
      f"Argentina {p3['probabilities']['away_win']*100:.1f}%")
print(f"  Expected score: {p3['expected_score']['display']}")
