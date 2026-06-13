"""Sanity-Gates für das V9-Ensemble (E8Net+LGBM, kalibriert)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.predict_v8 import predict_match_v8

TESTS = [
    ("Germany", "San Marino", False, "home", 0.98, "Gate: >=98% home"),
    ("France", "San Marino", True, "home", 0.95, "Gate: >=95% home"),
    ("Brazil", "Germany", True, None, None, "Spread erwartet"),
    ("Argentina", "France", True, None, None, "Spread erwartet"),
    ("Germany", "Morocco", True, None, None, "~50-60% home"),
    ("Spain", "Panama", True, "home", 0.75, "Mismatch-Konfidenz"),
]

print("=" * 74)
print(" V9 Sanity Gates")
print("=" * 74)
all_ok = True
for h, a, neutral, gate_side, gate_min, note in TESTS:
    try:
        r = predict_match_v8(h, a, neutral=neutral, tag="v9")
    except Exception as e:
        print(f"{h} vs {a}: ERROR {e}")
        all_ok = False
        continue
    p = r["probabilities"]
    best = r["most_likely_scores"][0]
    gate_ok = True
    if gate_side:
        gate_ok = p[f"{gate_side}_win"] >= gate_min
        all_ok &= gate_ok
    print(f"{h:10s} vs {a:11s} H/U/A {p['home_win']*100:5.1f}/{p['draw']*100:4.1f}/"
          f"{p['away_win']*100:4.1f}%  best {best['home']}:{best['away']}"
          f"({best['prob']*100:.0f}%)  lam {r['lambdas']['home']:.2f}:{r['lambdas']['away']:.2f}"
          f"  [{'OK' if gate_ok else 'FAIL'}] {note}")
print(f"\n Model: {r['model_version']}")
print(f" Result: {'ALL GATES PASS' if all_ok else 'GATE FAILURES - CHECK ABOVE'}")
sys.exit(0 if all_ok else 1)
