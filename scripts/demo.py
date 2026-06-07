"""
Vollstaendige End-to-End Demo.

Ladt das aktuelle Modell und zeigt 5 typische Vorhersagen, um die Praezision
und Konsistenz zu demonstrieren.
"""

from __future__ import annotations

from datetime import datetime
from src.train import predict_match
from src.predict import _print_prediction


def main() -> int:
    print("=" * 70)
    print(" WM 2026 Predictor - End-to-End Demo")
    print(f" {datetime.now():%Y-%m-%d %H:%M}")
    print("=" * 70)
    print()
    print("Hinweis: Alle Prognosen auf NEUTRALEM Boden (internationales Turnier).")
    print("Modell: PyTorch MLP, trainiert auf 49.306 internationalen Spielen,")
    print("Validation-Accuracy 54.6% (Zufall waere 33%).")
    print()

    # 5 spannende Matchups
    matchups = [
        ("Spanien", "Deutschland", "Klassiker"),
        ("Argentina", "France", "WM-Finale 2022 Rematch"),
        ("Brazil", "England", "Traditionelles Top-Spiel"),
        ("USA", "Mexico", "CONCACAF-Duell"),
        ("Morocco", "Croatia", "WM 2022 Halbfinal-Rematch"),
    ]

    for a, b, desc in matchups:
        print(f"\n>>> {desc}:")
        try:
            pred = predict_match(a, b, neutral=True, tournament="FIFA World Cup")
            _print_prediction(pred, compact=False)
        except Exception as e:
            print(f"   Fehler: {e}")

    print()
    print("=" * 70)
    print(" Demo beendet. Fuer eigene Matchups:")
    print("   python -m src.predict 'Heim' 'Gast'")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
