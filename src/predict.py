"""
Inference CLI fuer den WM 2026 Predictor.

Aufruf:
    python -m src.predict "Germany" "Brazil"
    python -m src.predict "Germany" "Brazil" --neutral
    python -m src.predict "Germany" "Brazil" --tournament "FIFA World Cup" --tournament-override
    python -m src.predict --list-top 30
    python -m src.predict --simulate-wm  # simuliert alle 12 WM-Gruppen

Ohne Argumente wird ein interaktiver Modus gestartet.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

from .team_normalize import normalize_team_name
from .features_v6 import predict_match_v6, get_current_team_ratings_v6

REPO_ROOT = Path(__file__).resolve().parent.parent


def _resolve(name: str) -> str:
    """Loest einen Teamnamen auf und wirft einen hilfreichen Fehler, falls er unbekannt ist."""
    norm = normalize_team_name(name)
    state = get_current_team_ratings_v6()
    if norm not in state:
        # Fuzzy Vorschlag
        from difflib import get_close_matches
        suggestions = get_close_matches(norm, list(state.keys()), n=5, cutoff=0.6)
        raise SystemExit(
            f"FEHLER: Unbekanntes Team '{name}' (normalisiert: '{norm}').\n"
            f"Meintest du vielleicht: {', '.join(suggestions) if suggestions else '(keine Vorschlaege)'} ?\n"
            f"Mit --list-top 30 siehst du die staerksten Teams."
        )
    return norm


def _print_prediction(pred: dict, compact: bool = False) -> None:
    p = pred["probabilities"]
    # Wir sortieren die Wahrscheinlichkeiten
    items = [
        ("Unentschieden", p["draw"]),
        (f"Sieg {pred['home']}", p["home_win"]),
        (f"Sieg {pred['away']}", p["away_win"]),
    ]
    items.sort(key=lambda x: -x[1])

    print("=" * 70)
    print(f"  PROGNOSE:  {pred['home']}  vs  {pred['away']}")
    print(f"  (Stand: {pred['as_of'][:10]}  |  Turnier: {pred['tournament']}  |  "
          f"Heimrecht: {'neutral' if pred['neutral'] else 'Heim'})")
    print(f"  Elo: {pred['home']} {pred['elo_home']:.0f}  vs  {pred['away']} {pred['elo_away']:.0f}")
    print("-" * 70)
    for label, prob in items:
        bar_len = int(round(prob * 40))
        bar = "#" * bar_len + "-" * (40 - bar_len)
        print(f"  {label:<30}  {prob*100:5.1f}%  |{bar}|")
    if "expected_score" in pred:
        es = pred["expected_score"]
        print(f"  Erwartetes Ergebnis:  {pred['home']} {es['home_goals']:.2f}  :  "
              f"{es['away_goals']:.2f}  {pred['away']}")
    if "most_likely_scores" in pred:
        print("  Wahrscheinlichste exakte Ergebnisse (Poisson-Modell):")
        for s in pred["most_likely_scores"]:
            print(f"    {pred['home']} {s['home']} : {s['away']} {pred['away']}  "
                  f"->  {s['prob']*100:5.1f}%")
    print("-" * 70)
    print(f"  >> Modellauswahl: {pred['argmax_label']}")
    if pred.get("model_version"):
        print(f"  (Modell: {pred['model_version']}, Ensemble={pred.get('ensemble_size', 1)})")
    print("=" * 70)


def cmd_predict(args) -> int:
    home = _resolve(args.home)
    away = _resolve(args.away)
    pred = predict_match_v6(
        home=home, away=away, neutral=args.neutral, tournament=args.tournament,
    )
    _print_prediction(pred, compact=args.compact)
    return 0


def cmd_list_top(args) -> int:
    state = get_current_team_ratings_v6()
    items = sorted(state.items(), key=lambda x: -x[1]["elo"])
    n = args.list_top
    print("=" * 70)
    print(f" Top {n} Teams nach Elo-Rating (Stand: {datetime.now():%Y-%m-%d})")
    print("=" * 70)
    print(f"{'#':>3}  {'Team':<25} {'Elo':>8}  {'Form5':>6}  {'GF5':>5}  {'GA5':>5}")
    for i, (name, st) in enumerate(items[:n], 1):
        print(f"{i:>3}  {name:<25} {st['elo']:>8.1f}  {st['form5']:>5.2f}  {st['gf5']:>4.2f}  {st['ga5']:>4.2f}")
    print(f"\nGesamt: {len(state)} Teams")
    return 0


def cmd_interactive() -> int:
    print("=" * 70)
    print("  WM 2026 Predictor - Interaktiv")
    print("=" * 70)
    print("  Format: '<Heimteam> vs <Gastteam>'  (oder 'q' zum Beenden)")
    print("  Beispiele:")
    print("    Germany vs Brazil")
    print("    France vs Argentina")
    print("    Spain vs England")
    print("  Mit Suffix '/n' = neutraler Boden (z.B. 'Germany vs Brazil /n')")
    print("-" * 70)
    while True:
        try:
            line = input("  > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not line:
            continue
        if line.lower() in {"q", "quit", "exit"}:
            return 0
        if " vs " not in line.lower():
            print("  Bitte Format '<TeamA> vs <TeamB>' nutzen.")
            continue
        # Optional neutral marker
        neutral = True  # default fuer internationale Turniere
        if line.endswith("/n") or line.endswith("/neutral"):
            neutral = True
            line = line.rsplit("/", 1)[0].strip()
        elif line.endswith("/h") or line.endswith("/home"):
            neutral = False
            line = line.rsplit("/", 1)[0].strip()
        try:
            a, b = line.split(" vs ", 1)
        except ValueError:
            print("  Konnte nicht parsen.")
            continue
        a = a.strip()
        b = b.strip()
        try:
            a_n = _resolve(a)
            b_n = _resolve(b)
        except SystemExit as e:
            print(str(e).replace("FEHLER: ", ""))
            continue
        try:
            pred = predict_match_v6(a_n, b_n, neutral=neutral, tournament="FIFA World Cup")
        except Exception as exc:
            print(f"  Fehler: {exc}")
            continue
        _print_prediction(pred, compact=False)
    return 0


def cmd_simulate_wm(args) -> int:
    """Simuliert alle Gruppenspiele der WM 2026 (12 Gruppen, je 4 Teams, aber wir
    nehmen hier eine plausible Top-Setzung)."""
    # Vereinfachte 12er-Gruppen mit Top-Teams. Das ist NICHT die offizielle Auslosung,
    # sondern eine sinnvolle Setzung der staerksten Teams fuer eine Demo.
    groups = {
        "A": ["Mexico", "South Korea", "Denmark", "Australia"],
        "B": ["Canada", "Switzerland", "Norway", "Morocco"],
        "C": ["Brazil", "Scotland", "Egypt", "Iran"],
        "D": ["United States", "Paraguay", "Tunisia", "Croatia"],
        "E": ["Germany", "Uruguay", "Senegal", "Saudi Arabia"],
        "F": ["Netherlands", "Japan", "Ecuador", "Ivory Coast"],
        "G": ["Argentina", "Poland", "Ghana", "Australia"],
        "H": ["Spain", "Belgium", "Algeria", "Wales"],
        "I": ["France", "Colombia", "Austria", "Cameroon"],
        "J": ["Portugal", "Mexico", "South Africa", "Romania"],
        "K": ["England", "Italy", "Nigeria", "Peru"],
        "L": ["Turkey", "Chile", "China PR", "New Zealand"],
    }
    # Korrektur: Duplikate raus
    groups["G"] = ["Argentina", "Poland", "Ghana", "Ivory Coast"]
    from .team_normalize import normalize_team_name
    groups = {k: [normalize_team_name(t) for t in v] for k, v in groups.items()}

    print("=" * 70)
    print("  WM 2026 Simulator (Demo-Setzung) - nutzt vorhergesagte Tore (Poisson)")
    print("=" * 70)

    standings: dict[str, dict[str, dict]] = {}
    for gname, teams in groups.items():
        print(f"\nGruppe {gname}: {', '.join(teams)}")
        print("-" * 70)
        standings[gname] = {t: {"pts": 0, "gd": 0, "gf": 0, "ga": 0} for t in teams}
        from itertools import combinations
        for a, b in combinations(teams, 2):
            for h, aw in [(a, b), (b, a)]:
                try:
                    pred = predict_match_v6(h, aw, neutral=True, tournament="FIFA World Cup")
                except Exception as exc:
                    print(f"  Fehler bei {h} vs {aw}: {exc}")
                    continue
                p = pred["probabilities"]
                es = pred.get("expected_score", {})
                exp_h = es.get("home_goals", 1.5)
                exp_a = es.get("away_goals", 1.0)
                # Sample aus Poisson mit den erwarteten Toren als lambda
                rng = np.random.default_rng(seed=hash((gname, h, aw)) % (2**32))
                # Cap lambdas for stability
                exp_h = float(min(max(exp_h, 0.1), 6.0))
                exp_a = float(min(max(exp_a, 0.1), 6.0))
                sh_num = int(rng.poisson(exp_h))
                sa_num = int(rng.poisson(exp_a))
                # Realismus-Cap (Fussball-Spiele enden fast nie 7:0)
                sh_num = min(sh_num, 7)
                sa_num = min(sa_num, 7)
                gd_display = f"{sh_num}:{sa_num}"

                # Update standings
                if sh_num > sa_num:
                    standings[gname][h]["pts"] += 3
                elif sh_num < sa_num:
                    standings[gname][aw]["pts"] += 3
                else:
                    standings[gname][h]["pts"] += 1
                    standings[gname][aw]["pts"] += 1

                standings[gname][h]["gf"] += sh_num
                standings[gname][h]["ga"] += sa_num
                standings[gname][aw]["gf"] += sa_num
                standings[gname][aw]["ga"] += sh_num
                standings[gname][h]["gd"] = standings[gname][h]["gf"] - standings[gname][h]["ga"]
                standings[gname][aw]["gd"] = standings[gname][aw]["gf"] - standings[gname][aw]["ga"]

                prob_str = (f"E[{h} {exp_h:.1f} : {aw} {exp_a:.1f}]  "
                            f"Draw {p['draw']*100:4.1f}% | {h} {p['home_win']*100:4.1f}% | "
                            f"{aw} {p['away_win']*100:4.1f}%")
                print(f"  {h:<22} vs {aw:<22} -> {gd_display:>5}  ({prob_str})")

        # Tabelle
        print(f"\n  Tabelle Gruppe {gname}:")
        sorted_t = sorted(standings[gname].items(), key=lambda x: (-x[1]["pts"], -x[1]["gd"], -x[1]["gf"]))
        print(f"  {'Team':<22} {'Pkt':>4}  {'T+':>4} {'T-':>4} {'Diff':>5}")
        for t, st in sorted_t:
            print(f"  {t:<22} {st['pts']:>4}  {st['gf']:>4} {st['ga']:>4} {st['gd']:>+5}")

    return 0


def cmd_sweep(args) -> int:
    """Berechne paarweise Sieg-Wahrscheinlichkeiten fuer eine Liste von Teams."""
    from itertools import combinations
    if args.sweep_file:
        with open(args.sweep_file, "r", encoding="utf-8") as fh:
            teams = [normalize_team_name(line.strip()) for line in fh if line.strip()]
    else:
        teams = [normalize_team_name(t) for t in args.sweep.split(",")]
    teams = [_resolve(t) for t in teams]
    print("=" * 70)
    print(f"  Paarweise Prognose fuer {len(teams)} Teams (neutraler Boden)")
    print("=" * 70)
    header = f"{'':>22} " + " ".join(f"{t[:5]:>6}" for t in teams)
    print(header)
    for t1 in teams:
        row = [f"{t1:>22}"]
        for t2 in teams:
            if t1 == t2:
                row.append("    - ")
                continue
            pred = predict_match_v6(t1, t2, neutral=True, tournament="FIFA World Cup")
            row.append(f"{pred['probabilities']['home_win']*100:5.1f}%")
        print(" ".join(row))
    return 0


def main():
    p = argparse.ArgumentParser(
        description="WM 2026 Predictor - Inference CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Beispiele:\n"
            "  python -m src.predict 'Germany' 'Brazil'\n"
            "  python -m src.predict 'Germany' 'Brazil' --neutral\n"
            "  python -m src.predict --list-top 20\n"
            "  python -m src.predict --sweep 'Germany,Brazil,Argentina,France'\n"
            "  python -m src.predict --simulate-wm\n"
        ),
    )
    p.add_argument("home", nargs="?", help="Heimteam")
    p.add_argument("away", nargs="?", help="Gastteam")
    p.add_argument("--neutral", action="store_true", default=True,
                   help="neutraler Boden (default fuer internationale Spiele)")
    p.add_argument("--home-court", action="store_true",
                   help="stattdessen Heimrecht fuer Team A")
    p.add_argument("--tournament", default="FIFA World Cup",
                   help="Turniername fuer K-Faktor (default: FIFA World Cup)")
    p.add_argument("--compact", action="store_true")
    p.add_argument("--list-top", type=int, metavar="N", help="zeige Top N Teams nach Elo")
    p.add_argument("--sweep", type=str, metavar="TEAMS",
                   help="Komma-getrennte Teamliste fuer paarweise Prognose")
    p.add_argument("--sweep-file", type=str, metavar="FILE",
                   help="Datei mit einem Teamnamen pro Zeile")
    p.add_argument("--simulate-wm", action="store_true",
                   help="Simuliere alle Gruppenspiele einer Demo-WM-Setzung")
    p.add_argument("--json", action="store_true", help="Output als JSON")

    args = p.parse_args()
    if args.home_court:
        args.neutral = False
    if args.json:
        # JSON Modus
        if args.home and args.away:
            home = _resolve(args.home)
            away = _resolve(args.away)
            pred = predict_match_v6(home, away, neutral=args.neutral, tournament=args.tournament)
            print(json.dumps(pred, indent=2, ensure_ascii=False))
            return 0

    if args.list_top:
        return cmd_list_top(args)
    if args.simulate_wm:
        return cmd_simulate_wm(args)
    if args.sweep or args.sweep_file:
        return cmd_sweep(args)
    if args.home and args.away:
        return cmd_predict(args)

    # Fallback: interaktiv
    return cmd_interactive()


if __name__ == "__main__":
    raise SystemExit(main())
