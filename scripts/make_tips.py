"""
KickTipp-Tipps für anstehende Spiele generieren.

Filtert results.csv auf kommende Spiele (ohne Ergebnis) im Datumsfenster,
ruft predict_match_v6 (inkl. Odds-Blend + KickTipp Decision Layer) auf und
druckt eine Tipp-Tabelle mit erwarteten Punkten.

Usage:
    python -m scripts.make_tips                          # heute + morgen, FIFA World Cup
    python -m scripts.make_tips --date 2026-06-11        # bestimmter Tag
    python -m scripts.make_tips --date 2026-06-11 --days 3
    python -m scripts.make_tips --tournaments ""         # alle Turniere
"""
from __future__ import annotations

import argparse
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.features_v6 import predict_match_v6


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate KickTipp tips for upcoming fixtures")
    ap.add_argument("--date", type=str, default=str(date.today()), help="start date (YYYY-MM-DD)")
    ap.add_argument("--days", type=int, default=2, help="window length in days")
    ap.add_argument("--tournaments", type=str, default="FIFA World Cup",
                    help="substring filter on tournament ('' = all)")
    ap.add_argument("--model", type=str, default="v7", choices=["v7", "v8"],
                    help="v7 = predict_match_v6 (deployed), v8 = E8Net-Checkpoints")
    ap.add_argument("--tag", type=str, default="",
                    help="Checkpoint-Tag für --model v8 (models/v8_seed*_TAG.pt)")
    args = ap.parse_args()

    if args.model == "v8":
        from src.predict_v8 import predict_match_v8
        predict_fn = lambda h, a, neutral, tournament: predict_match_v8(
            h, a, neutral=neutral, tournament=tournament, tag=args.tag)
    else:
        predict_fn = lambda h, a, neutral, tournament: predict_match_v6(
            h, a, neutral=neutral, tournament=tournament)

    start = pd.Timestamp(args.date)
    end = start + timedelta(days=args.days)

    df = pd.read_csv(REPO_ROOT / "data" / "raw" / "results.csv", parse_dates=["date"])
    df = df[(df["date"] >= start) & (df["date"] < end) & df["home_score"].isna()]
    if args.tournaments:
        df = df[df["tournament"].str.contains(args.tournaments, case=False, na=False)]
    df = df.sort_values("date").reset_index(drop=True)

    if df.empty:
        print(f"Keine offenen Spiele im Fenster {start.date()} – {end.date()}.")
        return 0

    print("=" * 96)
    print(f" KickTipp-Tipps  {start.date()} – {(end - timedelta(days=1)).date()}   ({len(df)} Spiele)")
    print("=" * 96)

    total_ep = 0.0
    for _, row in df.iterrows():
        home, away = str(row["home_team"]), str(row["away_team"])
        try:
            pred = predict_fn(home, away, bool(row["neutral"]), str(row["tournament"]))
        except Exception as e:
            print(f"{row['date'].date()}  {home} – {away}: FEHLER {e}")
            continue

        p = pred["probabilities"]
        kt = pred.get("kicktipp_tip")
        best = pred["most_likely_scores"][0]
        odds_flag = "odds" if pred.get("odds_blended") else "    "

        if kt:
            tip_str = f"{kt['home']}:{kt['away']}"
            ep = float(kt["expected_points"])
            total_ep += ep
            alts = "  ".join(
                f"{a['tip'][0]}:{a['tip'][1]}({a['expected_points']:.2f})"
                for a in kt.get("alternatives", [])[:3]
            )
        else:
            tip_str, ep, alts = f"{best['home']}:{best['away']}", float("nan"), ""

        print(f"{row['date'].date()}  {home:>22s} – {away:<22s} [{odds_flag}]  "
              f"H/U/A {p['home_win']*100:4.1f}/{p['draw']*100:4.1f}/{p['away_win']*100:4.1f}%")
        print(f"{'':12s}TIPP {tip_str}  E[Pkt]={ep:.2f}   (Modus {best['home']}:{best['away']} "
              f"{best['prob']*100:.0f}%)   Alt: {alts}")

    print("-" * 96)
    print(f" Summe erwartete Punkte: {total_ep:.2f}   ({pred.get('model_version', '?')})")
    print("=" * 96)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
