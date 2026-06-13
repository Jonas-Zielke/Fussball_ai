"""
KickTipp Decision-Layer Evaluation.

Replays completed Val matches (>=2024) through V7 predictions and measures
KickTipp points/match for three strategies:
  1. optimal_tip  — argmax expected KickTipp points (the new decision layer)
  2. mode_tip     — most likely score (current V7 behaviour)
  3. argmax_1x2   — always tip generic score for the argmax tendency (1:0 / 0:0 / 0:1)

Usage:
    python -m scripts.eval_kicktipp
    python -m scripts.eval_kicktipp --n 200          # limit to first N matches
    python -m scripts.eval_kicktipp --tournaments wm  # filter to FIFA World Cup
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.features_v6 import predict_match_v6
from src.kicktipp import load_scheme, points


def _odds_probs(pred: dict) -> dict[str, float]:
    mp = pred.get("market_probs")
    if mp:
        return {"home": float(mp["home_win"]),
                "draw": float(mp["draw"]),
                "away": float(mp["away_win"])}
    p = pred["probabilities"]
    return {"home": float(p["home_win"]),
            "draw": float(p["draw"]),
            "away": float(p["away_win"])}


_ARGMAX_SCORE = {
    "home": (1, 0),
    "draw": (0, 0),
    "away": (0, 1),
}


def _argmax_tendency(pred: dict) -> tuple[int, int]:
    p = pred["probabilities"]
    if p["home_win"] >= p["draw"] and p["home_win"] >= p["away_win"]:
        return _ARGMAX_SCORE["home"]
    if p["draw"] >= p["home_win"] and p["draw"] >= p["away_win"]:
        return _ARGMAX_SCORE["draw"]
    return _ARGMAX_SCORE["away"]


def evaluate(df: pd.DataFrame, scheme, verbose: bool = False, predict_fn=None) -> dict:
    if predict_fn is None:
        predict_fn = lambda h, a, neutral, tournament: predict_match_v6(
            h, a, neutral=neutral, tournament=tournament)
    records = []
    n = len(df)
    errors = 0

    for i, (_, row) in enumerate(df.iterrows()):
        if i % 50 == 0:
            print(f"  {i}/{n} ...", end="\r", flush=True)
        try:
            pred = predict_fn(
                row["home_team"],
                row["away_team"],
                bool(row.get("neutral", True)),
                str(row.get("tournament", "FIFA World Cup")),
            )
        except Exception as e:
            errors += 1
            continue

        actual = (int(row["home_score"]), int(row["away_score"]))
        ops = _odds_probs(pred)

        # Strategy 1 — optimal tip
        kt = pred.get("kicktipp_tip")
        if kt:
            opt_tip = (int(kt["home"]), int(kt["away"]))
        else:
            opt_tip = (int(pred["most_likely_scores"][0]["home"]),
                       int(pred["most_likely_scores"][0]["away"]))

        # Strategy 2 — mode tip
        mode_tip = (int(pred["most_likely_scores"][0]["home"]),
                    int(pred["most_likely_scores"][0]["away"]))

        # Strategy 3 — argmax tendency
        argmax_tip = _argmax_tendency(pred)

        pts_opt = points(opt_tip, actual, ops, scheme)
        pts_mode = points(mode_tip, actual, ops, scheme)
        pts_argmax = points(argmax_tip, actual, ops, scheme)

        rec = {
            "date": row["date"],
            "home": row["home_team"],
            "away": row["away_team"],
            "actual_h": actual[0],
            "actual_a": actual[1],
            "opt_h": opt_tip[0], "opt_a": opt_tip[1],
            "mode_h": mode_tip[0], "mode_a": mode_tip[1],
            "argmax_h": argmax_tip[0], "argmax_a": argmax_tip[1],
            "pts_opt": pts_opt,
            "pts_mode": pts_mode,
            "pts_argmax": pts_argmax,
            "has_odds": pred.get("market_probs") is not None,
            "expected_pts_opt": kt["expected_points"] if kt else None,
            "tournament": row.get("tournament", ""),
        }
        records.append(rec)

        if verbose and (opt_tip != mode_tip):
            diff = pts_opt - pts_mode
            sign = "+" if diff >= 0 else ""
            print(f"  {row['home_team']} {actual[0]}:{actual[1]} {row['away_team']}"
                  f"  | opt={opt_tip[0]}:{opt_tip[1]}  mode={mode_tip[0]}:{mode_tip[1]}"
                  f"  | pts opt={pts_opt:.0f} mode={pts_mode:.0f} ({sign}{diff:.0f})")

    print(f"\n  Done. {len(records)} matches processed, {errors} errors.")
    return {"records": records}


def _print_summary(records: list[dict]) -> None:
    if not records:
        print("No records.")
        return
    df = pd.DataFrame(records)
    n = len(df)
    print()
    print("=" * 55)
    print(f"KickTipp Evaluation — {n} matches")
    print("=" * 55)
    for col, label in [("pts_opt", "Optimal-Tip  (new)"),
                        ("pts_mode", "Mode-Tip     (V7 old)"),
                        ("pts_argmax", "ArgMax-1x2   (baseline)")]:
        mean = df[col].mean()
        total = df[col].sum()
        exact = (df[col] >= 4).mean() * 100  # >=4 pts = exact or higher
        tend = (df[col] >= 2).mean() * 100
        print(f"  {label:25s}  {mean:5.3f} pts/match  "
              f"(total {total:.0f}  exact% {exact:.1f}%  tendency% {tend:.1f}%)")

    # Delta optimal vs mode
    delta = df["pts_opt"] - df["pts_mode"]
    print()
    print(f"  Optimal vs Mode  delta/match: {delta.mean():+.4f}")
    print(f"  Opt>Mode: {(delta>0).sum()}, Opt<Mode: {(delta<0).sum()}, Tie: {(delta==0).sum()}")
    tips_differed = ((df['opt_h'] != df['mode_h']) | (df['opt_a'] != df['mode_a'])).sum()
    print(f"  Tip differed: {tips_differed} / {n}")

    if df["has_odds"].any():
        sub = df[df["has_odds"]]
        print()
        print(f"  --- Matches with bookmaker odds ({len(sub)}) ---")
        for col, label in [("pts_opt", "Optimal-Tip"),
                            ("pts_mode", "Mode-Tip")]:
            print(f"    {label:15s} {sub[col].mean():.3f} pts/match")


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate KickTipp decision layer")
    ap.add_argument("--n", type=int, default=0, help="limit to first N matches (0=all)")
    ap.add_argument("--tournaments", type=str, default="", help="comma-separated tournament filter substrings")
    ap.add_argument("--from-date", type=str, default="2024-01-01")
    ap.add_argument("--model", type=str, default="v7", choices=["v7", "v8"])
    ap.add_argument("--tag", type=str, default="", help="Checkpoint-Tag für --model v8")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    predict_fn = None
    if args.model == "v8":
        from src.predict_v8 import predict_match_v8
        predict_fn = lambda h, a, neutral, tournament: predict_match_v8(
            h, a, neutral=neutral, tournament=tournament, tag=args.tag)

    df = pd.read_csv(REPO_ROOT / "data" / "raw" / "results.csv")
    df = df[(df["date"] >= args.from_date)
            & df["home_score"].notna()
            & df["away_score"].notna()]

    if args.tournaments:
        filters = [t.strip().lower() for t in args.tournaments.split(",")]
        mask = df["tournament"].str.lower().apply(
            lambda t: any(f in t for f in filters)
        )
        df = df[mask]

    if args.n and args.n > 0:
        df = df.head(args.n)

    df = df.reset_index(drop=True)
    print(f"Evaluating {len(df)} matches (from {args.from_date})...")

    scheme = load_scheme()
    t0 = time.time()
    result = evaluate(df, scheme, verbose=args.verbose, predict_fn=predict_fn)
    elapsed = time.time() - t0
    print(f"  Time: {elapsed:.1f}s ({elapsed/len(result['records']):.2f}s/match)")

    _print_summary(result["records"])

    # Optionally save
    out_path = REPO_ROOT / "data" / "processed" / "eval_kicktipp.parquet"
    pd.DataFrame(result["records"]).to_parquet(out_path, index=False)
    print(f"\n  Saved to {out_path}")


if __name__ == "__main__":
    main()
