"""
One-command tournament refresh for the live WC 2026.

Chains the daily update pipeline so a single command keeps tips fresh as
martj42 publishes new results (it updates hourly during the tournament).

Fast daily path (default) — everything inference needs:
  1. data_download --force   refresh results.csv + shootouts.csv from martj42
  2. fetch_wm2026_odds        refresh bookmaker odds (live if WM_ODDS_API_KEY set)
  3. export_v8_state          rebuild per-team V8/V9 sequence+squad state
  6. make_tips                print KickTipp tips for the upcoming window

Heavy path (--rebuild-features) — only needed before a retrain or backtest:
  4. features_v6 + features_v8 rebuild training tables
  5. export_v8_tensors        refresh browser tensors

The deployed model is V9 (3x E8Net + LGBM + affine calibration), invoked as
``--model v8 --tag v9``; the script auto-detects the v9 checkpoints and falls
back to the V7 path if they are missing.

Usage:
    python -m scripts.refresh_tournament                    # fast daily refresh + V9 tips
    python -m scripts.refresh_tournament --days 3
    python -m scripts.refresh_tournament --rebuild-features # full rebuild (slow, for retrain/backtest)
    python -m scripts.refresh_tournament --no-tips          # just refresh data/state/odds
    python -m scripts.refresh_tournament --goalscorers      # also pull goalscorers.csv
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from datetime import date
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def run_step(title: str, module: str, extra: list[str], *, fatal: bool = True) -> bool:
    """Run ``python -m <module> <extra>`` from the repo root, timed and logged.

    Returns True on success. If ``fatal`` and the step fails, aborts the whole
    refresh (a stale-data deploy mid-contest is worse than a loud failure).
    """
    print("\n" + "=" * 78)
    print(f" STEP: {title}")
    print("=" * 78, flush=True)
    t0 = time.time()
    proc = subprocess.run(
        [sys.executable, "-m", module, *extra],
        cwd=str(REPO_ROOT),
    )
    dt = time.time() - t0
    if proc.returncode != 0:
        msg = f" [FAIL] {title} (exit {proc.returncode}, {dt:.1f}s)"
        if fatal:
            print(msg + " — aborting refresh.", flush=True)
            raise SystemExit(proc.returncode)
        print(msg + " — continuing.", flush=True)
        return False
    print(f" [ok] {title} ({dt:.1f}s)", flush=True)
    return True


def _v9_available(tag: str) -> bool:
    return (REPO_ROOT / "models" / f"v8_seed0_{tag}.pt").exists()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Daily WC 2026 refresh pipeline")
    ap.add_argument("--date", type=str, default=str(date.today()),
                    help="tip window start date (YYYY-MM-DD)")
    ap.add_argument("--days", type=int, default=2, help="tip window length in days")
    ap.add_argument("--tournaments", type=str, default="FIFA World Cup",
                    help="substring filter for make_tips ('' = all)")
    ap.add_argument("--model", type=str, default="auto", choices=["auto", "v7", "v8"],
                    help="auto = v8/--tag if checkpoints exist, else v7")
    ap.add_argument("--tag", type=str, default="v9", help="V8 checkpoint tag")
    ap.add_argument("--rebuild-features", action="store_true",
                    help="also rebuild features_v6/v8 + browser tensors (slow)")
    ap.add_argument("--no-tips", action="store_true", help="skip the make_tips step")
    ap.add_argument("--goalscorers", action="store_true", help="also pull goalscorers.csv")
    args = ap.parse_args(argv)

    overall_t0 = time.time()

    # 1) Fresh results + shootouts from martj42 (always force during the tournament)
    dl_extra = ["--force"] + (["--goalscorers"] if args.goalscorers else [])
    run_step("Refresh results.csv + shootouts.csv (martj42)", "src.data_download", dl_extra)

    # 2) Refresh bookmaker odds (uses WM_ODDS_API_KEY env if present, else fallback).
    #    Non-fatal: a stale odds file should not block fresh-state tips.
    run_step("Refresh bookmaker odds", "scripts.fetch_wm2026_odds", [], fatal=False)

    # 3) Rebuild V8/V9 per-team sequence+squad state (consumed by predict_v8).
    run_step("Export V8 final-state (sequence + squad)", "scripts.export_v8_state", [])

    # 4/5) Heavy rebuild — only needed before retrain/backtest.
    if args.rebuild_features:
        run_step("Rebuild features_v6 (static)", "src.features_v6", [])
        run_step("Rebuild features_v8 (sequence + squad)", "src.features_v8", [])
        run_step("Export browser tensors", "scripts.export_v8_tensors", [], fatal=False)

    # 6) Tips for the upcoming window.
    if not args.no_tips:
        model = args.model
        tag = args.tag
        if model == "auto":
            if _v9_available(tag):
                model = "v8"
            else:
                print(f"\n[note] models/v8_seed0_{tag}.pt missing — falling back to V7 tips.")
                model, tag = "v7", ""
        tip_extra = ["--date", args.date, "--days", str(args.days),
                     "--tournaments", args.tournaments, "--model", model]
        if model == "v8":
            tip_extra += ["--tag", tag]
        run_step(f"Generate tips ({model}{'/' + tag if tag else ''})",
                 "scripts.make_tips", tip_extra, fatal=False)

    print(f"\n{'=' * 78}\n Refresh complete in {time.time() - overall_t0:.1f}s.\n{'=' * 78}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
