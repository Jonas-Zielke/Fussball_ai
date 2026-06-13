"""
V8 Training — E8Net (Transformer + Poisson + KO heads). CLI-Wrapper.

Die eigentliche Trainingslogik liegt in src/train_v8_lib.py (parametrisierbar
für Sweeps/Backtests). Ohne Argumente reproduziert dieser Wrapper das
bisherige Verhalten (HL=90d, Train 2015–2022, Cal 2023, Val 2024+, 3 Seeds).

Usage:
  .\\venv\\Scripts\\python.exe -m scripts.train_v8
  .\\venv\\Scripts\\python.exe -m scripts.train_v8 --half-life 1095 --train-start 2010-01-01
  .\\venv\\Scripts\\python.exe -m scripts.train_v8 --half-life inf --seeds 1 --tag sweep_hlinf

Output:
  models/v8_seed{N}[_tag].pt           — PyTorch-Checkpoints
  data/models/v8_latest[_tag].json     — Ensemble-Metadaten (für export_onnx)
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.features_v8 import load_v8
from src.train_v8_lib import DEVICE, TrainConfig, train_v8


def _parse_half_life(s: str) -> float | None:
    if s.strip().lower() in ("none", "inf", "infinity", "0"):
        return None
    return float(s)


def main() -> int:
    ap = argparse.ArgumentParser(description="Train E8Net (V8)")
    ap.add_argument("--half-life", type=_parse_half_life, default=90.0,
                    help="Recency-Halbwertszeit in Tagen ('inf' = aus; default 90)")
    ap.add_argument("--train-start", type=str, default="2015-01-01")
    ap.add_argument("--cal-start", type=str, default="2023-01-01")
    ap.add_argument("--val-start", type=str, default="2024-01-01")
    ap.add_argument("--val-end", type=str, default=None)
    ap.add_argument("--anchor", type=str, default="max-train",
                    help="'max-train' | 'val-start' | ISO-Datum")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--tag", type=str, default="",
                    help="Suffix für Checkpoints/Meta (Schutz der deployten Artefakte)")
    ap.add_argument("--no-save", action="store_true", help="keine Checkpoints schreiben")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    cfg = TrainConfig(
        half_life_days=args.half_life,
        train_start=args.train_start,
        cal_start=args.cal_start,
        val_start=args.val_start,
        val_end=args.val_end,
        anchor=args.anchor,
        n_seeds=args.seeds,
        epochs=args.epochs,
        patience=args.patience,
        lr=args.lr,
        batch_size=args.batch_size,
        save_dir=None if args.no_save else "models",
        tag=args.tag,
        quiet=args.quiet,
    )

    print("=" * 70)
    print(" V8 Training: E8Net (Transformer + Poisson + Cross-Attention)")
    print("=" * 70)
    print(f"   Device: {DEVICE}")
    hl = "inf" if cfg.half_life_days is None else f"{cfg.half_life_days:.0f}d"
    print(f"   HL={hl}  Train=[{cfg.train_start}, {cfg.cal_start})  "
          f"Cal=[{cfg.cal_start}, {cfg.val_start})  Val>={cfg.val_start}"
          + (f" <{cfg.val_end}" if cfg.val_end else ""))

    print("\n   Loading features_v8 ...")
    data = load_v8()
    res = train_v8(data, cfg)

    print(f"\n{'=' * 70}")
    print(" V8 Training complete.")
    print(f"{'=' * 70}")
    for s in res["seeds"]:
        print(f"   Seed {s['seed']}: cal_loss={s['cal_loss']:.4f}  val_acc={s['val_acc']:.4f}")
    print(f"   Ensemble val_acc: {res['ensemble_val_acc']:.4f}   ESS={res['ess']:,.0f}")

    if not args.no_save:
        # Meta-Format kompatibel zu scripts/export_onnx.py ("val_loss" = Cal-NLL)
        meta = {
            "checkpoints": [
                {"path": s["path"], "val_loss": s["cal_loss"], "val_acc": s["val_acc"]}
                for s in res["seeds"]
            ],
            "norm_mean": res["norm_mean"].tolist(),
            "norm_std": res["norm_std"].tolist(),
            "col_medians": res["col_medians"].tolist(),
            "dc_rho": res["dc_rho"],
            "cfg": res["cfg"]["model_cfg"],
            "train_cfg": res["cfg"],
            "ess": res["ess"],
            "model_version": "v8-e8net",
        }
        suffix = f"_{args.tag}" if args.tag else ""
        meta_path = Path(__file__).resolve().parent.parent / "data" / "models" / f"v8_latest{suffix}.json"
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
        print(f"   Meta: {meta_path}")
        print("   Next step: python -m scripts.export_onnx")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
