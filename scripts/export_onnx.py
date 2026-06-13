"""
Export E8Net to ONNX for browser inference via ONNX Runtime Web.

Steps:
  1. Load best V8 checkpoint (lowest val_loss across seeds)
  2. Export PyTorch model → ONNX fp32 (opset 17, dynamic batch)
  3. Verify parity: |PyTorch − ORT| < 1e-3 over 64 random samples
  4. Dynamic int8 quantization → v8_e8net_q8.onnx
  5. Save preprocessing params JSON for browser normalization

Outputs:
  models/v8_e8net.onnx          — fp32 ONNX (full precision)
  models/v8_e8net_q8.onnx       — int8 quantized (smaller, faster)
  data/models/v8_preproc.json   — norm params + dc_rho for browser

Usage:
  cd E:/Projects/Fussball_ai
  .\\venv\\Scripts\\python.exe -m scripts.export_onnx
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# Force UTF-8 output so torch's internal print() calls (which print Unicode) don't crash on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import torch
import onnx
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.model_v8 import E8Net, E8Config
from src.features_v8 import load_v8, SEQ_LEN, N_PLAYERS, SEQ_DIM, PLAYER_DIM

REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = REPO_ROOT / "models"
DATA_MODELS = REPO_ROOT / "data" / "models"

ONNX_FP32  = MODELS_DIR / "v8_e8net.onnx"
ONNX_INT8  = MODELS_DIR / "v8_e8net_q8.onnx"
PREPROC    = DATA_MODELS / "v8_preproc.json"
META_PATH  = DATA_MODELS / "v8_latest.json"

OPSET      = 17
STATIC_DIM = 57
CTX_DIM    = 2


def _load_best_checkpoint(meta_path: Path = META_PATH):
    with open(meta_path) as f:
        meta = json.load(f)
    best = min(meta["checkpoints"], key=lambda c: c["val_loss"])
    ckpt = torch.load(best["path"], map_location="cpu", weights_only=True)
    cfg = E8Config(**{k: v for k, v in ckpt["cfg"].items() if k in E8Config.__dataclass_fields__})
    model = E8Net(cfg)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    print(f"   Loaded seed checkpoint: {best['path']}")
    print(f"   val_loss={best['val_loss']:.4f}  val_acc={best.get('val_acc', '?'):.4f}")
    return model, meta, ckpt


def _dummy_inputs(batch: int = 4):
    rng = np.random.default_rng(0)
    static  = torch.from_numpy(rng.standard_normal((batch, STATIC_DIM)).astype(np.float32))
    seq_h   = torch.from_numpy(rng.standard_normal((batch, SEQ_LEN, SEQ_DIM)).astype(np.float32))
    squad_h = torch.from_numpy(rng.standard_normal((batch, N_PLAYERS, PLAYER_DIM)).astype(np.float32))
    seq_a   = torch.from_numpy(rng.standard_normal((batch, SEQ_LEN, SEQ_DIM)).astype(np.float32))
    squad_a = torch.from_numpy(rng.standard_normal((batch, N_PLAYERS, PLAYER_DIM)).astype(np.float32))
    ctx     = torch.from_numpy(rng.standard_normal((batch, CTX_DIM)).astype(np.float32))
    return static, seq_h, squad_h, seq_a, squad_a, ctx


def export_fp32(model: E8Net, out_path: Path):
    print("\n   Exporting fp32 ONNX (dynamo exporter) ...")
    dummy = _dummy_inputs(4)

    # Wrap E8Net.forward so it returns a flat tuple (ONNX can't export dataclasses)
    class _Wrapper(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, static_X, seq_home, squad_home, seq_away, squad_away, context):
            out = self.inner(static_X, seq_home, squad_home, seq_away, squad_away, context)
            return out.log_lam_home, out.log_lam_away, out.p_et, out.p_pen_given_et, out.p_home_pen

    wrapper = _Wrapper(model)
    wrapper.eval()

    from torch.export import Dim
    batch = Dim("batch", min=1, max=2048)
    dynamic_shapes = (
        {0: batch},   # static_X
        {0: batch},   # seq_home
        {0: batch},   # squad_home
        {0: batch},   # seq_away
        {0: batch},   # squad_away
        {0: batch},   # context
    )

    onnx_program = torch.onnx.export(
        wrapper,
        dummy,
        dynamic_shapes=dynamic_shapes,
        input_names=["static_X", "seq_home", "squad_home", "seq_away", "squad_away", "context"],
        output_names=["log_lam_home", "log_lam_away", "p_et", "p_pen_given_et", "p_home_pen"],
        opset_version=OPSET,
    )
    onnx_program.save(str(out_path))

    # Basic structural check
    model_proto = onnx.load(str(out_path))
    onnx.checker.check_model(model_proto)
    size_mb = out_path.stat().st_size / 1e6
    print(f"   Saved: {out_path}  ({size_mb:.1f} MB)")
    return wrapper


def verify_parity(model: torch.nn.Module, out_path: Path, n_samples: int = 64):
    print("\n   Verifying parity (PyTorch vs ORT) ...")
    dummy = _dummy_inputs(n_samples)
    inputs_np = [t.numpy() for t in dummy]

    with torch.no_grad():
        pt_out = model(*dummy)  # tuple of 5 tensors

    sess = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
    ort_out = sess.run(None, {
        "static_X":   inputs_np[0],
        "seq_home":   inputs_np[1],
        "squad_home": inputs_np[2],
        "seq_away":   inputs_np[3],
        "squad_away": inputs_np[4],
        "context":    inputs_np[5],
    })

    names = ["log_lam_home", "log_lam_away", "p_et", "p_pen_given_et", "p_home_pen"]
    max_err = 0.0
    for i, name in enumerate(names):
        err = float(np.abs(pt_out[i].numpy() - ort_out[i]).max())
        max_err = max(max_err, err)
        status = "OK" if err < 1e-3 else "FAIL"
        print(f"   {name:<20} max_err={err:.2e}  [{status}]")

    if max_err >= 1e-3:
        raise RuntimeError(f"Parity check FAILED: max_err={max_err:.2e} >= 1e-3")
    print(f"   Parity OK (max_err={max_err:.2e})")


def quantize_int8(fp32_path: Path, int8_path: Path) -> bool:
    print("\n   Quantizing to int8 (dynamic) ...")
    try:
        quantize_dynamic(
            str(fp32_path),
            str(int8_path),
            weight_type=QuantType.QUInt8,
        )
        size_mb = int8_path.stat().st_size / 1e6
        print(f"   Saved: {int8_path}  ({size_mb:.1f} MB)")
        return True
    except Exception as e:
        print(f"   WARNING: int8 quantization failed ({e}); using fp32 only.")
        return False


def save_preproc(meta: dict, ckpt: dict, meta_path: Path = META_PATH):
    with open(meta_path) as f:
        meta = json.load(f)
    best = min(meta["checkpoints"], key=lambda c: c["val_loss"])
    ckpt = torch.load(best["path"], map_location="cpu", weights_only=True)

    preproc = {
        "norm_mean":    ckpt["norm_mean"],
        "norm_std":     ckpt["norm_std"],
        "col_medians":  ckpt["col_medians"],
        "dc_rho":       ckpt["dc_rho"],
        "static_dim":   STATIC_DIM,
        "seq_len":      SEQ_LEN,
        "seq_dim":      SEQ_DIM,
        "n_players":    N_PLAYERS,
        "player_dim":   PLAYER_DIM,
        "ctx_dim":      CTX_DIM,
    }
    if "lam_cal" in meta:
        preproc["lam_cal"] = meta["lam_cal"]
    DATA_MODELS.mkdir(parents=True, exist_ok=True)
    with open(PREPROC, "w") as f:
        json.dump(preproc, f, indent=2)
    print(f"\n   Preprocessing params: {PREPROC}")
    if "lam_cal" in meta:
        c = meta["lam_cal"]
        print(f"   lam_cal: b_home={c['b_home']:.4f} b_away={c['b_away']:.4f}")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", type=str, default="",
                    help="Meta tag, e.g. 'v9' → reads v8_latest_v9.json")
    args = ap.parse_args()
    meta_path = DATA_MODELS / (f"v8_latest_{args.tag}.json" if args.tag else "v8_latest.json")

    print("=" * 70)
    print(f" E8Net ONNX Export  (meta={meta_path.name})")
    print("=" * 70)

    model, meta, ckpt = _load_best_checkpoint(meta_path)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    wrapper = export_fp32(model, ONNX_FP32)
    verify_parity(wrapper, ONNX_FP32)
    q8_ok = quantize_int8(ONNX_FP32, ONNX_INT8)

    if q8_ok:
        print("\n   Verifying int8 parity ...")
        dummy = _dummy_inputs(32)
        inputs_np = [t.numpy() for t in dummy]
        with torch.no_grad():
            pt_out = wrapper(*dummy)
        sess_q = ort.InferenceSession(str(ONNX_INT8), providers=["CPUExecutionProvider"])
        ort_q  = sess_q.run(None, {
            "static_X":   inputs_np[0],
            "seq_home":   inputs_np[1],
            "squad_home": inputs_np[2],
            "seq_away":   inputs_np[3],
            "squad_away": inputs_np[4],
            "context":    inputs_np[5],
        })
        errs = [float(np.abs(pt_out[i].numpy() - ort_q[i]).max()) for i in range(5)]
        print(f"   int8 max_err over 5 outputs: {max(errs):.4f}  (threshold 0.05)")

    save_preproc(meta, ckpt, meta_path)

    print("\n" + "=" * 70)
    print(" Export complete.")
    print("=" * 70)
    print(f"   fp32:  {ONNX_FP32}  ({ONNX_FP32.stat().st_size/1e6:.1f} MB)")
    if q8_ok and ONNX_INT8.exists():
        print(f"   int8:  {ONNX_INT8}  ({ONNX_INT8.stat().st_size/1e6:.1f} MB)")
    browser_model = "v8_e8net_q8.onnx" if q8_ok else "v8_e8net.onnx"
    print(f"   Next step: integrate {browser_model} in wm-predictor.js")


if __name__ == "__main__":
    main()
