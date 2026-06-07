"""Int8 quantize V4 model for browser deployment.
Strategy: scale weights to int8, store scale factors per tensor.
"""
import json
import numpy as np
from pathlib import Path


def quantize_in_place(src_path: Path, dst_path: Path) -> float:
    bundle = json.loads(src_path.read_text())
    for m in bundle["models"]:
        for k, v in list(m.items()):
            if not isinstance(v, list) or not v or not isinstance(v[0], list):
                continue
            arr = np.array(v, dtype=np.float32)
            abs_max = float(np.abs(arr).max())
            if abs_max == 0:
                scale = 1.0
            else:
                scale = abs_max / 127.0
            q = np.clip(np.round(arr / scale), -127, 127).astype(np.int8)
            m[k] = {
                "data": q.flatten().tolist(),
                "scale": scale,
                "shape": list(arr.shape),
                "dtype": "int8",
            }
    bundle["_quantization"] = "int8"
    with open(dst_path, "w", encoding="utf-8") as fh:
        json.dump(bundle, fh, separators=(",", ":"))
    return dst_path.stat().st_size


def main():
    src = Path("E:/Profilov2/public/data/wm-predictor/model.json")
    dst = Path("E:/Profilov2/public/data/wm-predictor/model_int8.json")
    out_size = quantize_in_place(src, dst)
    print(f"Int8 size: {out_size/1e6:.2f} MB (vs {src.stat().st_size/1e6:.2f} MB float32)")
    print(f"geschrieben: {dst}")


if __name__ == "__main__":
    main()
