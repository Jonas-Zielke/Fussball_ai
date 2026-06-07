"""Quantize V4 model to float16 for browser deployment."""
import json
from pathlib import Path
import numpy as np

# Lade das V4 Modell
src = Path("E:/Profilov2/public/data/wm-predictor/model.json")
dst = Path("E:/Profilov2/public/data/wm-predictor/model_f16.json")
bundle = json.loads(src.read_text())

# Konvertiere alle Modell-Weights zu float16-Listen
def to_f16(lst):
    if isinstance(lst, list):
        return [to_f16(x) for x in lst]
    return float(np.float16(lst))

for m in bundle["models"]:
    for k, v in list(m.items()):
        if isinstance(v, list) and v and isinstance(v[0], list):
            m[k] = to_f16(v)

bundle["_quantization"] = "float16"
out_size = len(json.dumps(bundle, separators=(",", ":")))
print(f"V4 f16 size: {out_size/1e6:.2f} MB (vs {src.stat().st_size/1e6:.2f} MB float32)")

with open(dst, "w", encoding="utf-8") as fh:
    json.dump(bundle, fh, separators=(",", ":"))
print(f"geschrieben: {dst}")
