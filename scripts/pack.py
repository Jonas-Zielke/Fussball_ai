"""Packt das Projekt in ein zip ohne venv und raw data."""
import os
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "Fussball_ai_source.zip"
SKIP = {"venv", "__pycache__", "data", "models"}


def should_skip(path: Path) -> bool:
    parts = set(path.parts)
    return bool(parts & SKIP)


count = 0
with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        if should_skip(path):
            continue
        rel = path.relative_to(ROOT)
        zf.write(path, rel.as_posix())
        count += 1
print(f"OK: {OUT} ({OUT.stat().st_size/1024:.1f} KB, {count} files)")
