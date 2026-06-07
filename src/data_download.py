"""
Datendownload fuer den WM 2026 Predictor.

Quelle: Martj42 International Football Results (1872-2024+)
- URL: https://raw.githubusercontent.com/martj42/international-football-results-from-1872-to-2017/master/results.csv
- Spalten: date, home_team, away_team, home_score, away_score, tournament, city, country, neutral

Der Datensatz wird beim ersten Aufruf heruntergeladen und landet in data/raw/results.csv.
"""

from __future__ import annotations

import csv
import hashlib
import sys
from pathlib import Path

import requests

# Konfiguration
REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = REPO_ROOT / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

RESULTS_URL = (
    "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
)
RESULTS_PATH = RAW_DIR / "results.csv"

# Optional: Goal-Scorers (fuer spaetere Spieler-Features, nicht zwingend)
GOALSCORERS_URL = (
    "https://raw.githubusercontent.com/martj42/international_results/master/goalscorers.csv"
)
GOALSCORERS_PATH = RAW_DIR / "goalscorers.csv"

SHOOTOUTS_URL = (
    "https://raw.githubusercontent.com/martj42/international_results/master/shootouts.csv"
)
SHOOTOUTS_PATH = RAW_DIR / "shootouts.csv"


def _download_with_progress(url: str, dest: Path) -> None:
    """Laedt eine Datei herunter und schreibt sie atomar."""
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  [skip] {dest.name} existiert bereits ({dest.stat().st_size:,} bytes).")
        return
    print(f"  [get ] {url}")
    resp = requests.get(url, stream=True, timeout=60)
    resp.raise_for_status()
    total = int(resp.headers.get("Content-Length", 0))
    tmp = dest.with_suffix(dest.suffix + ".part")
    hash_sha = hashlib.sha256()
    written = 0
    with open(tmp, "wb") as fh:
        for chunk in resp.iter_content(chunk_size=64 * 1024):
            if not chunk:
                continue
            fh.write(chunk)
            hash_sha.update(chunk)
            written += len(chunk)
            if total:
                pct = written * 100 / total
                bar_w = 30
                filled = int(bar_w * written / total)
                bar = "#" * filled + "-" * (bar_w - filled)
                sys.stdout.write(f"\r    [{bar}] {pct:5.1f}%  ({written/1e6:5.2f} MB)")
                sys.stdout.flush()
    if total:
        sys.stdout.write("\n")
    tmp.replace(dest)
    print(f"  [ok  ] geschrieben: {dest} ({dest.stat().st_size:,} bytes, sha256={hash_sha.hexdigest()[:12]})")


def download_results() -> Path:
    """Laedt die Hauptdaten (results.csv) herunter."""
    print(">> Lade Hauptdatensatz (results.csv)...")
    _download_with_progress(RESULTS_URL, RESULTS_PATH)
    return RESULTS_PATH


def download_shootouts() -> Path:
    """Laedt die Elfmeterschiessen-Historie herunter (optional, fuer K.o.-Runden)."""
    print(">> Lade shootouts.csv (optional)...")
    _download_with_progress(SHOOTOUTS_URL, SHOOTOUTS_PATH)
    return SHOOTOUTS_PATH


def download_goalscorers() -> Path:
    """Laedt die Torschuetzen-Historie herunter (optional, gross)."""
    print(">> Lade goalscorers.csv (optional, ~5MB)...")
    _download_with_progress(GOALSCORERS_URL, GOALSCORERS_PATH)
    return GOALSCORERS_PATH


def verify_results_integrity(path: Path) -> None:
    """Prueft dass die CSV nicht kaputt ist und genug Zeilen hat."""
    if not path.exists():
        raise FileNotFoundError(f"{path} fehlt - bitte erst download_results() rufen.")
    with open(path, "r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        rows = sum(1 for _ in reader)
    if rows < 10_000:
        raise ValueError(f"results.csv hat nur {rows} Zeilen, das ist zu wenig - Datei evtl. kaputt?")
    print(f"  [verify] {rows:,} Zeilen OK.")


def main() -> int:
    print("=" * 70)
    print(" WM 2026 Predictor - Datendownload")
    print("=" * 70)
    download_results()
    verify_results_integrity(RESULTS_PATH)
    download_shootouts()
    print("=" * 70)
    print(" Fertig. Rohdaten liegen in:", RAW_DIR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
