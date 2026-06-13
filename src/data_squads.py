"""
Tier 2A — Refresh national-team squad strength from current Transfermarkt values.

Why this exists
---------------
Der V6/V8-Feature-Vektor enthält Kader-Features (``sq_ovr/sq_att/sq_def``). Beim
*Training* kommen sie aus den historischen FIFA-Daten (``nation_strength.parquet``,
as-of-year, 2015–2022). Bei der *Inferenz* für die WM 2026 liest
``get_current_team_ratings_v6`` sie dagegen aus ``squads_2026.json`` — und die ist
ohne Overrides bei FIFA-22 eingefroren (≈3,5 Jahre alt). Ein Team mit junger,
gewachsener Generation ist dort unterbewertet, ein gealtertes überbewertet.

soccerdata 1.9 hat *keinen* Transfermarkt-Reader, aber TM ist mit ``requests``
direkt erreichbar (kein Cloudflare, kein Browser). Dieses Modul:

  1. scrapt die aktuellen Gesamt-Marktwerte der Nationalmannschaften,
  2. kalibriert sie via OLS ``sq_ovr ≈ a + b·log10(MV)`` auf die FIFA-OVR-Skala,
     auf der das Modell trainiert wurde (kein Extrapolieren aus der Verteilung),
  3. zieht den eingefrorenen FIFA-22-Wert konservativ Richtung Markt
     (``new = base + w·(implied − base)``, ``w`` = ``--blend``, Default 0.5),
  4. schreibt die aufgefrischten ``sq_ovr/att/def`` nach
     ``data/raw/squads_2026_overrides.json`` und baut ``squads_2026.json`` neu.

**Inference-only, kein Retrain, kein Leakage:** das Training nutzt weiterhin die
as-of-year-FIFA-Werte; nur die Live-2026-Vorhersage sieht die frischeren Kader.
Der Backtest kann diese Änderung daher *nicht* bewerten — Validierung läuft über
Sanity-Checks (Top-Teams hoch, Minnows niedrig) und Tipp-Plausibilität.

Usage:
    python -m src.data_squads                 # fetch → calibrate → write → rebuild
    python -m src.data_squads --blend 0.6     # stärker Richtung Markt
    python -m src.data_squads --dry-run       # nur Report, nichts schreiben
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import requests

from .fifa_squad import _normalize_nat, build_squads_2026
from .team_normalize import normalize_team_name

REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = REPO_ROOT / "data" / "raw"
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
OVERRIDES_PATH = RAW_DIR / "squads_2026_overrides.json"
TM_VALUES_PATH = RAW_DIR / "tm_national_values.json"

_TM_URL = ("https://www.transfermarkt.com/vereins-statistik/"
           "wertvollstenationalmannschaften/marktwertetop")
_HDR = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"),
    "Accept-Language": "en-US,en;q=0.9",
}

# OVR-Klammern: Nationalmannschafts-Top-23-Schnitt liegt realistisch in [62, 90].
_OVR_MIN, _OVR_MAX = 62.0, 90.0

# TM-Schreibweisen, die _normalize_nat/normalize_team_name nicht trifft.
_TM_ALIASES = {
    "bosnia-herzegovina": "Bosnia and Herzegovina",
    "democratic republic of the congo": "DR Congo",
}


def _canon_tm(raw_name: str) -> str:
    return _TM_ALIASES.get(raw_name.strip().lower()) or _normalize_nat(raw_name)


def _parse_market_value(text: str) -> float | None:
    """'€1.52bn' / '€947.00m' / '€800k' → Millionen Euro (float)."""
    s = text.strip().replace("\xa0", " ")
    m = re.search(r"€\s*([\d.,]+)\s*(bn|m|k)?", s)
    if not m:
        return None
    num = float(m.group(1).replace(",", ""))
    unit = (m.group(2) or "m").lower()
    return {"bn": num * 1000.0, "m": num, "k": num / 1000.0}[unit]


def fetch_tm_national_values(max_pages: int = 10, pause: float = 1.5,
                             cache: bool = True, from_cache: bool = False) -> dict[str, float]:
    """Scrapt die TM-Nationalmannschafts-Marktwerte → {canonical_name: MV_in_mio_€}.

    from_cache=True liest den letzten Scrape aus tm_national_values.json (TM-schonend
    für Wiederholungen/Tests), statt erneut zu laden.
    """
    if from_cache and TM_VALUES_PATH.exists():
        cached = json.loads(TM_VALUES_PATH.read_text(encoding="utf-8"))
        vals = {_canon_tm(k): v for k, v in cached.get("values_mio_eur", {}).items()}
        print(f"   aus Cache: {len(vals)} Teams ({TM_VALUES_PATH.name})")
        return vals

    from bs4 import BeautifulSoup

    values: dict[str, float] = {}
    for page in range(1, max_pages + 1):
        url = _TM_URL if page == 1 else f"{_TM_URL}?page={page}"
        r = requests.get(url, headers=_HDR, timeout=25)
        r.encoding = "utf-8"
        if r.status_code != 200:
            print(f"   [warn] page {page}: HTTP {r.status_code} — Abbruch.")
            break
        soup = BeautifulSoup(r.text, "lxml")
        table = soup.select_one("table.items")
        rows = table.select("tbody > tr") if table else []
        if not rows:
            break
        page_n = 0
        for tr in rows:
            a = tr.select_one("td.hauptlink a[title]") or tr.select_one("td a[title]")
            if not a:
                continue
            raw_name = a.get("title", "").strip()
            tds = tr.select("td")
            val = None
            for td in reversed(tds):                       # Gesamtwert steht rechts
                if "€" in td.get_text():
                    val = _parse_market_value(td.get_text())
                    break
            if not raw_name or val is None:
                continue
            canon = _canon_tm(raw_name)
            values.setdefault(canon, val)                  # erster (höchster) Treffer
            page_n += 1
        print(f"   page {page}: +{page_n} Teams (kumuliert {len(values)})")
        time.sleep(pause)

    if cache and values:
        TM_VALUES_PATH.write_text(
            json.dumps({"_source": _TM_URL, "values_mio_eur": values},
                       indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"   gecached: {TM_VALUES_PATH}")
    return values


def _latest_fifa_strength() -> dict[str, dict]:
    """Neuester FIFA-Jahrgang je Nation aus nation_strength.parquet."""
    import pandas as pd
    p = PROCESSED_DIR / "nation_strength.parquet"
    if not p.exists():
        from .fifa_squad import build_nation_strength
        build_nation_strength()
    ns = pd.read_parquet(p)
    latest = ns.sort_values("year").groupby("nation").last()
    return {
        n: {"sq_ovr": float(r["sq_ovr"]), "sq_att": float(r["sq_att"]),
            "sq_def": float(r["sq_def"])}
        for n, r in latest.iterrows()
    }


def _wc2026_teams() -> list[str]:
    import pandas as pd
    df = pd.read_csv(RAW_DIR / "results.csv", parse_dates=["date"])
    wc = df[(df["tournament"] == "FIFA World Cup") & (df["date"] >= "2026-01-01")]
    return sorted(set(wc["home_team"]) | set(wc["away_team"]))


def calibrate_overrides(market: dict[str, float], blend: float = 0.5,
                        write: bool = True) -> dict:
    """Kalibriert MV→OVR (rang-/quantil-erhaltend), frischt die WM-Teams auf.

    *Quantil-Mapping statt Regression:* eine OLS ``ovr~log(MV)`` schrumpft die
    Ränder (Top-Teams werden Richtung Mittelwert gezogen → Favoriten-Abstände
    kollabieren, schlecht für eine Fixpunkt-Liga). Stattdessen wird der MV-Rang
    eines Teams auf das gleiche Quantil der FIFA-OVR-Verteilung abgebildet: die
    Streuung bleibt erhalten, nur die *Reihenfolge* wird auf den aktuellen Markt
    aktualisiert. Bewegung gibt es nur, wo MV-Rang und FIFA-OVR-Rang auseinander
    laufen (echte Auf-/Absteiger seit FIFA-22).
    """
    fifa = _latest_fifa_strength()

    # Schnittmenge MV ∩ FIFA definiert beide Verteilungen.
    overlap = [n for n in market if n in fifa]
    if len(overlap) < 20:
        raise RuntimeError(f"Zu wenig Overlap für Kalibrierung ({len(overlap)}).")
    mv_sorted = np.sort(np.array([market[n] for n in overlap], dtype=float))
    ovr_pool = np.array([fifa[n]["sq_ovr"] for n in overlap], dtype=float)
    N = len(overlap)
    rho = float(np.corrcoef(np.log10([market[n] for n in overlap]),
                            [fifa[n]["sq_ovr"] for n in overlap])[0, 1])

    def _implied_ovr(m: float) -> float:
        # Anteil der Teams mit MV ≤ m → gleiches Quantil der OVR-Verteilung.
        rank = np.searchsorted(mv_sorted, m, side="right")
        p = min(max((rank - 0.5) / N, 0.0), 1.0)
        return float(np.quantile(ovr_pool, p))

    print(f"\n   Quantil-Mapping MV→OVR  (Overlap n={N}, "
          f"r[log MV, OVR]={rho:.3f}, blend={blend})")

    wc_teams = _wc2026_teams()
    refreshed: dict[str, dict] = {}
    report, uncovered = [], []
    for team in wc_teams:
        base = fifa.get(team)
        m = market.get(team)
        if base is None or m is None:
            uncovered.append(team)
            continue
        implied = _implied_ovr(m)
        new_ovr = float(np.clip(base["sq_ovr"] + blend * (implied - base["sq_ovr"]),
                                _OVR_MIN, _OVR_MAX))
        delta = new_ovr - base["sq_ovr"]
        refreshed[team] = {
            "sq_ovr": round(new_ovr, 2),
            "sq_att": round(float(np.clip(base["sq_att"] + delta, _OVR_MIN, _OVR_MAX)), 2),
            "sq_def": round(float(np.clip(base["sq_def"] + delta, _OVR_MIN, _OVR_MAX)), 2),
            "_mv_mio_eur": round(m, 1),
            "_sq_ovr_fifa22": round(base["sq_ovr"], 2),
        }
        report.append((delta, team, m, base["sq_ovr"], new_ovr))

    # Report: größte Bewegungen zuerst + Streuungs-Erhalt (Anti-Shrinkage-Check)
    base_spread = float(np.std([r[3] for r in report]))
    new_spread = float(np.std([r[4] for r in report]))
    print(f"\n   {len(refreshed)}/{len(wc_teams)} WM-Teams aufgefrischt. "
          f"OVR-Streuung WM-Teams: {base_spread:.2f} (FIFA22) → {new_spread:.2f} (neu) "
          f"— Spread bleibt erhalten (kein Shrinkage).")
    print("   Größte Verschiebungen (echte Auf-/Absteiger seit FIFA-22):")
    print("     Δovr   team                 MV(m€)   FIFA22 → neu")
    for delta, team, m, base_o, new_o in sorted(report, key=lambda x: -abs(x[0]))[:16]:
        print(f"     {delta:+5.2f}  {team:<20s} {m:8.0f}   {base_o:5.1f} → {new_o:5.1f}")
    if uncovered:
        print(f"   ohne TM-Wert (FIFA-22 bleibt): {', '.join(uncovered)}")

    if write:
        existing = {}
        if OVERRIDES_PATH.exists():
            existing = json.loads(OVERRIDES_PATH.read_text(encoding="utf-8"))
        # bestehende manuelle Keys (coach, Verletzungen) erhalten, sq_* überschreiben
        for team, vals in refreshed.items():
            existing.setdefault(team, {})
            existing[team].update(vals)
        existing["_tier2a_comment"] = (
            "sq_ovr/att/def 2026-06 aus Transfermarkt-Marktwerten kalibriert "
            f"(rang-/quantil-erhaltend auf FIFA-Skala, blend={blend}). "
            "Inference-only, kein Retrain. src/data_squads.py")
        OVERRIDES_PATH.write_text(
            json.dumps(existing, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\n   geschrieben: {OVERRIDES_PATH}")
    return refreshed


def main() -> int:
    ap = argparse.ArgumentParser(description="Tier 2A: TM-Marktwert-Refresh der WM-Kader")
    ap.add_argument("--blend", type=float, default=0.5,
                    help="Gewicht Richtung Markt (0=FIFA-22, 1=voll Markt-implied)")
    ap.add_argument("--max-pages", type=int, default=10)
    ap.add_argument("--from-cache", action="store_true",
                    help="TM-Werte aus tm_national_values.json statt neu scrapen")
    ap.add_argument("--dry-run", action="store_true", help="nur Report, nichts schreiben")
    args = ap.parse_args()

    try:                                  # Windows-Konsole (cp1252) → UTF-8
        import sys
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    print("=" * 70)
    print(" Tier 2A: Nationalmannschafts-Marktwerte → Kader-Refresh (WM 2026)")
    print("=" * 70)
    market = fetch_tm_national_values(max_pages=args.max_pages, from_cache=args.from_cache)
    if not market:
        print("   FEHLER: keine Marktwerte geladen."); return 1
    calibrate_overrides(market, blend=args.blend, write=not args.dry_run)
    if not args.dry_run:
        print()
        build_squads_2026()       # regeneriert squads_2026.json inkl. Overrides
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
