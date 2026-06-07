"""
FIFA Squad Data Pipeline.

Lädt FIFA/EA-FC Spielerdaten (2015-2022) von öffentlichen GitHub-Mirrors,
aggregiert je (nation, year) die Kader-Stärke und baut:
  - data/processed/nation_strength.parquet  -- (nation, year, sq_ovr, sq_att, sq_def, sq_age, sq_depth)
  - data/processed/squads_2026.json         -- aktuelle WM-Kader (latest FIFA + overrides)

Quellen (kein Login nötig):
  2015-2020: github.com/apoorva-21/fifa-analysis  (sofifa-Format A)
  2021:      github.com/toheeb-olamilekan/fifa21_data_cleaning_challenge  (Format B)
  2022:      github.com/abineshta/FIFA-22-complete-player-dataset-EDA  (sofifa-Format C)
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from .team_normalize import normalize_team_name

REPO_ROOT = Path(__file__).resolve().parent.parent
FIFA_DIR = REPO_ROOT / "data" / "raw" / "fifa"
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
FIFA_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# ---------- Download sources ----------
_BASE_A = "https://raw.githubusercontent.com/apoorva-21/fifa-analysis/master/data"
_SOURCES: dict[int, tuple[str, str]] = {
    2015: (f"{_BASE_A}/players_15.csv", "A"),
    2016: (f"{_BASE_A}/players_16.csv", "A"),
    2017: (f"{_BASE_A}/players_17.csv", "A"),
    2018: (f"{_BASE_A}/players_18.csv", "A"),
    2019: (f"{_BASE_A}/players_19.csv", "A"),
    2020: (f"{_BASE_A}/players_20.csv", "A"),
    2021: (
        "https://raw.githubusercontent.com/toheeb-olamilekan/"
        "fifa21_data_cleaning_challenge/main/fifa21_raw_data_v2.csv",
        "B",
    ),
    2022: (
        "https://raw.githubusercontent.com/abineshta/"
        "FIFA-22-complete-player-dataset-EDA/main/players_22.csv",
        "C",
    ),
}

# Attacker and defender position sets (sofifa codes)
_ATT_POS = {"ST", "CF", "LW", "RW", "LF", "RF", "CAM", "LS", "RS", "SS"}
_DEF_POS = {"CB", "LCB", "RCB", "LB", "RB", "LWB", "RWB", "GK"}

# Extra FIFA nationality → Martj42 canonical (beyond what team_normalize.py already covers)
_FIFA_NAT_EXTRAS: dict[str, str] = {
    "korea republic": "South Korea",
    "republic of korea": "South Korea",
    "dpr korea": "North Korea",
    "china pr": "China PR",
    "dr congo": "DR Congo",
    "congo dr": "DR Congo",
    "cote d'ivoire": "Ivory Coast",
    "côte d'ivoire": "Ivory Coast",
    "republic of ireland": "Republic of Ireland",
    "ireland": "Republic of Ireland",
    "northern ireland": "Northern Ireland",
    "bosnia and herzegovina": "Bosnia and Herzegovina",
    "bosnia herzegovina": "Bosnia and Herzegovina",
    "bosnia & herzegovina": "Bosnia and Herzegovina",
    "cape verde islands": "Cape Verde",
    "cape verde": "Cape Verde",
    "slovak republic": "Slovakia",
    "trinidad & tobago": "Trinidad and Tobago",
    "trinidad and tobago": "Trinidad and Tobago",
    "china": "China PR",
    "türkiye": "Turkey",
    "turkiye": "Turkey",
    "turkey": "Turkey",
    "curacao": "Curaçao",
    "eswatini": "Eswatini",
    "north macedonia": "North Macedonia",
    "usa": "United States",
    "united states": "United States",
}


def _normalize_nat(name: str) -> str:
    key = str(name).strip().lower()
    if key in _FIFA_NAT_EXTRAS:
        return _FIFA_NAT_EXTRAS[key]
    return normalize_team_name(name)


def _download_year(year: int) -> Path:
    url, fmt = _SOURCES[year]
    ext = ".csv"
    fname = f"players_{str(year)[2:]}{ext}"
    path = FIFA_DIR / fname
    if path.exists():
        return path
    print(f"   Lade {year} von {url.split('/')[-3]} ...")
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    path.write_bytes(r.content)
    print(f"   gespeichert: {path} ({path.stat().st_size // 1024} KB)")
    return path


def _player_positions(pos_str: str) -> set[str]:
    if not isinstance(pos_str, str):
        return set()
    return {p.strip().upper() for p in pos_str.replace(",", " ").split()}


def _load_year(year: int) -> pd.DataFrame:
    path = _download_year(year)
    _, fmt = _SOURCES[year]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if fmt in ("A", "C"):
            df = pd.read_csv(path, low_memory=False, encoding="utf-8-sig")
            # Format A: 'nationality'   Format C: 'nationality_name'
            nat_col = "nationality_name" if "nationality_name" in df.columns else "nationality"
            df = df.rename(columns={nat_col: "nation"})
            # both formats use same stat column names
            keep = ["nation", "overall", "pace", "shooting", "passing",
                    "dribbling", "defending", "physic", "player_positions", "age"]
            # player_positions might be missing in very old format
            if "player_positions" not in df.columns and "team_position" in df.columns:
                df["player_positions"] = df["team_position"]
            elif "player_positions" not in df.columns:
                df["player_positions"] = ""
        else:  # Format B - FIFA21 messy format
            df = pd.read_csv(path, low_memory=False, encoding="utf-8-sig")
            # Rename to canonical names; overall col is '↓OVA'
            ovr_col = next((c for c in df.columns if "OVA" in c.upper()), None)
            pos_col = next((c for c in df.columns if c in ("Positions", "Position")), None)
            df = df.rename(columns={
                "Nationality": "nation",
                ovr_col: "overall",
                "Age": "age",
                pos_col: "player_positions",
                "PAC": "pace",
                "SHO": "shooting",
                "PAS": "passing",
                "DRI": "dribbling",
                "DEF": "defending",
                "PHY": "physic",
            })
            if "player_positions" not in df.columns:
                df["player_positions"] = ""
            keep = ["nation", "overall", "pace", "shooting", "passing",
                    "dribbling", "defending", "physic", "player_positions", "age"]

    # Keep only needed columns (some may not exist yet)
    keep = [c for c in keep if c in df.columns]
    df = df[keep].copy()

    # Coerce numerics
    for col in ["overall", "pace", "shooting", "passing", "dribbling", "defending", "physic", "age"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["nation", "overall"])
    df["nation"] = df["nation"].apply(_normalize_nat)
    df["year"] = year
    return df


def _aggregate_nation_year(group: pd.DataFrame) -> dict:
    """Aggregiert Kader-Stats für eine (nation, year)-Gruppe."""
    g = group.sort_values("overall", ascending=False)
    top23 = g.head(23)

    ovr_vals = top23["overall"].dropna()
    sq_ovr = float(ovr_vals.mean()) if len(ovr_vals) > 0 else 74.0
    if np.isnan(sq_ovr):
        sq_ovr = 74.0

    if "age" in top23.columns:
        age_vals = top23["age"].dropna()
        sq_age = float(age_vals.mean()) if len(age_vals) > 0 else 27.0
        if np.isnan(sq_age):
            sq_age = 27.0
    else:
        sq_age = 27.0

    sq_depth = float(ovr_vals.std()) if len(ovr_vals) >= 2 else 0.0
    if np.isnan(sq_depth):
        sq_depth = 0.0

    # Attacker/defender splits from positions
    def _pos_mean(pos_set: set[str], fallback_col: str) -> float:
        pos_mask = g["player_positions"].apply(
            lambda p: bool(_player_positions(p) & pos_set)
        )
        sub = g[pos_mask].head(6)
        if len(sub) >= 2:
            val = float(sub["overall"].dropna().mean())
            if not np.isnan(val):
                return val
        # fallback: use attribute column if available
        if fallback_col in g.columns:
            top6 = g.head(6)
            vals = top6[fallback_col].dropna()
            if len(vals) > 0:
                val = float(vals.mean())
                if not np.isnan(val):
                    return val
        return sq_ovr

    sq_att = _pos_mean(_ATT_POS, "shooting")
    sq_def = _pos_mean(_DEF_POS, "defending")

    return {
        "sq_ovr": sq_ovr,
        "sq_att": sq_att,
        "sq_def": sq_def,
        "sq_age": sq_age,
        "sq_depth": sq_depth,
    }


def build_nation_strength() -> Path:
    """Baut nation_strength.parquet aus FIFA-Spielerdaten."""
    print("=" * 70)
    print(" FIFA Squad Pipeline: nation_strength")
    print("=" * 70)
    frames = []
    for year in sorted(_SOURCES.keys()):
        try:
            df = _load_year(year)
            frames.append(df)
            print(f"   {year}: {len(df):,} Spieler, {df['nation'].nunique()} Nationen")
        except Exception as e:
            print(f"   WARN {year}: {e}")

    all_df = pd.concat(frames, ignore_index=True)

    rows = []
    for (nation, year), group in all_df.groupby(["nation", "year"]):
        stats = _aggregate_nation_year(group)
        rows.append({"nation": nation, "year": int(year), **stats})

    ns_df = pd.DataFrame(rows)
    out = PROCESSED_DIR / "nation_strength.parquet"
    ns_df.to_parquet(out, index=False)
    print(f"   Gespeichert: {out}  ({len(ns_df):,} Einträge, {ns_df['nation'].nunique()} Nationen)")
    print("=" * 70)
    return out


def build_squads_2026(overrides_path: Path | None = None) -> Path:
    """
    Baut squads_2026.json: aktuelle Kader-Stats für WM-2026-Teams.

    Basis: neueste verfügbare FIFA-Daten (2022), angepasst durch
    optionale Overrides aus squads_2026_overrides.json.
    """
    print(" FIFA Squad Pipeline: squads_2026")
    ns_path = PROCESSED_DIR / "nation_strength.parquet"
    if not ns_path.exists():
        build_nation_strength()
    ns_df = pd.read_parquet(ns_path)

    # Neuestes Jahr je Nation
    latest = ns_df.sort_values("year").groupby("nation").last().reset_index()

    squads: dict[str, dict] = {}
    for _, row in latest.iterrows():
        squads[row["nation"]] = {
            "sq_ovr": float(row["sq_ovr"]),
            "sq_att": float(row["sq_att"]),
            "sq_def": float(row["sq_def"]),
            "sq_age": float(row["sq_age"]),
            "sq_depth": float(row["sq_depth"]),
            "coach": "",
        }

    # Coaches einlesen (falls vorhanden)
    coaches_path = REPO_ROOT / "data" / "raw" / "coaches_2026.json"
    if coaches_path.exists():
        with open(coaches_path, encoding="utf-8") as fh:
            coaches = json.load(fh)
        for entry in coaches:
            team = _normalize_nat(entry.get("team", ""))
            if team in squads:
                squads[team]["coach"] = entry.get("coach", "")

    # Overrides einlesen (Verletzungen, Nachnominierungen, aktuelle Kader)
    if overrides_path is None:
        overrides_path = REPO_ROOT / "data" / "raw" / "squads_2026_overrides.json"
    if overrides_path.exists():
        with open(overrides_path, encoding="utf-8") as fh:
            overrides = json.load(fh)
        for team_raw, vals in overrides.items():
            if team_raw.startswith("_"):
                continue
            team = _normalize_nat(team_raw)
            if team not in squads:
                squads[team] = {
                    "sq_ovr": 75.0, "sq_att": 75.0, "sq_def": 74.0,
                    "sq_age": 27.0, "sq_depth": 4.0, "coach": "",
                }
            squads[team].update(vals)

    out = PROCESSED_DIR / "squads_2026.json"
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(squads, fh, indent=2, ensure_ascii=False)
    print(f"   Gespeichert: {out}  ({len(squads)} Teams)")
    return out


def load_nation_strength() -> pd.DataFrame:
    """Lädt nation_strength.parquet, baut es bei Bedarf."""
    p = PROCESSED_DIR / "nation_strength.parquet"
    if not p.exists():
        build_nation_strength()
    return pd.read_parquet(p)


def load_squads_2026() -> dict[str, dict]:
    """Lädt squads_2026.json, baut es bei Bedarf."""
    p = PROCESSED_DIR / "squads_2026.json"
    if not p.exists():
        build_squads_2026()
    with open(p, encoding="utf-8") as fh:
        return json.load(fh)


def get_squad_lookup() -> dict[tuple[str, int], dict]:
    """
    Gibt einen (nation, year)-Lookup zurück.
    Für unbekannte Jahre wird der nächste vorhandene Jahrgang verwendet (forward/backward fill).
    """
    ns = load_nation_strength()
    lookup: dict[tuple[str, int], dict] = {}
    years_per_nation: dict[str, list[int]] = {}

    for _, row in ns.iterrows():
        key = (row["nation"], int(row["year"]))
        lookup[key] = {
            "sq_ovr": float(row["sq_ovr"]),
            "sq_att": float(row["sq_att"]),
            "sq_def": float(row["sq_def"]),
            "sq_age": float(row["sq_age"]),
            "sq_depth": float(row["sq_depth"]),
        }
        years_per_nation.setdefault(row["nation"], []).append(int(row["year"]))

    for nation in years_per_nation:
        years_per_nation[nation] = sorted(years_per_nation[nation])

    def get(nation: str, year: int) -> dict:
        if (nation, year) in lookup:
            return lookup[(nation, year)]
        yrs = years_per_nation.get(nation)
        if not yrs:
            return _default_squad()
        # nearest year
        nearest = min(yrs, key=lambda y: abs(y - year))
        return lookup.get((nation, nearest), _default_squad())

    return get, years_per_nation  # type: ignore


def _default_squad() -> dict:
    return {
        "sq_ovr": 74.0,
        "sq_att": 73.0,
        "sq_def": 73.0,
        "sq_age": 27.0,
        "sq_depth": 5.0,
    }


def main() -> int:
    build_nation_strength()
    build_squads_2026()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
