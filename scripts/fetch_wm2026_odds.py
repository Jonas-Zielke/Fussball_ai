"""
Fetch WM 2026 group-stage bookmaker odds and save to data/raw/wm2026_odds.json.

Usage:
    python scripts/fetch_wm2026_odds.py            # uses FALLBACK_ODDS (no API key needed)
    WM_ODDS_API_KEY=xxx python scripts/fetch_wm2026_odds.py  # fetches live from the-odds-api.com

Output: data/raw/wm2026_odds.json
  {
    "source": ...,
    "fetched": "YYYY-MM-DD",
    "blend_weight": 0.45,
    "matches": {
      "Netherlands|Japan": {"home": 0.65, "draw": 0.21, "away": 0.14},
      "Japan|Netherlands": {"home": 0.14, "draw": 0.21, "away": 0.65},
      ...
    }
  }
Both directions are stored for O(1) lookup regardless of home/away assignment.
"""
from __future__ import annotations

import json
import os
import sys
from datetime import date
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_FILE = REPO_ROOT / "data" / "raw" / "wm2026_odds.json"
BLEND_WEIGHT = 0.45
SPORT_KEY = "soccer_fifa_world_cup"

# the-odds-api uses its own team spellings ("USA", "Korea Republic", "Türkiye");
# the model looks odds up by martj42-canonical names, so live keys MUST be
# normalized or every live lookup silently misses (the hardcoded fallback only
# works because it is already written in canonical names).
sys.path.insert(0, str(REPO_ROOT))
from src.team_normalize import normalize_team_name  # noqa: E402

# ---------------------------------------------------------------------------
# Fallback odds (current pre-tournament market, vig-removed)
# Keyed as "Home|Away" matching wc2026.json canonical team names.
# Probabilities sum to 1.0 (vig already removed).
# ---------------------------------------------------------------------------
FALLBACK_ODDS: dict[str, dict[str, float]] = {
    # ── Group A: Mexico, South Korea, Czech Republic, South Africa ──────────
    "Mexico|South Africa":        {"home": 0.70, "draw": 0.17, "away": 0.13},
    "South Korea|Czech Republic": {"home": 0.38, "draw": 0.28, "away": 0.34},
    "Czech Republic|South Africa":{"home": 0.50, "draw": 0.27, "away": 0.23},
    "Mexico|South Korea":         {"home": 0.52, "draw": 0.26, "away": 0.22},
    "Czech Republic|Mexico":      {"home": 0.32, "draw": 0.25, "away": 0.43},
    "South Africa|South Korea":   {"home": 0.30, "draw": 0.27, "away": 0.43},

    # ── Group B: Canada, Switzerland, Qatar, Bosnia and Herzegovina ──────────
    "Canada|Bosnia and Herzegovina": {"home": 0.46, "draw": 0.27, "away": 0.27},
    "Qatar|Switzerland":             {"home": 0.24, "draw": 0.25, "away": 0.51},
    "Switzerland|Bosnia and Herzegovina": {"home": 0.53, "draw": 0.26, "away": 0.21},
    "Canada|Qatar":                  {"home": 0.61, "draw": 0.22, "away": 0.17},
    "Switzerland|Canada":            {"home": 0.48, "draw": 0.27, "away": 0.25},
    "Bosnia and Herzegovina|Qatar":  {"home": 0.40, "draw": 0.27, "away": 0.33},

    # ── Group C: Brazil, Morocco, Haiti, Scotland ────────────────────────────
    "Brazil|Morocco":    {"home": 0.63, "draw": 0.22, "away": 0.15},
    "Haiti|Scotland":    {"home": 0.21, "draw": 0.23, "away": 0.56},
    "Scotland|Morocco":  {"home": 0.37, "draw": 0.28, "away": 0.35},
    "Brazil|Haiti":      {"home": 0.84, "draw": 0.11, "away": 0.05},
    "Scotland|Brazil":   {"home": 0.12, "draw": 0.18, "away": 0.70},
    "Morocco|Haiti":     {"home": 0.69, "draw": 0.20, "away": 0.11},

    # ── Group D: United States, Paraguay, Australia, Turkey ──────────────────
    "United States|Paraguay": {"home": 0.51, "draw": 0.25, "away": 0.24},
    "Australia|Turkey":       {"home": 0.38, "draw": 0.28, "away": 0.34},
    "United States|Australia":{"home": 0.53, "draw": 0.25, "away": 0.22},
    "Turkey|Paraguay":        {"home": 0.43, "draw": 0.28, "away": 0.29},
    "Turkey|United States":   {"home": 0.33, "draw": 0.25, "away": 0.42},
    "Paraguay|Australia":     {"home": 0.41, "draw": 0.28, "away": 0.31},

    # ── Group E: Germany, Curaçao, Ivory Coast, Ecuador ──────────────────────
    "Germany|Curaçao":       {"home": 0.91, "draw": 0.06, "away": 0.03},
    "Ivory Coast|Ecuador":   {"home": 0.39, "draw": 0.27, "away": 0.34},
    "Germany|Ivory Coast":   {"home": 0.68, "draw": 0.21, "away": 0.11},
    "Ecuador|Curaçao":       {"home": 0.72, "draw": 0.18, "away": 0.10},
    "Ecuador|Germany":       {"home": 0.17, "draw": 0.22, "away": 0.61},
    "Curaçao|Ivory Coast":   {"home": 0.19, "draw": 0.22, "away": 0.59},

    # ── Group F: Netherlands, Japan, Sweden, Tunisia ─────────────────────────
    "Netherlands|Japan":     {"home": 0.65, "draw": 0.22, "away": 0.13},
    "Sweden|Tunisia":        {"home": 0.53, "draw": 0.27, "away": 0.20},
    "Netherlands|Sweden":    {"home": 0.55, "draw": 0.26, "away": 0.19},
    "Tunisia|Japan":         {"home": 0.30, "draw": 0.28, "away": 0.42},
    "Japan|Sweden":          {"home": 0.37, "draw": 0.27, "away": 0.36},
    "Tunisia|Netherlands":   {"home": 0.13, "draw": 0.20, "away": 0.67},

    # ── Group G: Belgium, Egypt, Iran, New Zealand ───────────────────────────
    "Belgium|Egypt":         {"home": 0.69, "draw": 0.19, "away": 0.12},
    "Iran|New Zealand":      {"home": 0.49, "draw": 0.27, "away": 0.24},
    "Belgium|Iran":          {"home": 0.65, "draw": 0.22, "away": 0.13},
    "New Zealand|Egypt":     {"home": 0.29, "draw": 0.28, "away": 0.43},
    "Egypt|Iran":            {"home": 0.37, "draw": 0.27, "away": 0.36},
    "New Zealand|Belgium":   {"home": 0.11, "draw": 0.18, "away": 0.71},

    # ── Group H: Spain, Cape Verde, Saudi Arabia, Uruguay ───────────────────
    "Spain|Cape Verde":      {"home": 0.89, "draw": 0.07, "away": 0.04},
    "Saudi Arabia|Uruguay":  {"home": 0.34, "draw": 0.25, "away": 0.41},
    "Spain|Saudi Arabia":    {"home": 0.73, "draw": 0.18, "away": 0.09},
    "Uruguay|Cape Verde":    {"home": 0.76, "draw": 0.16, "away": 0.08},
    "Cape Verde|Saudi Arabia":{"home": 0.30, "draw": 0.27, "away": 0.43},
    "Uruguay|Spain":         {"home": 0.21, "draw": 0.22, "away": 0.57},

    # ── Group I: France, Senegal, Iraq, Norway ───────────────────────────────
    "France|Senegal":   {"home": 0.63, "draw": 0.22, "away": 0.15},
    "Iraq|Norway":      {"home": 0.28, "draw": 0.25, "away": 0.47},
    "France|Iraq":      {"home": 0.81, "draw": 0.12, "away": 0.07},
    "Norway|Senegal":   {"home": 0.46, "draw": 0.27, "away": 0.27},
    "Norway|France":    {"home": 0.27, "draw": 0.24, "away": 0.49},
    "Senegal|Iraq":     {"home": 0.55, "draw": 0.25, "away": 0.20},

    # ── Group J: Argentina, Algeria, Austria, Jordan ─────────────────────────
    "Argentina|Algeria": {"home": 0.79, "draw": 0.14, "away": 0.07},
    "Austria|Jordan":    {"home": 0.59, "draw": 0.25, "away": 0.16},
    "Argentina|Austria": {"home": 0.67, "draw": 0.21, "away": 0.12},
    "Jordan|Algeria":    {"home": 0.30, "draw": 0.27, "away": 0.43},
    "Algeria|Austria":   {"home": 0.30, "draw": 0.26, "away": 0.44},
    "Jordan|Argentina":  {"home": 0.07, "draw": 0.13, "away": 0.80},

    # ── Group K: Portugal, DR Congo, Uzbekistan, Colombia ───────────────────
    "Portugal|DR Congo":    {"home": 0.83, "draw": 0.11, "away": 0.06},
    "Uzbekistan|Colombia":  {"home": 0.24, "draw": 0.25, "away": 0.51},
    "Portugal|Uzbekistan":  {"home": 0.81, "draw": 0.12, "away": 0.07},
    "Colombia|DR Congo":    {"home": 0.65, "draw": 0.22, "away": 0.13},
    "Colombia|Portugal":    {"home": 0.24, "draw": 0.23, "away": 0.53},
    "DR Congo|Uzbekistan":  {"home": 0.37, "draw": 0.27, "away": 0.36},

    # ── Group L: England, Croatia, Ghana, Panama ─────────────────────────────
    "England|Croatia":  {"home": 0.56, "draw": 0.25, "away": 0.19},
    "Ghana|Panama":     {"home": 0.43, "draw": 0.27, "away": 0.30},
    "England|Ghana":    {"home": 0.71, "draw": 0.18, "away": 0.11},
    "Panama|Croatia":   {"home": 0.27, "draw": 0.27, "away": 0.46},
    "Panama|England":   {"home": 0.11, "draw": 0.17, "away": 0.72},
    "Croatia|Ghana":    {"home": 0.56, "draw": 0.25, "away": 0.19},
}


def _remove_vig(raw: dict[str, float]) -> dict[str, float]:
    """Normalize raw implied probs (1/odd) to remove vig."""
    total = sum(raw.values())
    if total <= 0:
        return raw
    return {k: v / total for k, v in raw.items()}


def _mirror(matches: dict) -> dict:
    """Add reversed keys so lookup works regardless of which team is 'home'."""
    result = dict(matches)
    for key, v in matches.items():
        home, away = key.split("|", 1)
        rev = f"{away}|{home}"
        if rev not in result:
            result[rev] = {"home": v["away"], "draw": v["draw"], "away": v["home"]}
    return result


def fetch_live(api_key: str) -> dict | None:
    """Fetch odds from the-odds-api.com. Returns matches dict or None on error."""
    try:
        import urllib.request
        import urllib.parse

        params = urllib.parse.urlencode({
            "apiKey": api_key,
            "regions": "eu",
            "markets": "h2h",
            "oddsFormat": "decimal",
        })
        url = f"https://api.the-odds-api.com/v4/sports/{SPORT_KEY}/odds/?{params}"
        with urllib.request.urlopen(url, timeout=30) as resp:
            data = json.loads(resp.read().decode())

        matches: dict[str, dict[str, float]] = {}
        for game in data:
            home_team = game.get("home_team", "")
            away_team = game.get("away_team", "")
            # Aggregate odds across bookmakers
            sums: dict[str, list[float]] = {"home": [], "draw": [], "away": []}
            for bk in game.get("bookmakers", []):
                for mkt in bk.get("markets", []):
                    if mkt.get("key") != "h2h":
                        continue
                    outcomes = {o["name"]: o["price"] for o in mkt.get("outcomes", [])}
                    if home_team in outcomes and away_team in outcomes and "Draw" in outcomes:
                        sums["home"].append(1.0 / outcomes[home_team])
                        sums["draw"].append(1.0 / outcomes["Draw"])
                        sums["away"].append(1.0 / outcomes[away_team])
            if sums["home"]:
                raw = {
                    "home": sum(sums["home"]) / len(sums["home"]),
                    "draw": sum(sums["draw"]) / len(sums["draw"]),
                    "away": sum(sums["away"]) / len(sums["away"]),
                }
                # normalize to martj42-canonical so model lookups hit (knockouts incl.)
                key = f"{normalize_team_name(home_team)}|{normalize_team_name(away_team)}"
                matches[key] = _remove_vig(raw)

        return matches if matches else None
    except Exception as exc:
        print(f"   Live fetch failed: {exc}", file=sys.stderr)
        return None


def main() -> None:
    api_key = os.environ.get("WM_ODDS_API_KEY", "")
    matches: dict | None = None
    source = "fallback-hardcoded"

    if api_key:
        print(">> Versuche Live-Fetch von the-odds-api.com...")
        matches = fetch_live(api_key)
        if matches:
            source = "the-odds-api.com"
            print(f"   {len(matches)} Spiele geladen (live)")
        else:
            print("   Fallback auf hardcoded Quoten")

    if matches is None:
        matches = {k: _remove_vig(v) for k, v in FALLBACK_ODDS.items()}
        print(f">> Verwende hardcoded Fallback-Quoten ({len(matches)} Spiele)")

    matches = _mirror(matches)

    out = {
        "source": source,
        "fetched": date.today().isoformat(),
        "blend_weight": BLEND_WEIGHT,
        "matches": matches,
    }

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_FILE, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, ensure_ascii=False)

    print(f"   gespeichert: {OUT_FILE} ({OUT_FILE.stat().st_size / 1024:.1f} KB, {len(matches)} Einträge)")


if __name__ == "__main__":
    main()
