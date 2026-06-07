"""
KickTipp Decision Layer — Punkte-Erwartungswert-optimaler Tipp.

Kernidee: Statt den wahrscheinlichsten Score (Modus des Poisson-Grids) zu
tippen, suchen wir den Tipp der den ERWARTETEN KICKTIPP-PUNKTWERT maximiert:

    optimal = argmax_tip  Σ_(h,a) P(h,a) · points(tip, (h,a), odds, scheme)

Das macht einen Unterschied, sobald der Quoten-/Risiko-Bonus aktiv ist:
Ein Außenseiter-Tipp mit nur 28 % Wahrscheinlichkeit kann trotzdem
mehr erwartete Punkte liefern als der sichere Favorit, wenn die Quoten hoch
genug sind.

Verwendung:
    from src.kicktipp import load_scheme, optimal_tip

    scheme = load_scheme()
    result = optimal_tip(grid, odds_probs, scheme)
    print(result["tip"], result["expected_points"])
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import TypedDict

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SCHEME_PATH = REPO_ROOT / "data" / "raw" / "kicktipp_scheme.json"


# ---------------------------------------------------------------------------
# Scheme definition
# ---------------------------------------------------------------------------

@dataclass
class RisikoBonus:
    enabled: bool = True
    formula: str = "floor_odds_minus_1"
    max_bonus: int = 8


@dataclass
class Scheme:
    base_tendency: int = 2
    base_difference: int = 3
    base_exact: int = 4
    risiko_bonus: RisikoBonus = field(default_factory=RisikoBonus)


def load_scheme(path: Path | None = None) -> Scheme:
    p = path or DEFAULT_SCHEME_PATH
    try:
        with open(p, encoding="utf-8") as fh:
            d = json.load(fh)
        rb_d = d.get("risiko_bonus", {})
        rb = RisikoBonus(
            enabled=bool(rb_d.get("enabled", True)),
            formula=str(rb_d.get("formula", "floor_odds_minus_1")),
            max_bonus=int(rb_d.get("max_bonus", 8)),
        )
        return Scheme(
            base_tendency=int(d.get("base_tendency", 2)),
            base_difference=int(d.get("base_difference", 3)),
            base_exact=int(d.get("base_exact", 4)),
            risiko_bonus=rb,
        )
    except (FileNotFoundError, json.JSONDecodeError):
        return Scheme()


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _tendency(h: int, a: int) -> str:
    if h > a:
        return "home"
    if h == a:
        return "draw"
    return "away"


def _risiko_bonus(
    tipped_tendency: str,
    odds_probs: dict[str, float],
    rb: RisikoBonus,
) -> float:
    if not rb.enabled:
        return 0.0
    # Map tendency to market probability
    prob_map = {"home": odds_probs.get("home", 0.5),
                "draw": odds_probs.get("draw", 0.25),
                "away": odds_probs.get("away", 0.25)}
    p = max(1e-6, prob_map[tipped_tendency])
    decimal_odds = 1.0 / p
    if rb.formula == "floor_odds_minus_1":
        bonus = max(0.0, math.floor(decimal_odds - 1.0))
    else:
        bonus = 0.0
    return min(float(bonus), float(rb.max_bonus))


def points(
    tip: tuple[int, int],
    actual: tuple[int, int],
    odds_probs: dict[str, float],
    scheme: Scheme,
) -> float:
    """Points scored for tipping `tip` when the actual result is `actual`.

    Args:
        tip: (home_goals_tipped, away_goals_tipped)
        actual: (home_goals, away_goals)
        odds_probs: bookmaker probabilities {"home": p, "draw": p, "away": p}
            (sum ≈ 1.0, not decimal odds). If empty, bonus is 0.
        scheme: scoring scheme

    Returns:
        float: KickTipp points (0 if tendency wrong)
    """
    th, ta = tip
    ah, aa = actual
    tip_tend = _tendency(th, ta)
    act_tend = _tendency(ah, aa)

    if tip_tend != act_tend:
        return 0.0

    # Base points
    if th == ah and ta == aa:
        base = float(scheme.base_exact)
    elif (th - ta) == (ah - aa):
        base = float(scheme.base_difference)
    else:
        base = float(scheme.base_tendency)

    # Risiko-Bonus (earned for tipping the correct tendency correctly,
    # regardless of whether difference/exact was right too)
    bonus = _risiko_bonus(tip_tend, odds_probs, scheme.risiko_bonus)

    return base + bonus


# ---------------------------------------------------------------------------
# Optimal tip
# ---------------------------------------------------------------------------

class TipResult(TypedDict):
    tip: tuple[int, int]
    expected_points: float
    alternatives: list[dict]  # top-5 tips with their expected_points


def optimal_tip(
    grid: np.ndarray,
    odds_probs: dict[str, float],
    scheme: Scheme,
    n_scores: int = 10,
) -> TipResult:
    """Find the score tip that maximises expected KickTipp points.

    Args:
        grid: (n x n) bivariate Poisson probability array P[home_goals][away_goals]
        odds_probs: bookmaker win/draw/loss probabilities (as fractions, not decimal)
        scheme: KickTipp scoring scheme
        n_scores: grid size to consider for actual results

    Returns:
        TipResult with best tip and expected_points for top alternatives
    """
    n = min(grid.shape[0], n_scores)
    best_ev = -1.0
    best_tip: tuple[int, int] = (1, 1)
    all_tips: list[dict] = []

    for th in range(n):
        for ta in range(n):
            ev = 0.0
            for ah in range(n):
                for aa in range(n):
                    p = float(grid[ah, aa])
                    if p < 1e-9:
                        continue
                    ev += p * points((th, ta), (ah, aa), odds_probs, scheme)
            all_tips.append({"tip": (th, ta), "expected_points": round(ev, 4)})
            if ev > best_ev:
                best_ev = ev
                best_tip = (th, ta)

    all_tips.sort(key=lambda x: -x["expected_points"])
    return TipResult(
        tip=best_tip,
        expected_points=round(best_ev, 4),
        alternatives=all_tips[:5],
    )
