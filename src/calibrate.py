"""
λ-Kalibrierung für Poisson-Modelle (V8/V9).

Affine Korrektur im log-Raum, auf dem Cal-Split gefittet:
    log λ' = a + b · log λ̂
Das ist ein Poisson-GLM mit einem Feature (log λ̂) und entspricht genau der
im train_v8-Docstring versprochenen (aber nie implementierten) Kalibrierung —
nur direkt auf den λs statt Temperatur auf W/D/L, damit Scoreline-Tipps und
Tendenz-Wahrscheinlichkeiten konsistent aus EINEM korrigierten Grid kommen.

b < 1  → Modell war überkonfident (λ-Spreizung wird gedämpft)
b > 1  → Modell war unterkonfident (Spreizung wird verstärkt)
"""
from __future__ import annotations

import numpy as np


def fit_affine_lam(cal_log_lams: np.ndarray, y_hg: np.ndarray, y_ag: np.ndarray) -> dict:
    """Fittet je eine affine log-λ-Korrektur für Heim- und Auswärtstore.

    Args:
        cal_log_lams: (N, 2) [log λ_home, log λ_away] auf dem Cal-Split
        y_hg, y_ag:   tatsächliche Tore
    Returns:
        {"a_home", "b_home", "a_away", "b_away"}
    """
    from sklearn.linear_model import PoissonRegressor

    out = {}
    for side, (ll, y) in {
        "home": (cal_log_lams[:, 0:1], y_hg),
        "away": (cal_log_lams[:, 1:2], y_ag),
    }.items():
        reg = PoissonRegressor(alpha=1e-8, max_iter=500)
        reg.fit(ll, np.asarray(y, dtype=np.float64))
        out[f"a_{side}"] = float(reg.intercept_)
        out[f"b_{side}"] = float(reg.coef_[0])
    return out


def apply_affine_lam(log_lams: np.ndarray, cal: dict) -> np.ndarray:
    """Wendet die affine Korrektur auf (N, 2) log-λs an."""
    out = np.empty_like(log_lams)
    out[:, 0] = cal["a_home"] + cal["b_home"] * log_lams[:, 0]
    out[:, 1] = cal["a_away"] + cal["b_away"] * log_lams[:, 1]
    return out
