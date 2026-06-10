from __future__ import annotations

import pandas as pd


def stress_scenarios(weights: dict[str, float]) -> pd.DataFrame:
    """
    Transparent multi-asset stress scenario engine.
    """
    scenarios = {
        "Equity crash": {
            "Equity": -0.20, "Rates": 0.03, "Credit": -0.08,
            "FX": -0.02, "Commodity": -0.05
        },
        "Rates shock": {
            "Equity": -0.05, "Rates": -0.06, "Credit": -0.04,
            "FX": 0.01, "Commodity": -0.02
        },
        "Credit widening": {
            "Equity": -0.08, "Rates": 0.01, "Credit": -0.12,
            "FX": -0.01, "Commodity": -0.03
        },
        "Commodity spike": {
            "Equity": -0.04, "Rates": -0.01, "Credit": -0.02,
            "FX": 0.00, "Commodity": 0.18
        },
        "Risk-on rally": {
            "Equity": 0.10, "Rates": -0.02, "Credit": 0.05,
            "FX": 0.01, "Commodity": 0.04
        },
        "Liquidity crisis": {
            "Equity": -0.15, "Rates": -0.02, "Credit": -0.15,
            "FX": -0.04, "Commodity": -0.10
        },
    }

    rows = []
    for name, shocks in scenarios.items():
        ret = sum(weights[a] * shocks[a] for a in weights)
        rows.append({
            "scenario": name,
            "portfolio_return": ret,
            "portfolio_loss": -ret,
        })

    return pd.DataFrame(rows).sort_values("portfolio_return")
