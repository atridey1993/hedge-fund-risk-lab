from __future__ import annotations

import numpy as np
import pandas as pd


def generate_synthetic_returns(
    n_days: int = 1800,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Generate realistic synthetic multi-asset returns.

    Features:
    - five asset classes
    - realistic covariance structure
    - Student-t fat-tailed shocks
    - hidden calm/normal/stress regimes
    """
    rng = np.random.default_rng(random_state)

    assets = ["Equity", "Rates", "Credit", "FX", "Commodity"]

    mu = np.array([0.07, 0.025, 0.04, 0.015, 0.035]) / 252
    vol = np.array([0.18, 0.06, 0.10, 0.09, 0.22]) / np.sqrt(252)

    corr = np.array([
        [1.00, -0.20, 0.55, 0.20, 0.30],
        [-0.20, 1.00, -0.10, -0.05, -0.15],
        [0.55, -0.10, 1.00, 0.15, 0.25],
        [0.20, -0.05, 0.15, 1.00, 0.20],
        [0.30, -0.15, 0.25, 0.20, 1.00],
    ])

    cov = np.outer(vol, vol) * corr
    chol = np.linalg.cholesky(cov)

    # Hidden volatility regimes:
    # 0 = calm, 1 = normal, 2 = stress
    regime_probs = np.array([0.58, 0.34, 0.08])
    regime_scale = np.array([0.65, 1.00, 2.80])
    regimes = rng.choice(3, size=n_days, p=regime_probs)

    # Student-t shocks create fat tails.
    df = 5
    z = rng.standard_t(df=df, size=(n_days, len(assets)))
    z = z / np.sqrt(df / (df - 2))

    returns = mu + (z @ chol.T) * regime_scale[regimes, None]

    dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=n_days)
    df_returns = pd.DataFrame(returns, index=dates, columns=assets)
    df_returns["TrueRegime"] = regimes

    return df_returns


def get_portfolio_returns(returns: pd.DataFrame, weights: dict[str, float]) -> pd.Series:
    asset_cols = [c for c in returns.columns if c != "TrueRegime"]
    w = np.array([weights[a] for a in asset_cols])
    return returns[asset_cols].dot(w).rename("Portfolio")
