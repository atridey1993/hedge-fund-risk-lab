from __future__ import annotations

import numpy as np
import pandas as pd


def classify_volatility_regimes(
    returns: pd.Series,
    window: int = 60,
    low_q: float = 0.35,
    high_q: float = 0.75,
) -> pd.DataFrame:
    """
    Classify days into calm, normal, and stress regimes using rolling volatility.
    """
    r = returns.dropna()
    roll_vol = r.rolling(window).std()

    low = roll_vol.quantile(low_q)
    high = roll_vol.quantile(high_q)

    regime = pd.Series("normal", index=r.index)
    regime[roll_vol <= low] = "calm"
    regime[roll_vol >= high] = "stress"

    out = pd.DataFrame({
        "return": r,
        "loss": -r,
        "rolling_vol": roll_vol,
        "regime": regime,
    }).dropna()

    return out


def weighted_quantile(
    values: np.ndarray,
    weights: np.ndarray,
    q: float,
) -> float:
    """
    Compute weighted quantile.
    """
    values = np.asarray(values)
    weights = np.asarray(weights)

    sorter = np.argsort(values)
    values = values[sorter]
    weights = weights[sorter]

    cumulative = np.cumsum(weights)
    cumulative = cumulative / cumulative[-1]

    return float(np.interp(q, cumulative, values))


def regime_weighted_var_cvar(
    returns: pd.Series,
    alpha: float = 0.99,
    window: int = 60,
    lambda_vol: float = 1.5,
    recent_halflife: int | None = 252,
) -> dict:
    """
    Data-driven Adaptive Regime-Weighted VaR/CVaR.

    Main idea:

        weight_t = 1 + lambda_vol * (rolling_vol_t / average_rolling_vol)

    So high-volatility periods automatically receive larger weight.
    """

    df = classify_volatility_regimes(returns, window=window)

    losses = df["loss"].values
    rolling_vol = df["rolling_vol"].values

    avg_vol = np.nanmean(rolling_vol)

    vol_weights = 1.0 + lambda_vol * (rolling_vol / avg_vol)

    weights = vol_weights.copy()

    if recent_halflife is not None:
        n = len(df)

        # oldest observation gets largest age, newest gets age 0
        age = np.arange(n)[::-1]

        recency_weights = 0.5 ** (age / recent_halflife)

        weights = weights * recency_weights

    var = weighted_quantile(losses, weights, alpha)

    tail_mask = losses >= var

    if tail_mask.sum() == 0:
        cvar = var
    else:
        cvar = np.average(losses[tail_mask], weights=weights[tail_mask])

    regime_counts = df["regime"].value_counts(normalize=True).to_dict()

    return {
        "var": float(var),
        "cvar": float(cvar),
        "classified": df,
        "regime_counts": regime_counts,
        "weights": weights,
        "lambda_vol": lambda_vol,
        "average_rolling_vol": float(avg_vol),
    }


def compare_standard_vs_regime_weighted(
    returns: pd.Series,
    alpha: float = 0.99,
) -> pd.DataFrame:
    from src.risk import historical_var_cvar

    hist_var, hist_cvar = historical_var_cvar(returns, alpha=alpha)
    reg = regime_weighted_var_cvar(returns, alpha=alpha)

    return pd.DataFrame([
        {
            "Model": "Historical equal-weight",
            "VaR": hist_var,
            "CVaR": hist_cvar,
        },
        {
            "Model": "Volatility-adaptive weighted",
            "VaR": reg["var"],
            "CVaR": reg["cvar"],
        },
    ])
