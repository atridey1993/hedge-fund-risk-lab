from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm, t, genpareto, skew, kurtosis


def historical_var_cvar(returns: pd.Series, alpha: float = 0.99) -> tuple[float, float]:
    losses = -returns.dropna()
    var = np.quantile(losses, alpha)
    cvar = losses[losses >= var].mean()
    return float(var), float(cvar)


def gaussian_var_cvar(returns: pd.Series, alpha: float = 0.99) -> tuple[float, float]:
    r = returns.dropna()
    mu, sigma = r.mean(), r.std(ddof=1)
    z = norm.ppf(alpha)
    var = -(mu - z * sigma)
    cvar = -(mu - sigma * norm.pdf(z) / (1 - alpha))
    return float(var), float(cvar)


def student_t_var_cvar(returns: pd.Series, alpha: float = 0.99) -> tuple[float, float]:
    r = returns.dropna()
    df, loc, scale = t.fit(r)

    # VaR on loss side: alpha quantile of loss = negative left-tail return quantile.
    q_return = t.ppf(1 - alpha, df, loc=loc, scale=scale)
    var = -q_return

    rng = np.random.default_rng(123)
    sim = t.rvs(df, loc=loc, scale=scale, size=300_000, random_state=rng)
    losses = -sim
    cvar = losses[losses >= var].mean()
    return float(var), float(cvar)


def cornish_fisher_var(returns: pd.Series, alpha: float = 0.99) -> float:
    r = returns.dropna()
    mu, sigma = r.mean(), r.std(ddof=1)
    s = skew(r)
    k = kurtosis(r, fisher=True)
    z = norm.ppf(alpha)

    z_cf = (
        z
        + (z**2 - 1) * s / 6
        + (z**3 - 3*z) * k / 24
        - (2*z**3 - 5*z) * s**2 / 36
    )
    return float(-(mu - z_cf * sigma))


def evt_var_cvar(
    returns: pd.Series,
    alpha: float = 0.99,
    threshold_quantile: float = 0.90,
) -> tuple[float, float]:
    losses = -returns.dropna()
    u = np.quantile(losses, threshold_quantile)
    excess = losses[losses > u] - u

    if len(excess) < 30:
        return historical_var_cvar(returns, alpha)

    shape, loc, scale = genpareto.fit(excess, floc=0)
    n = len(losses)
    nu = len(excess)

    if abs(shape) > 1e-8:
        var = u + scale / shape * (((n / nu) * (1 - alpha)) ** (-shape) - 1)
    else:
        var = u - scale * np.log((n / nu) * (1 - alpha))

    if shape < 1:
        cvar = (var + scale - shape * u) / (1 - shape)
    else:
        cvar = np.nan

    return float(var), float(cvar)


def monte_carlo_portfolio(
    returns: pd.DataFrame,
    weights: dict[str, float],
    n_sims: int = 120_000,
    horizon_days: int = 20,
    random_state: int = 42,
    use_student_t: bool = True,
) -> pd.Series:
    rng = np.random.default_rng(random_state)
    asset_cols = [c for c in returns.columns if c != "TrueRegime"]
    r = returns[asset_cols].dropna()

    mu = r.mean().values
    cov = r.cov().values
    w = np.array([weights[a] for a in asset_cols])

    if use_student_t:
        df = 6
        z = rng.standard_t(df=df, size=(n_sims, horizon_days, len(asset_cols)))
        z = z / np.sqrt(df / (df - 2))
    else:
        z = rng.normal(size=(n_sims, horizon_days, len(asset_cols)))

    chol = np.linalg.cholesky(cov)
    daily_sims = mu + z @ chol.T
    portfolio_daily = daily_sims @ w
    horizon_returns = np.prod(1 + portfolio_daily, axis=1) - 1

    return pd.Series(horizon_returns, name=f"{horizon_days}D_MC_Return")


def risk_contribution(returns: pd.DataFrame, weights: dict[str, float]) -> pd.DataFrame:
    asset_cols = [c for c in returns.columns if c != "TrueRegime"]
    cov = returns[asset_cols].cov().values
    w = np.array([weights[a] for a in asset_cols])
    port_vol = np.sqrt(w.T @ cov @ w)

    marginal = cov @ w / port_vol
    component = w * marginal
    pct = component / port_vol

    return pd.DataFrame({
        "asset": asset_cols,
        "weight": w,
        "marginal_risk": marginal,
        "component_risk": component,
        "risk_contribution_pct": pct,
    })


def rolling_var_backtest(
    returns: pd.Series,
    window: int = 500,
    alpha: float = 0.99,
    method: str = "historical",
) -> pd.DataFrame:
    from src.regime import regime_weighted_var_cvar

    rows = []
    r = returns.dropna()

    for i in range(window, len(r)):
        train = r.iloc[i-window:i]
        realized = r.iloc[i]

        if method == "historical":
            var, cvar = historical_var_cvar(train, alpha)
        elif method == "regime_weighted":
            result = regime_weighted_var_cvar(train, alpha=alpha)
            var, cvar = result["var"], result["cvar"]
        else:
            raise ValueError("method must be 'historical' or 'regime_weighted'")

        rows.append({
            "date": r.index[i],
            "return": realized,
            "loss": -realized,
            "VaR": var,
            "CVaR": cvar,
            "exception": int(-realized > var),
            "method": method,
        })

    return pd.DataFrame(rows).set_index("date")
