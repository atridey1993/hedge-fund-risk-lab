from __future__ import annotations

from pathlib import Path
import pandas as pd

from src.data import generate_synthetic_returns, get_portfolio_returns
from src.risk import (
    historical_var_cvar,
    gaussian_var_cvar,
    student_t_var_cvar,
    cornish_fisher_var,
    evt_var_cvar,
    monte_carlo_portfolio,
    rolling_var_backtest,
    risk_contribution,
)
from src.regime import (
    classify_volatility_regimes,
    regime_weighted_var_cvar,
    compare_standard_vs_regime_weighted,
)
from src.backtest import kupiec_test, christoffersen_independence_test
from src.stress import stress_scenarios
from src.plots import (
    save_return_distribution,
    save_model_comparison,
    save_backtest_plot,
    save_backtest_comparison,
    save_risk_contribution_plot,
    save_stress_plot,
    save_regime_plot,
    save_regime_var_comparison,
)

OUT = Path("outputs")
OUT.mkdir(exist_ok=True)

weights = {
    "Equity": 0.40,
    "Rates": 0.20,
    "Credit": 0.20,
    "FX": 0.10,
    "Commodity": 0.10,
}

alpha = 0.99

# ------------------------------------------------------------
# Data and portfolio
# ------------------------------------------------------------
returns = generate_synthetic_returns(n_days=1800)
portfolio = get_portfolio_returns(returns, weights)

returns.to_csv(OUT / "synthetic_asset_returns.csv")
portfolio.to_csv(OUT / "portfolio_returns.csv")

# ------------------------------------------------------------
# Main risk models
# ------------------------------------------------------------
mc_20d = monte_carlo_portfolio(
    returns=returns,
    weights=weights,
    n_sims=120_000,
    horizon_days=20,
    use_student_t=True,
)

mc_var, mc_cvar = historical_var_cvar(mc_20d, alpha=alpha)

models = []
for name, fn in [
    ("Historical", historical_var_cvar),
    ("Gaussian", gaussian_var_cvar),
    ("Student-t", student_t_var_cvar),
    ("EVT", evt_var_cvar),
]:
    var, cvar = fn(portfolio, alpha=alpha)
    models.append({"Model": name, "VaR": var, "CVaR": cvar})

cf_var = cornish_fisher_var(portfolio, alpha=alpha)
models.append({"Model": "Cornish-Fisher", "VaR": cf_var, "CVaR": float("nan")})
models.append({"Model": "MC 20D Student-t", "VaR": mc_var, "CVaR": mc_cvar})

# ------------------------------------------------------------
# New unique method: adaptive regime-weighted risk
# ------------------------------------------------------------
regime_result = regime_weighted_var_cvar(portfolio, alpha=alpha)
models.append({
    "Model": "Adaptive regime-weighted",
    "VaR": regime_result["var"],
    "CVaR": regime_result["cvar"],
})

model_df = pd.DataFrame(models)
model_df.to_csv(OUT / "risk_model_comparison.csv", index=False)

classified = regime_result["classified"]
classified.to_csv(OUT / "classified_volatility_regimes.csv")

regime_compare = compare_standard_vs_regime_weighted(portfolio, alpha=alpha)
regime_compare.to_csv(OUT / "standard_vs_regime_weighted.csv", index=False)

# ------------------------------------------------------------
# Backtests
# ------------------------------------------------------------
hist_bt = rolling_var_backtest(portfolio, window=500, alpha=alpha, method="historical")
regime_bt = rolling_var_backtest(portfolio, window=500, alpha=alpha, method="regime_weighted")

hist_bt.to_csv(OUT / "rolling_var_backtest_historical.csv")
regime_bt.to_csv(OUT / "rolling_var_backtest_regime_weighted.csv")

bt_summary = pd.DataFrame([
    {"method": "historical", **kupiec_test(hist_bt["exception"], alpha=alpha)},
    {"method": "regime_weighted", **kupiec_test(regime_bt["exception"], alpha=alpha)},
])
bt_summary.to_csv(OUT / "kupiec_comparison.csv", index=False)

christoffersen_summary = pd.DataFrame([
    {"method": "historical", **christoffersen_independence_test(hist_bt["exception"])},
    {"method": "regime_weighted", **christoffersen_independence_test(regime_bt["exception"])},
])
christoffersen_summary.to_csv(OUT / "christoffersen_comparison.csv", index=False)

# ------------------------------------------------------------
# Risk contribution and stress testing
# ------------------------------------------------------------
rc = risk_contribution(returns, weights)
rc.to_csv(OUT / "risk_contribution.csv", index=False)

stress = stress_scenarios(weights)
stress.to_csv(OUT / "stress_scenarios.csv", index=False)

# ------------------------------------------------------------
# Plots
# ------------------------------------------------------------
save_return_distribution(mc_20d, mc_var, mc_cvar, OUT / "mc_loss_distribution.png")
save_model_comparison(model_df.fillna(0), OUT / "var_cvar_model_comparison.png")
save_backtest_plot(hist_bt, OUT / "rolling_var_backtest_historical.png", "Rolling VaR backtest: historical VaR")
save_backtest_plot(regime_bt, OUT / "rolling_var_backtest_regime_weighted.png", "Rolling VaR backtest: adaptive regime-weighted VaR")
save_backtest_comparison(hist_bt, regime_bt, OUT / "historical_vs_regime_weighted_backtest.png")
save_risk_contribution_plot(rc, OUT / "risk_contribution.png")
save_stress_plot(stress, OUT / "stress_scenarios.png")
save_regime_plot(classified, OUT / "detected_volatility_regimes.png")
save_regime_var_comparison(regime_compare, OUT / "standard_vs_regime_weighted.png")

# ------------------------------------------------------------
# Console summary
# ------------------------------------------------------------
print("\n=== Hedge Fund Risk Lab with Adaptive Regime Risk completed ===")
print("\nRisk model comparison:")
print(model_df.to_string(index=False))

print("\nRegime counts:")
print(regime_result["regime_counts"])

print("\nKupiec comparison:")
print(bt_summary.to_string(index=False))

print("\nChristoffersen comparison:")
print(christoffersen_summary.to_string(index=False))

print("\nOutputs saved in ./outputs/")
