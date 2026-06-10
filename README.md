# Adaptive Regime-Weighted Tail Risk
## Monte Carlo VaR, CVaR, EVT, and Risk Backtesting for Hedge Fund Applications

### Overview

This project develops a hedge-fund-style portfolio risk engine to estimate, validate, and analyze market risk.

The framework combines classical risk management techniques with a novel **Adaptive Regime-Weighted Tail Risk Model**, designed to account for changing market conditions through volatility-based regime detection and dynamic risk weighting.

Unlike standard VaR implementations that treat all historical observations equally, this approach assigns larger importance to observations occurring during stressed market environments and periods of elevated volatility.

The objective is to build a practical risk laboratory that bridges statistical modelling, quantitative finance, and portfolio risk management.

---

## Key Features

### Classical Risk Models

- Historical VaR
- Historical CVaR (Expected Shortfall)
- Gaussian VaR and CVaR
- Student-t VaR and CVaR
- Cornish-Fisher Modified VaR
- Extreme Value Theory (EVT) Tail Risk
- Monte Carlo VaR and CVaR

### Risk Validation

- Rolling VaR Backtesting
- Kupiec Coverage Test
- Christoffersen Independence Test
- Exception Analysis

### Portfolio Analytics

- Risk Contribution Analysis
- Asset-Level Risk Decomposition
- Stress Testing Framework
- Scenario Analysis

### Novel Contribution

- Volatility Regime Detection
- Adaptive Regime-Weighted VaR
- Adaptive Regime-Weighted CVaR
- Dynamic Tail-Risk Adjustment
- Recency Weighting
- Volatility-Adaptive Risk Estimation

---

# Motivation

Traditional Historical VaR assumes that:

\[
P(r_t)
=
P(r_{t+1})
\]

for all periods.

In practice, financial markets exhibit:

- Volatility clustering
- Crisis periods
- Regime shifts
- Changing correlations
- Non-stationary behaviour

A loss observed during a stressed market environment is often more informative about future downside risk than a loss observed during a calm market environment.

To address this limitation, this project introduces a volatility-adaptive weighting framework that increases the influence of stressed observations when estimating tail risk.

---

# Methodology

## Step 1: Portfolio Construction

Portfolio returns are computed as

\[
R_p
=
\mathbf w^T \mathbf r
\]

where

- \( \mathbf w \) = portfolio weights
- \( \mathbf r \) = asset return vector

Portfolio volatility is

\[
\sigma_p
=
\sqrt{
\mathbf w^T
\Sigma
\mathbf w
}
\]

---

## Step 2: Monte Carlo Simulation

Future market scenarios are generated using correlated random variables.

Covariance structure:

\[
\Sigma = LL^T
\]

Simulated returns:

\[
R = \mu + LZ
\]

where

\[
Z \sim N(0,I)
\]

or

\[
Z \sim t_\nu
\]

for heavy-tailed simulations.

---

## Step 3: Tail Risk Estimation

The framework computes:

### Historical VaR

\[
VaR_\alpha
=
Q_\alpha(L)
\]

### CVaR

\[
CVaR_\alpha
=
E[L \mid L > VaR_\alpha]
\]

### EVT VaR

Using a Peaks-over-Threshold approach:

\[
Y = L-u
\]

and fitting a Generalized Pareto Distribution.

---

## Step 4: Volatility Regime Detection

Rolling volatility:

\[
\sigma_t^{roll}
=
\sqrt{
\frac1{W-1}
\sum
(r_i-\bar r)^2
}
\]

Market states are classified into:

- Calm
- Normal
- Stress

using volatility quantiles.

---

## Step 5: Adaptive Regime-Weighted Tail Risk

Instead of assigning equal importance to all observations,

weights are determined dynamically using:

\[
w_t
=
1
+
\lambda
\frac{
\sigma_t^{roll}
}{
\bar{\sigma}^{roll}
}
\]

where

- high-volatility periods receive larger weights
- calm periods receive smaller weights

Adaptive VaR becomes

\[
VaR_\alpha^{RW}
=
Q_\alpha^{weighted}(L,w)
\]

and

\[
CVaR_\alpha^{RW}
=
\frac{
\sum w_t L_t
}{
\sum w_t
}
\]

over tail observations.

---

# Repository Structure

```text
hedge-fund-risk-lab/
│
├── README.md
├── requirements.txt
├── run_project.py
│
├── src/
│   ├── data.py
│   ├── risk.py
│   ├── regime.py
│   ├── backtest.py
│   ├── stress.py
│   └── plots.py
│
├── docs/
│   ├── Risk_Report.pdf
│   └── Risk_Report.tex
│
├── outputs/
│   ├── mc_loss_distribution.png
│   ├── risk_contribution.png
│   ├── detected_volatility_regimes.png
│   ├── rolling_var_backtest.png
│   └── stress_scenarios.png
│
└── notebooks/
```

# Sample Outputs

The project automatically generates:

- Monte Carlo Loss Distribution
- VaR/CVaR Model Comparison
- Volatility Regime Detection
- Rolling VaR Backtests
- Risk Contribution Analysis
- Stress Scenario Performance

---

# Model Validation

The Adaptive Regime-Weighted Risk Model is evaluated against standard Historical VaR using:

### Kupiec Test

Tests whether the observed exception frequency matches the expected frequency.

### Christoffersen Test

Tests whether VaR exceptions are independent or clustered.

### Stress Period Performance

Evaluates model responsiveness during volatile market environments.

---

# Main Findings

The adaptive framework generally produces:

- Higher tail-risk estimates during volatile periods
- More conservative CVaR estimates
- Greater sensitivity to changing market conditions
- Improved economic interpretability of downside risk

The methodology is intended as a practical enhancement to classical Historical VaR rather than a replacement.

---

# Future Extensions

Potential future improvements include:

- GARCH Volatility Forecasting
- Filtered Historical Simulation
- CVaR Portfolio Optimization
- Risk Parity Construction
- Factor-Based Risk Models
- Liquidity-Adjusted VaR
- Hidden Markov Regime Detection
- Machine Learning-Based Risk Forecasting

---

# Technologies

- Python
- NumPy
- Pandas
- SciPy
- Matplotlib
- Statistical Modelling
- Monte Carlo Simulation
- Quantitative Finance

---

# CV Summary

Built a hedge-fund-style portfolio risk engine implementing Monte Carlo VaR/CVaR, EVT tail-risk modelling, rolling backtesting, Kupiec and Christoffersen validation tests, stress testing, risk contribution analysis, and a novel Adaptive Regime-Weighted Tail Risk framework based on volatility regime detection.
