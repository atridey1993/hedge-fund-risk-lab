# Hedge Fund Risk Lab

### Monte Carlo VaR, CVaR, EVT, and Adaptive Regime-Weighted Tail Risk

---

## Overview

This repository implements a portfolio risk management framework inspired by quantitative risk practices used in hedge funds and asset management.

The project combines classical risk models, simulation-based approaches, tail-risk estimation techniques, statistical backtesting, and a novel Adaptive Regime-Weighted Tail Risk framework designed to account for changing market conditions.

The objective is to investigate how different approaches estimate downside risk and to evaluate whether volatility-aware weighting schemes can improve the economic interpretation of tail-risk measures.

---

## Key Features

### Risk Models

* Historical VaR
* Historical CVaR (Expected Shortfall)
* Gaussian VaR
* Gaussian CVaR
* Student-t VaR
* Student-t CVaR
* Cornish-Fisher Modified VaR
* Extreme Value Theory (EVT)
* Monte Carlo VaR
* Monte Carlo CVaR

### Risk Validation

* Rolling VaR Backtesting
* Kupiec Coverage Test
* Christoffersen Independence Test
* Exception Analysis

### Portfolio Analytics

* Risk Contribution Analysis
* Portfolio Volatility Decomposition
* Stress Testing
* Scenario Analysis

### Novel Contribution

* Volatility Regime Detection
* Adaptive Volatility-Aware Weighting
* Adaptive Regime-Weighted VaR
* Adaptive Regime-Weighted CVaR

---

## Motivation

Traditional Historical VaR treats all observations equally.

However, financial markets exhibit:

* Volatility clustering
* Regime shifts
* Crisis periods
* Structural breaks
* Non-stationary behaviour

A loss observed during a stressed market environment may contain more information about future downside risk than a loss observed during a calm market environment.

To address this limitation, this project introduces a volatility-adaptive weighting framework that increases the influence of observations occurring during elevated volatility regimes when estimating VaR and CVaR.

---

## Methodology

The framework combines several complementary approaches:

### Historical Risk

* Historical VaR
* Historical CVaR

### Parametric Risk

* Gaussian VaR/CVaR
* Student-t VaR/CVaR
* Cornish-Fisher VaR

### Simulation-Based Risk

* Monte Carlo VaR
* Monte Carlo CVaR

### Tail-Risk Modelling

* Extreme Value Theory (EVT)

### Backtesting

* Kupiec Test
* Christoffersen Test

### Adaptive Regime-Weighted Risk

Rolling volatility is used to classify observations into:

* Calm
* Normal
* Stress

market regimes.

Observations occurring during periods of elevated volatility receive larger statistical weights when estimating tail-risk measures.

---

## Results

### Risk Model Comparison

| Model                    | VaR   | CVaR  |
| ------------------------ | ----- | ----- |
| Historical               | 1.68% | 2.53% |
| Gaussian                 | 1.51% | 1.73% |
| Student-t                | 2.19% | 4.19% |
| EVT                      | 1.70% | 2.55% |
| Cornish-Fisher           | 5.18% | N/A   |
| Monte Carlo (20-Day)     | 6.38% | 7.35% |
| Adaptive Regime-Weighted | 1.79% | 2.77% |

The Adaptive Regime-Weighted model produces more conservative tail-risk estimates than standard Historical VaR while maintaining statistically acceptable backtest performance.

---

## Regime Detection Results

Market observations were classified as:

| Regime | Frequency |
| ------ | --------- |
| Calm   | 35.0%     |
| Normal | 39.9%     |
| Stress | 25.0%     |

The presence of a substantial stress regime reflects volatility clustering within the return series and motivates the use of volatility-aware tail-risk estimation.

---

## Backtesting Results

### Kupiec Coverage Test

| Method                   | Exceptions | Expected |
| ------------------------ | ---------- | -------- |
| Historical               | 15         | 13       |
| Adaptive Regime-Weighted | 16         | 13       |

Kupiec p-values:

* Historical: 0.586
* Adaptive Regime-Weighted: 0.420

Both models pass the coverage test.

---

### Christoffersen Independence Test

Christoffersen p-values:

* Historical: 0.554
* Adaptive Regime-Weighted: 0.528

Both models pass the independence test, indicating no statistically significant clustering of exceptions.

---

## Key Findings

* Historical VaR tends to underestimate extreme downside risk.
* Student-t and EVT models capture heavy tails more effectively than Gaussian models.
* Monte Carlo simulation provides forward-looking scenario analysis.
* The Adaptive Regime-Weighted framework produces more conservative tail-risk estimates by emphasizing observations from stressed market environments.
* Both Historical and Adaptive models achieve statistically acceptable backtest performance.

---

## Visualizations

### Monte Carlo Loss Distribution

![Monte Carlo Loss Distribution](outputs/mc_loss_distribution.png)

### Volatility Regime Detection

![Volatility Regime Detection](outputs/detected_volatility_regimes.png)

### Risk Contribution Analysis

![Risk Contribution Analysis](outputs/risk_contribution.png)

### Stress Testing

![Stress Testing](outputs/stress_scenarios.png)

### VaR and CVaR Model Comparison

![VaR and CVaR Comparison](outputs/var_cvar_model_comparison.png)

### Historical vs Adaptive Regime-Weighted Backtest

![Historical vs Adaptive Backtest](outputs/historical_vs_regime_weighted_backtest.png)

---

## Installation

```bash
pip install -r requirements.txt

python run_project.py
```

---

## Technologies

* Python
* NumPy
* Pandas
* SciPy
* Matplotlib
* Monte Carlo Simulation
* Statistical Modelling
* Portfolio Risk Management
* Quantitative Finance

---

## Future Work

Potential extensions include:

* GARCH Volatility Forecasting
* Filtered Historical Simulation
* Hidden Markov Regime Detection
* CVaR Portfolio Optimization
* Risk Parity Construction
* Liquidity-Adjusted VaR
* Factor-Based Risk Models
* Machine Learning Based Volatility Forecasting
