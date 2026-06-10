# Adaptive Regime-Weighted Tail Risk Model

## Motivation

Standard historical VaR treats every past observation equally.

That is often unrealistic.

A calm market day from two years ago should not have the same importance as a recent stressed market day when estimating today's risk.

## Method

The method has five steps.

### Step 1: Compute rolling volatility

For portfolio returns \(r_t\), compute rolling volatility:

\[
\sigma_t^{roll}
=
\sqrt{
\frac{1}{W-1}
\sum_{i=t-W+1}^{t}
(r_i-\bar r_t)^2
}.
\]

### Step 2: Classify regimes

Use rolling volatility quantiles:

\[
\text{calm}: \sigma_t^{roll} \le q_{low}
\]

\[
\text{normal}: q_{low} < \sigma_t^{roll} < q_{high}
\]

\[
\text{stress}: \sigma_t^{roll} \ge q_{high}.
\]

### Step 3: Assign regime weights

\[
w_t =
\begin{cases}
0.60, & \text{calm},\\
1.00, & \text{normal},\\
2.50, & \text{stress}.
\end{cases}
\]

### Step 4: Add recency weighting

More recent observations receive higher weights:

\[
d_t =
0.5^{\frac{age_t}{h}},
\]

where \(h\) is the half-life.

Final weight:

\[
\tilde w_t = w_t d_t.
\]

### Step 5: Weighted VaR and CVaR

Weighted VaR is the weighted quantile:

\[
\mathrm{VaR}_{\alpha}^{RW}
=
Q_{\alpha}^{weighted}(L_t,\tilde w_t).
\]

Weighted CVaR is:

\[
\mathrm{CVaR}_{\alpha}^{RW}
=
\frac{
\sum_{t:L_t \ge \mathrm{VaR}_{\alpha}^{RW}}
\tilde w_t L_t
}{
\sum_{t:L_t \ge \mathrm{VaR}_{\alpha}^{RW}}
\tilde w_t
}.
\]

## Interview explanation

My extension improves standard historical VaR by making it adaptive. Standard VaR treats all past losses equally. My method detects calm, normal and stressed regimes using rolling volatility and gives higher importance to stressed losses and recent observations. This makes the risk estimate more responsive during volatile markets.

## Honest limitation

This method is not guaranteed to always be statistically superior. It is a practical adaptive heuristic. The correct way to evaluate it is by comparing backtest exceptions, Kupiec p-values, Christoffersen clustering, and stress behaviour against standard VaR.
