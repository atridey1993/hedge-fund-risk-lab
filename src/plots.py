from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def save_return_distribution(mc_returns: pd.Series, var: float, cvar: float, path: Path):
    losses = -mc_returns
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(losses, bins=80, alpha=0.75, density=True)
    ax.axvline(var, linestyle="--", linewidth=2, label=f"VaR = {var:.2%}")
    ax.axvline(cvar, linestyle="-.", linewidth=2, label=f"CVaR = {cvar:.2%}")
    ax.set_title("Monte Carlo loss distribution with VaR and CVaR")
    ax.set_xlabel("Portfolio loss")
    ax.set_ylabel("Density")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_model_comparison(df: pd.DataFrame, path: Path):
    plot_df = df.copy()
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(plot_df))
    width = 0.35
    ax.bar(x - width/2, plot_df["VaR"], width, label="VaR")
    ax.bar(x + width/2, plot_df["CVaR"].fillna(0), width, label="CVaR")
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["Model"], rotation=25, ha="right")
    ax.set_ylabel("Loss")
    ax.set_title("Model risk: VaR/CVaR comparison")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_backtest_plot(backtest: pd.DataFrame, path: Path, title: str):
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(backtest.index, backtest["loss"], linewidth=1, label="Realized loss")
    ax.plot(backtest.index, backtest["VaR"], linewidth=2, label="Rolling VaR")
    exceptions = backtest[backtest["exception"] == 1]
    ax.scatter(exceptions.index, exceptions["loss"], s=28, label="VaR exception")
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Loss")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_backtest_comparison(hist_bt: pd.DataFrame, regime_bt: pd.DataFrame, path: Path):
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(hist_bt.index, hist_bt["VaR"], linewidth=2, label="Historical VaR")
    ax.plot(regime_bt.index, regime_bt["VaR"], linewidth=2, label="Regime-weighted VaR")
    ax.plot(hist_bt.index, hist_bt["loss"], linewidth=0.8, alpha=0.5, label="Realized loss")
    ax.set_title("Standard historical VaR vs adaptive regime-weighted VaR")
    ax.set_xlabel("Date")
    ax.set_ylabel("Loss")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_risk_contribution_plot(rc: pd.DataFrame, path: Path):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(rc["asset"], 100 * rc["risk_contribution_pct"])
    ax.set_xlabel("Risk contribution (%)")
    ax.set_title("Portfolio risk contribution by asset")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_stress_plot(stress: pd.DataFrame, path: Path):
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(stress["scenario"], 100 * stress["portfolio_return"])
    ax.set_xlabel("Portfolio return under scenario (%)")
    ax.set_title("Stress testing: scenario P&L")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_regime_plot(classified: pd.DataFrame, path: Path):
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(classified.index, classified["rolling_vol"], linewidth=1.5)
    ax.set_title("Detected volatility regimes using rolling volatility")
    ax.set_xlabel("Date")
    ax.set_ylabel("Rolling volatility")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_regime_var_comparison(compare_df: pd.DataFrame, path: Path):
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(compare_df))
    width = 0.35
    ax.bar(x - width/2, compare_df["VaR"], width, label="VaR")
    ax.bar(x + width/2, compare_df["CVaR"], width, label="CVaR")
    ax.set_xticks(x)
    ax.set_xticklabels(compare_df["Model"], rotation=15, ha="right")
    ax.set_title("Equal-weight historical vs adaptive regime-weighted tail risk")
    ax.set_ylabel("Loss")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
