"""
temporal_analysis.py
--------------------
Time-series analysis of VIIRS nighttime light trends.

Provides:
  - Mann-Kendall trend test (non-parametric)
  - Theil-Sen slope estimator
  - CUSUM change-point detection
  - Panel-level trend summaries and visualisations

Author: Bouchra Daddaoui
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from typing import Optional, Tuple


# ---------------------------------------------------------------------------
# Mann-Kendall trend test
# ---------------------------------------------------------------------------

def mann_kendall(series: np.ndarray) -> dict:
    """
    Non-parametric Mann-Kendall trend test.

    Parameters
    ----------
    series : 1-D time-ordered array of observations.

    Returns
    -------
    dict with keys:
        S        : MK statistic
        tau      : Kendall's tau
        p_value  : Two-tailed p-value (normal approximation)
        trend    : "increasing" | "decreasing" | "no trend"
        slope    : Theil-Sen slope estimate (units per time step)
        intercept: Theil-Sen intercept
    """
    n = len(series)
    # S statistic
    s = 0
    for i in range(n - 1):
        for j in range(i + 1, n):
            diff = series[j] - series[i]
            if diff > 0:
                s += 1
            elif diff < 0:
                s -= 1

    # Variance (no ties correction for simplicity)
    var_s = n * (n - 1) * (2 * n + 5) / 18.0

    if s > 0:
        z = (s - 1) / np.sqrt(var_s)
    elif s < 0:
        z = (s + 1) / np.sqrt(var_s)
    else:
        z = 0.0

    from scipy import stats
    p_value = 2.0 * (1.0 - stats.norm.cdf(abs(z)))
    tau = s / (0.5 * n * (n - 1))

    # Theil-Sen slope
    slopes = []
    for i in range(n):
        for j in range(i + 1, n):
            slopes.append((series[j] - series[i]) / (j - i))
    slope = float(np.median(slopes))
    intercept = float(np.median(series) - slope * np.median(np.arange(n)))

    if p_value < 0.05:
        trend = "increasing" if s > 0 else "decreasing"
    else:
        trend = "no trend"

    return {
        "S": s,
        "tau": tau,
        "z": z,
        "p_value": p_value,
        "trend": trend,
        "slope": slope,
        "intercept": intercept,
    }


def panel_trend_summary(
    df: pd.DataFrame,
    country_col: str = "country",
    year_col: str = "year",
    ntl_col: str = "ntl_mean",
) -> pd.DataFrame:
    """
    Run Mann-Kendall for each country in the panel.

    Parameters
    ----------
    df          : Long-format panel DataFrame.
    country_col : Column identifying countries.
    year_col    : Column identifying year.
    ntl_col     : Column with NTL values.

    Returns
    -------
    DataFrame indexed by country with MK statistics.
    """
    rows = []
    for country, grp in df.groupby(country_col):
        grp_sorted = grp.sort_values(year_col)
        result = mann_kendall(grp_sorted[ntl_col].values)
        result[country_col] = country
        rows.append(result)
    return pd.DataFrame(rows).set_index(country_col)


# ---------------------------------------------------------------------------
# CUSUM change-point detection
# ---------------------------------------------------------------------------

def cusum_changepoint(
    series: np.ndarray,
    threshold: Optional[float] = None,
) -> dict:
    """
    CUSUM (Cumulative Sum) change-point detection.

    Detects the time index of the most likely structural break.

    Parameters
    ----------
    series    : 1-D time-ordered array.
    threshold : Detection threshold (defaults to 1 × std of series).

    Returns
    -------
    dict with keys:
        changepoint_idx  : Index of detected change point (or None).
        cusum_pos        : Positive CUSUM series.
        cusum_neg        : Negative CUSUM series (absolute values).
        mean_pre         : Mean before change point.
        mean_post        : Mean after change point.
    """
    if threshold is None:
        threshold = series.std()

    mu = series.mean()
    cusum_pos = np.zeros(len(series))
    cusum_neg = np.zeros(len(series))

    for i in range(1, len(series)):
        cusum_pos[i] = max(0.0, cusum_pos[i - 1] + (series[i] - mu) - threshold / 2)
        cusum_neg[i] = max(0.0, cusum_neg[i - 1] - (series[i] - mu) - threshold / 2)

    cp_pos = int(np.argmax(cusum_pos)) if cusum_pos.max() > threshold else None
    cp_neg = int(np.argmax(cusum_neg)) if cusum_neg.max() > threshold else None

    # Pick the stronger signal
    changepoint = None
    if cp_pos is not None and cp_neg is not None:
        changepoint = cp_pos if cusum_pos[cp_pos] >= cusum_neg[cp_neg] else cp_neg
    elif cp_pos is not None:
        changepoint = cp_pos
    elif cp_neg is not None:
        changepoint = cp_neg

    mean_pre = float(series[:changepoint].mean()) if changepoint else float(series.mean())
    mean_post = float(series[changepoint:].mean()) if changepoint else float(series.mean())

    return {
        "changepoint_idx": changepoint,
        "cusum_pos": cusum_pos,
        "cusum_neg": cusum_neg,
        "mean_pre": mean_pre,
        "mean_post": mean_post,
    }


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def plot_ntl_trends(
    df: pd.DataFrame,
    country_col: str = "country",
    year_col: str = "year",
    ntl_col: str = "ntl_mean",
    trend_results: Optional[pd.DataFrame] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Multi-panel NTL trend plot with Theil-Sen trend lines.

    Parameters
    ----------
    df             : Long-format panel DataFrame.
    country_col    : Country identifier column.
    year_col       : Year column.
    ntl_col        : NTL values column.
    trend_results  : Output of panel_trend_summary (for adding slope/p annotations).
    save_path      : Optional save path.

    Returns
    -------
    fig : Matplotlib Figure.
    """
    countries = sorted(df[country_col].unique())
    n = len(countries)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), sharey=False)
    if n == 1:
        axes = [axes]

    colors = plt.cm.Set2(np.linspace(0, 0.8, n))

    for ax, country, color in zip(axes, countries, colors):
        grp = df[df[country_col] == country].sort_values(year_col)
        years = grp[year_col].values
        ntl = grp[ntl_col].values

        ax.plot(years, ntl, "o-", color=color, linewidth=2, markersize=4, label=country)

        # Theil-Sen trend line
        if trend_results is not None and country in trend_results.index:
            slope = trend_results.loc[country, "slope"]
            intercept = trend_results.loc[country, "intercept"]
            t_idx = np.arange(len(years))
            ax.plot(years, intercept + slope * t_idx, "--", color="black",
                    linewidth=1.2, alpha=0.7, label=f"Theil-Sen (β={slope:.3f})")
            p = trend_results.loc[country, "p_value"]
            trend = trend_results.loc[country, "trend"]
            ax.set_title(f"{country}\n{trend.capitalize()}  (p={p:.3f})", fontsize=11)
        else:
            ax.set_title(country, fontsize=11)

        ax.set_xlabel("Year")
        ax.set_ylabel("Mean NTL (nW/cm²/sr)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle("VIIRS Nighttime Light Trends by Country (2014–2023)",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_cusum(
    series: np.ndarray,
    years: np.ndarray,
    cusum_result: dict,
    country: str = "",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    CUSUM chart with change-point annotation.

    Parameters
    ----------
    series       : Raw NTL time series.
    years        : Corresponding year values.
    cusum_result : Output of cusum_changepoint().
    country      : Country label for title.
    save_path    : Optional save path.

    Returns
    -------
    fig : Matplotlib Figure.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    ax1.plot(years, series, "o-", color="#2c7bb6", linewidth=2)
    ax1.set_ylabel("Mean NTL (nW/cm²/sr)")
    ax1.set_title(f"CUSUM Change-Point Analysis — {country}", fontsize=11, fontweight="bold")

    ax2.plot(years, cusum_result["cusum_pos"], "-", color="#d7191c", label="CUSUM+")
    ax2.plot(years, cusum_result["cusum_neg"], "-", color="#1a9641", label="CUSUM−")
    ax2.set_ylabel("CUSUM statistic")
    ax2.set_xlabel("Year")
    ax2.legend()

    cp = cusum_result["changepoint_idx"]
    if cp is not None:
        for ax in (ax1, ax2):
            ax.axvline(years[cp], color="orange", linestyle="--", linewidth=1.5,
                       label=f"Change point ({years[cp]})")
        ax1.legend(fontsize=8)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
