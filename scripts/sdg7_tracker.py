"""
sdg7_tracker.py
---------------
Core methodology for tracking SDG 7 (Affordable and Clean Energy) progress
using VIIRS nighttime light as an electrification proxy.

Implements:
  - NTL threshold-based electrification classification (tile level)
  - Population-weighted access rate (SDG 7.1.1 metric)
  - Theil-Sen projection to 2030 with bootstrap 95% CI
  - SDG 7.1.1 gap analysis and on-track / at-risk / off-track classification
  - Composite Electrification Index (access rate + spatial equity)
  - Urban / rural disaggregation

All heavy computation (gini, pop_weighted_ntl, mann_kendall) is delegated
to the existing inequality.py and temporal_analysis.py modules.

Author: Bouchra Daddaoui
"""

from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import theilslopes, linregress

from inequality import gini, pop_weighted_ntl, energy_poverty_share
from temporal_analysis import mann_kendall

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SDG7_UNIVERSAL_TARGET: float = 0.95   # SDG 7.1.1 — 95% = "universal access"
NTL_ELECTRIFICATION_THRESHOLD: float = 2.0  # nW/cm²/sr — consistent with data_utils default
NTL_URBAN_PROXY: float = 10.0         # tiles above this treated as urban proxy

COUNTRY_COLORS = {
    "Brazil":  "#009c3b",
    "China":   "#de2910",
    "Morocco": "#c1272d",
}


# ---------------------------------------------------------------------------
# 1. NTL threshold classification
# ---------------------------------------------------------------------------

def classify_electrification(
    ntl_values: np.ndarray,
    threshold: float = NTL_ELECTRIFICATION_THRESHOLD,
    method: str = "fixed",
    percentile: float = 25.0,
) -> np.ndarray:
    """
    Binary electrification classification per tile.

    Parameters
    ----------
    ntl_values : 1-D NTL radiance array.
    threshold  : Fixed threshold (nW/cm²/sr) used when method='fixed'.
    method     : 'fixed' | 'otsu' | 'percentile'
    percentile : Cut-off percentile when method='percentile'.

    Returns
    -------
    Binary ndarray (1 = electrified, 0 = unelectrified).
    """
    v = np.asarray(ntl_values, dtype=float)
    if method == "otsu":
        try:
            from skimage.filters import threshold_otsu
            t = threshold_otsu(np.log1p(v[np.isfinite(v)]))
            return (np.log1p(v) >= t).astype(int)
        except ImportError:
            warnings.warn("skimage not installed — falling back to fixed threshold.")
            return (v >= threshold).astype(int)
    elif method == "percentile":
        t = np.nanpercentile(v, percentile)
        return (v >= t).astype(int)
    else:  # "fixed" — reproduces data_utils.build_analysis_gdf(ntl_threshold=2.0)
        return (v >= threshold).astype(int)


# ---------------------------------------------------------------------------
# 2. Population-weighted access rate  (SDG 7.1.1)
# ---------------------------------------------------------------------------

def population_weighted_access_rate(
    ntl_values: np.ndarray,
    pop_values: np.ndarray,
    threshold: float = NTL_ELECTRIFICATION_THRESHOLD,
    method: str = "fixed",
) -> float:
    """
    Population-weighted electricity access rate — SDG 7.1.1 metric.

    Formula:  sum(pop_i × electrified_i) / sum(pop_i)

    Returns value in [0, 1].
    """
    pop = np.asarray(pop_values, dtype=float)
    electrified = classify_electrification(ntl_values, threshold, method)
    valid = np.isfinite(pop) & (pop > 0)
    if valid.sum() == 0:
        return np.nan
    return float(np.sum(pop[valid] * electrified[valid]) / np.sum(pop[valid]))


def access_rate_timeseries(
    panel_df: pd.DataFrame,
    ntl_col: str = "ntl_mean",
    pop_col: str = "pop_density",
    year_col: str = "year",
    country_col: str = "country",
    threshold: float = NTL_ELECTRIFICATION_THRESHOLD,
    method: str = "fixed",
) -> pd.DataFrame:
    """
    Compute population_weighted_access_rate for every (country, year) pair.

    Returns long-format DataFrame:
        [country, year, access_rate, n_tiles, pop_total, ntl_mean_pw]
    """
    rows = []
    for (country, year), grp in panel_df.groupby([country_col, year_col]):
        ntl = grp[ntl_col].values
        pop = grp[pop_col].values if pop_col in grp.columns else np.ones(len(grp))
        rate = population_weighted_access_rate(ntl, pop, threshold, method)
        rows.append({
            "country":     country,
            "year":        int(year),
            "access_rate": rate,
            "n_tiles":     len(grp),
            "pop_total":   float(pop[np.isfinite(pop)].sum()),
            "ntl_mean_pw": float(pop_weighted_ntl(ntl, pop)),
        })
    return pd.DataFrame(rows).sort_values(["country", "year"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# 3. Theil-Sen projection with bootstrap 95% CI
# ---------------------------------------------------------------------------

def theil_sen_projection(
    years: np.ndarray,
    values: np.ndarray,
    target_year: int = 2030,
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    random_state: int = 42,
) -> dict:
    """
    Fit Theil-Sen slope on (years, values), project to target_year.
    Bootstrap resampling provides uncertainty bands.

    Parameters
    ----------
    years        : 1-D year array (e.g. 2015–2023).
    values       : 1-D access rate or NTL array corresponding to years.
    target_year  : Year to project to (default 2030).
    n_bootstrap  : Number of bootstrap resamples for CI.
    ci           : Confidence level (default 0.95).
    random_state : RNG seed.

    Returns
    -------
    dict with keys:
        slope, intercept, projected_value, ci_lower, ci_upper,
        proj_years, proj_values, proj_lower, proj_upper,
        p_value, trend
    """
    years = np.asarray(years, dtype=float)
    values = np.asarray(values, dtype=float)

    # Point estimate
    res = theilslopes(values, years)
    slope, intercept = float(res[0]), float(res[1])

    proj_years = np.arange(int(years[-1]) + 1, target_year + 1, dtype=float)
    all_years = np.concatenate([years, proj_years])
    proj_values_full = intercept + slope * all_years
    proj_values = intercept + slope * proj_years
    projected_value = float(intercept + slope * target_year)

    # Bootstrap CI
    rng = np.random.default_rng(random_state)
    n = len(years)
    boot_proj = np.zeros((n_bootstrap, len(proj_years)))
    boot_proj_at_target = np.zeros(n_bootstrap)
    for b in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        yr_b, val_b = years[idx], values[idx]
        try:
            r_b = theilslopes(val_b, yr_b)
            s_b, i_b = float(r_b[0]), float(r_b[1])
        except Exception:
            s_b, i_b = slope, intercept
        boot_proj[b] = i_b + s_b * proj_years
        boot_proj_at_target[b] = i_b + s_b * target_year

    alpha = (1.0 - ci) / 2.0
    ci_lower = float(np.percentile(boot_proj_at_target, 100 * alpha))
    ci_upper = float(np.percentile(boot_proj_at_target, 100 * (1 - alpha)))
    proj_lower = np.percentile(boot_proj, 100 * alpha, axis=0)
    proj_upper = np.percentile(boot_proj, 100 * (1 - alpha), axis=0)

    # Mann-Kendall for p-value
    mk = mann_kendall(values)

    return {
        "slope":            slope,
        "intercept":        intercept,
        "projected_value":  projected_value,
        "ci_lower":         ci_lower,
        "ci_upper":         ci_upper,
        "proj_years":       proj_years,
        "proj_values":      proj_values,
        "proj_lower":       proj_lower,
        "proj_upper":       proj_upper,
        "obs_years":        years,
        "obs_values":       values,
        "p_value":          mk["p_value"],
        "trend":            mk["trend"],
        "annual_slope_pp":  slope * 100,  # in percentage points per year
    }


# ---------------------------------------------------------------------------
# 4. SDG 7.1.1 gap analysis
# ---------------------------------------------------------------------------

def sdg7_gap_analysis(
    access_rate_df: pd.DataFrame,
    country_col: str = "country",
    year_col: str = "year",
    rate_col: str = "access_rate",
    target_year: int = 2030,
    universal_target: float = SDG7_UNIVERSAL_TARGET,
    n_bootstrap: int = 1000,
) -> pd.DataFrame:
    """
    Project access rate to 2030, classify status, compute gap to SDG target.

    Status logic:
        'On track'  : projected_2030 >= target AND ci_lower >= target - 0.05
        'At risk'   : projected_2030 >= target BUT ci_lower < target - 0.05
        'Off track' : projected_2030 < target

    Returns publishable Table DataFrame with columns:
        country, access_2023, projected_2030, ci_lower, ci_upper,
        gap_to_target_pp, annual_slope_pp, years_to_target, status
    """
    rows = []
    for country, grp in access_rate_df.groupby(country_col):
        grp = grp.sort_values(year_col)
        yrs = grp[year_col].values.astype(float)
        rates = grp[rate_col].values

        proj = theil_sen_projection(
            yrs, rates, target_year=target_year, n_bootstrap=n_bootstrap
        )

        access_last = float(rates[-1])
        proj_2030 = min(proj["projected_value"], 1.0)
        ci_lo = min(proj["ci_lower"], 1.0)
        ci_hi = min(proj["ci_upper"], 1.0)
        slope_pp = proj["annual_slope_pp"]
        gap_pp = (universal_target - proj_2030) * 100

        if slope_pp > 0:
            years_to_target = (universal_target - access_last) / (slope_pp / 100)
        else:
            years_to_target = float("inf")

        if proj_2030 >= universal_target and ci_lo >= universal_target - 0.05:
            status = "On track"
        elif proj_2030 >= universal_target:
            status = "At risk"
        else:
            status = "Off track"

        rows.append({
            "Country":             country,
            "Access 2023 (%)":     round(access_last * 100, 1),
            "Projected 2030 (%)":  round(proj_2030 * 100, 1),
            "CI Lower (%)":        round(ci_lo * 100, 1),
            "CI Upper (%)":        round(ci_hi * 100, 1),
            "Gap to Target (pp)":  round(max(gap_pp, 0), 1),
            "Annual Slope (pp/yr)": round(slope_pp, 2),
            "Years to Target":     round(years_to_target, 1) if np.isfinite(years_to_target) else ">20",
            "Status":              status,
        })

    return pd.DataFrame(rows).sort_values("Country").reset_index(drop=True)


# ---------------------------------------------------------------------------
# 5. Composite Electrification Index
# ---------------------------------------------------------------------------

def electrification_index(
    ntl_values: np.ndarray,
    pop_values: np.ndarray,
    threshold: float = NTL_ELECTRIFICATION_THRESHOLD,
) -> float:
    """
    Composite Electrification Index [0, 1].

    EI = 0.5 × access_rate + 0.5 × (1 - gini)

    Penalises high access combined with high spatial inequality:
    a country with 90% access but Gini=0.8 ranks lower than one with
    85% access and Gini=0.2 — reflecting uneven distribution.
    """
    rate = population_weighted_access_rate(ntl_values, pop_values, threshold)
    g = gini(ntl_values)
    if np.isnan(g):
        return rate
    equity = 1.0 - g
    return float(0.5 * rate + 0.5 * equity)


def electrification_index_timeseries(
    panel_df: pd.DataFrame,
    ntl_col: str = "ntl_mean",
    pop_col: str = "pop_density",
    year_col: str = "year",
    country_col: str = "country",
    threshold: float = NTL_ELECTRIFICATION_THRESHOLD,
) -> pd.DataFrame:
    """
    Compute Electrification Index for every (country, year).

    Returns DataFrame: [country, year, EI, access_rate, gini_ntl]
    """
    rows = []
    for (country, year), grp in panel_df.groupby([country_col, year_col]):
        ntl = grp[ntl_col].values
        pop = grp[pop_col].values if pop_col in grp.columns else np.ones(len(grp))
        rate = population_weighted_access_rate(ntl, pop, threshold)
        g = gini(ntl)
        ei = float(0.5 * rate + 0.5 * (1.0 - g)) if not np.isnan(g) else rate
        rows.append({
            "country":    country,
            "year":       int(year),
            "EI":         round(ei, 4),
            "access_rate": round(rate, 4),
            "gini_ntl":   round(g, 4),
        })
    return pd.DataFrame(rows).sort_values(["country", "year"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# 6. Urban / rural disaggregation
# ---------------------------------------------------------------------------

def urban_rural_disaggregation(
    gdf: gpd.GeoDataFrame,
    ntl_col: str = "ntl_mean",
    pop_col: str = "pop_density",
    urban_ntl_threshold: float = NTL_URBAN_PROXY,
    electrification_threshold: float = NTL_ELECTRIFICATION_THRESHOLD,
) -> pd.DataFrame:
    """
    Split tiles into urban / rural strata using NTL as proxy.
    Compute access rate and Gini for each stratum.

    Note: Using NTL as urban proxy is an acknowledged limitation (see notebook 12).
    When GHSL 'built_area_m2' is present, use that instead.

    Returns DataFrame: [stratum, n_tiles, pop_share, access_rate, mean_ntl, gini_ntl]
    """
    gdf = gdf.copy()
    pop = gdf[pop_col].values if pop_col in gdf.columns else np.ones(len(gdf))
    ntl = gdf[ntl_col].values

    if "built_area_m2" in gdf.columns:
        urban_mask = gdf["built_area_m2"] > gdf["built_area_m2"].median()
    else:
        urban_mask = ntl >= urban_ntl_threshold

    rows = []
    for label, mask in [("Urban", urban_mask), ("Rural", ~urban_mask)]:
        ntl_s = ntl[mask]
        pop_s = pop[mask]
        if len(ntl_s) == 0:
            continue
        rate = population_weighted_access_rate(ntl_s, pop_s, electrification_threshold)
        rows.append({
            "Stratum":        label,
            "N Tiles":        int(mask.sum()),
            "Pop Share (%)":  round(100 * pop_s[np.isfinite(pop_s)].sum() / pop[np.isfinite(pop)].sum(), 1),
            "Access Rate (%)": round(rate * 100, 1),
            "Mean NTL":       round(ntl_s.mean(), 2),
            "Gini":           round(gini(ntl_s), 3),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 7. Visualization functions
# ---------------------------------------------------------------------------

def plot_access_rate_timeseries(
    access_df: pd.DataFrame,
    projection_results: dict,
    official_stats: dict | None = None,
    target_line: float = SDG7_UNIVERSAL_TARGET,
    save_path: str | None = None,
) -> plt.Figure:
    """
    Publication Figure: 3-panel SDG 7.1.1 progress chart.

    Each panel: observed access rate (markers), Theil-Sen trend line,
    95% CI ribbon, 2030 projection (dashed), SDG target horizontal line.
    Official statistics overlaid as X markers if provided.

    Parameters
    ----------
    access_df          : Output of access_rate_timeseries().
    projection_results : dict[country] → theil_sen_projection() output.
    official_stats     : dict[country][year] → float (optional overlay).
    target_line        : SDG 7.1.1 target rate (default 0.95).
    save_path          : Optional PNG save path.
    """
    countries = sorted(access_df["country"].unique())
    fig, axes = plt.subplots(1, len(countries), figsize=(6 * len(countries), 5), sharey=False)
    if len(countries) == 1:
        axes = [axes]

    for ax, country in zip(axes, countries):
        col = COUNTRY_COLORS.get(country, "steelblue")
        grp = access_df[access_df["country"] == country].sort_values("year")
        yrs = grp["year"].values.astype(float)
        rates = grp["access_rate"].values

        # Observed
        ax.plot(yrs, rates * 100, "o-", color=col, linewidth=2.2,
                markersize=6, label="Observed (NTL model)", zorder=4)

        # Projection + CI
        if country in projection_results:
            p = projection_results[country]
            proj_yrs = p["proj_years"]
            # Connect last observed to first projected
            connect_yrs = np.concatenate([[yrs[-1]], proj_yrs])
            connect_vals = np.concatenate([[rates[-1] * 100],
                                            np.clip(p["proj_values"], 0, 1) * 100])
            ax.plot(connect_yrs, connect_vals, "--", color=col,
                    linewidth=1.8, alpha=0.8, label="Projected (Theil-Sen)", zorder=3)
            ax.fill_between(
                proj_yrs,
                np.clip(p["proj_lower"], 0, 1) * 100,
                np.clip(p["proj_upper"], 0, 1) * 100,
                color=col, alpha=0.15, label=f"95% CI",
            )
            ax.scatter([2030], [min(p["projected_value"], 1.0) * 100],
                       marker="D", s=60, color=col, zorder=5)

        # Official stats overlay
        if official_stats and country in official_stats:
            off_yrs = sorted(official_stats[country].keys())
            off_vals = [official_stats[country][y] * 100 for y in off_yrs]
            ax.scatter(off_yrs, off_vals, marker="X", s=70, color="black",
                       zorder=6, label="Official (WB/IEA)", linewidths=1.5)

        # SDG target line
        ax.axhline(target_line * 100, color="gold", linewidth=1.8,
                   linestyle=":", label=f"SDG 7.1.1 target ({target_line*100:.0f}%)")
        ax.axvline(2030, color="grey", linewidth=0.8, linestyle=":")

        ax.set_title(country, fontsize=12, fontweight="bold")
        ax.set_xlabel("Year", fontsize=10)
        ax.set_ylabel("Population-Weighted Access Rate (%)", fontsize=9)
        ax.set_ylim(max(0, rates.min() * 100 - 5), 105)
        ax.legend(fontsize=7.5, loc="lower right")
        ax.grid(True, alpha=0.25)
        ax.set_xticks(list(yrs) + [2030])
        ax.set_xticklabels([str(int(y)) for y in yrs] + ["2030"],
                           rotation=45, ha="right", fontsize=8)

    fig.suptitle(
        "SDG 7.1.1 Progress: Population-Weighted Electricity Access Rate\n"
        "VIIRS NTL Model — 2015–2030 Projection (Theil-Sen + Bootstrap 95% CI)",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=180, bbox_inches="tight")
        print(f"Saved: {save_path}")
    return fig


def plot_sdg7_status_dashboard(
    gap_df: pd.DataFrame,
    save_path: str | None = None,
) -> plt.Figure:
    """
    Country comparison: projected 2030 access rate with CI error bars.
    Colour-coded by SDG 7 status.
    """
    status_colors = {"On track": "#2ca02c", "At risk": "#ff7f0e", "Off track": "#d62728"}
    fig, ax = plt.subplots(figsize=(8, 5))

    for _, row in gap_df.iterrows():
        c = row["Country"]
        proj = row["Projected 2030 (%)"]
        ci_lo = row["CI Lower (%)"]
        ci_hi = row["CI Upper (%)"]
        color = status_colors.get(row["Status"], "grey")
        ax.bar(c, proj, color=color, alpha=0.82, edgecolor="black", linewidth=0.8, width=0.5)
        ax.errorbar(c, proj, yerr=[[proj - ci_lo], [ci_hi - proj]],
                    fmt="none", color="black", capsize=6, linewidth=1.5)
        ax.text(c, ci_hi + 0.8, f"{proj:.1f}%", ha="center", va="bottom",
                fontsize=10, fontweight="bold")

    ax.axhline(SDG7_UNIVERSAL_TARGET * 100, color="gold", linewidth=2,
               linestyle="--", label=f"SDG 7 target ({SDG7_UNIVERSAL_TARGET*100:.0f}%)")

    legend_patches = [mpatches.Patch(color=v, label=k) for k, v in status_colors.items()]
    ax.legend(handles=legend_patches + [
        plt.Line2D([0], [0], color="gold", linestyle="--", linewidth=2, label="SDG 7 target")
    ], fontsize=9, loc="lower right")

    ax.set_ylabel("Projected 2030 Access Rate (%)", fontsize=11)
    ax.set_ylim(0, 115)
    ax.set_title("SDG 7.1.1 Attainment Projection — 2030\nCountry Status Classification",
                 fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.25, axis="y")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=180, bbox_inches="tight")
        print(f"Saved: {save_path}")
    return fig


def plot_country_comparison(
    ei_df: pd.DataFrame,
    save_path: str | None = None,
) -> plt.Figure:
    """
    Dual-axis: Electrification Index (left) + Gini (right, inverted).
    Shows that improving access WITHOUT reducing inequality is insufficient.
    """
    countries = sorted(ei_df["country"].unique())
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax2 = ax1.twinx()

    for country in countries:
        col = COUNTRY_COLORS.get(country, "steelblue")
        grp = ei_df[ei_df["country"] == country].sort_values("year")
        yrs = grp["year"].values
        ax1.plot(yrs, grp["EI"].values, "o-", color=col, linewidth=2.2,
                 markersize=5, label=f"{country} EI")
        ax2.plot(yrs, grp["gini_ntl"].values, "s--", color=col, linewidth=1.4,
                 markersize=4, alpha=0.6, label=f"{country} Gini")

    ax1.set_xlabel("Year", fontsize=10)
    ax1.set_ylabel("Composite Electrification Index (EI)", fontsize=10, color="black")
    ax2.set_ylabel("Gini Coefficient (NTL)", fontsize=10, color="grey")
    ax2.invert_yaxis()  # lower Gini = better → show rising = good
    ax1.set_title("Composite Electrification Index vs Spatial Inequality (Gini)\n"
                  "EI = 0.5 × Access Rate + 0.5 × (1 − Gini)",
                  fontsize=11, fontweight="bold")
    ax1.legend(loc="upper left", fontsize=8)
    ax2.legend(loc="upper right", fontsize=8)
    ax1.grid(True, alpha=0.25)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=180, bbox_inches="tight")
        print(f"Saved: {save_path}")
    return fig
