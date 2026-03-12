"""
inequality.py
-------------
Spatial inequality metrics for VIIRS nighttime lights + SDG 7 analysis.

Covers:
  - Gini coefficient and Lorenz curve
  - Theil T index (decomposes into within vs between country inequality)
  - Population-weighted NTL (electrification access proxy)
  - SDG 7 gap: distance to universal electricity access by 2030
  - Energy poverty mapping: high-pop / low-NTL tiles
"""

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import linregress


# ── Gini coefficient ──────────────────────────────────────────────────────────

def gini(values):
    """
    Gini coefficient for a 1-D array. Returns a value in [0, 1].
    0 = perfect equality, 1 = maximum concentration.
    """
    v = np.sort(np.asarray(values, dtype=float))
    v = v[v >= 0]  # drop negatives (NTL shouldn't have any but just in case)
    n = len(v)
    if n == 0 or v.sum() == 0:
        return np.nan
    idx = np.arange(1, n + 1)
    return (2 * (idx * v).sum() / (n * v.sum())) - (n + 1) / n


def lorenz_curve(values):
    """Return (population_share, ntl_share) arrays for the Lorenz curve."""
    v = np.sort(np.asarray(values, dtype=float))
    v = np.maximum(v, 0)
    cum_ntl = np.cumsum(v) / v.sum()
    cum_pop = np.arange(1, len(v) + 1) / len(v)
    # Prepend origin
    return np.concatenate([[0], cum_pop]), np.concatenate([[0], cum_ntl])


def plot_lorenz(country_series, save_path=None):
    """
    Overlay Lorenz curves for multiple countries.
    country_series: dict mapping country name → NTL array.
    """
    colors = plt.cm.Set1(np.linspace(0, 0.8, len(country_series)))
    fig, ax = plt.subplots(figsize=(6, 6))

    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Perfect equality')
    for (country, vals), color in zip(country_series.items(), colors):
        pop_share, ntl_share = lorenz_curve(vals)
        g = gini(vals)
        ax.plot(pop_share, ntl_share, linewidth=2, color=color,
                label=f'{country}  (Gini = {g:.3f})')
        ax.fill_between(pop_share, pop_share, ntl_share, alpha=0.08, color=color)

    ax.set_xlabel('Cumulative share of tiles (poorest → richest)')
    ax.set_ylabel('Cumulative share of NTL radiance')
    ax.set_title('Lorenz Curves — NTL Electrification Inequality', fontweight='bold')
    ax.legend(fontsize=9)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


# ── Theil T index ─────────────────────────────────────────────────────────────

def theil_T(values):
    """
    Theil T index — entropy-based inequality measure.
    More sensitive to top-end concentration than Gini.
    """
    v = np.asarray(values, dtype=float)
    v = v[v > 0]
    mu = v.mean()
    return float(np.mean((v / mu) * np.log(v / mu)))


def theil_decomposition(df, country_col, ntl_col):
    """
    Decompose Theil T into within-country and between-country components.

    Returns a dict with total, within, and between Theil T values.
    This shows whether NTL inequality is mainly a within-country or
    cross-country phenomenon.
    """
    groups = df.groupby(country_col)[ntl_col].apply(np.array)
    n = len(df)
    mu_total = df[ntl_col].mean()

    # Between-country component
    between = 0.0
    for country, vals in groups.items():
        n_k = len(vals)
        mu_k = vals.mean()
        if mu_k > 0:
            between += (n_k / n) * (mu_k / mu_total) * np.log(mu_k / mu_total)

    # Within-country component
    within = 0.0
    for country, vals in groups.items():
        n_k = len(vals)
        mu_k = vals.mean()
        if mu_k > 0 and n_k > 1:
            t_k = theil_T(vals)
            within += (n_k / n) * (mu_k / mu_total) * t_k

    total = within + between
    return {
        'total': total,
        'within': within,
        'between': between,
        'within_pct': 100 * within / total if total > 0 else np.nan,
        'between_pct': 100 * between / total if total > 0 else np.nan,
    }


# ── Population-weighted NTL ───────────────────────────────────────────────────

def pop_weighted_ntl(ntl_values, pop_values):
    """
    Population-weighted mean NTL — better proxy for electricity *access* than
    simple mean, since it accounts for where people actually live.
    """
    pop = np.asarray(pop_values, dtype=float)
    ntl = np.asarray(ntl_values, dtype=float)
    valid = (pop > 0) & np.isfinite(ntl) & np.isfinite(pop)
    if valid.sum() == 0:
        return np.nan
    return float(np.average(ntl[valid], weights=pop[valid]))


def energy_poverty_share(ntl_values, pop_values, ntl_threshold=1.0):
    """
    Fraction of population living in tiles below the NTL threshold.
    ntl_threshold=1.0 nW/cm²/sr is a common proxy for 'unelectrified'.
    """
    pop = np.asarray(pop_values, dtype=float)
    ntl = np.asarray(ntl_values, dtype=float)
    valid = np.isfinite(ntl) & np.isfinite(pop) & (pop > 0)
    poor = valid & (ntl < ntl_threshold)
    if pop[valid].sum() == 0:
        return np.nan
    return float(pop[poor].sum() / pop[valid].sum())


# ── SDG 7 gap analysis ────────────────────────────────────────────────────────

def sdg7_projection(yearly_ntl, years, target_year=2030, baseline_ntl=None):
    """
    Project NTL to 2030 using Theil-Sen linear fit and estimate the gap
    to a user-defined baseline (e.g. current highest-access country).

    Returns projected NTL, years of projection, and the gap at 2030.
    """
    from scipy.stats import theilslopes
    years = np.asarray(years)
    ntl = np.asarray(yearly_ntl)

    res = theilslopes(ntl, years)
    slope, intercept = res[0], res[1]

    proj_years = np.arange(years[-1] + 1, target_year + 1)
    proj_ntl = intercept + slope * proj_years

    ntl_2030 = float(intercept + slope * target_year)
    gap = (baseline_ntl - ntl_2030) if baseline_ntl is not None else None

    return {
        'proj_years': proj_years,
        'proj_ntl': proj_ntl,
        'ntl_2030': ntl_2030,
        'slope': slope,
        'gap_2030': gap,
    }


def plot_sdg7_projections(panel_df, country_col='country', year_col='year',
                          ntl_col='ntl_mean', target_year=2030,
                          sdg_threshold=None, save_path=None):
    """
    Multi-panel NTL trend + 2030 projection per country with SDG 7 target line.

    sdg_threshold: NTL value representing 'universal access' (optional horizontal line).
    """
    countries = sorted(panel_df[country_col].unique())
    colors = {'Brazil': '#009c3b', 'China': '#de2910', 'Morocco': '#c1272d'}
    default_colors = plt.cm.Set2(np.linspace(0, 0.8, len(countries)))

    fig, axes = plt.subplots(1, len(countries), figsize=(6 * len(countries), 5), sharey=False)
    if len(countries) == 1:
        axes = [axes]

    for ax, country in zip(axes, countries):
        grp = panel_df[panel_df[country_col] == country].sort_values(year_col)
        yrs = grp[year_col].values
        ntl = grp[ntl_col].values
        col = colors.get(country, 'steelblue')

        # Historical
        ax.plot(yrs, ntl, 'o-', color=col, linewidth=2.5, markersize=5, label='Observed')

        # Projection
        proj = sdg7_projection(ntl, yrs, target_year=target_year)
        proj_yrs = np.concatenate([[yrs[-1]], proj['proj_years']])
        proj_vals = np.concatenate([[ntl[-1]], proj['proj_ntl']])
        ax.plot(proj_yrs, proj_vals, '--', color=col, linewidth=1.8, alpha=0.7, label='Projected (Theil-Sen)')
        ax.fill_between(proj_yrs, proj_vals, alpha=0.1, color=col)

        # SDG threshold
        if sdg_threshold is not None:
            ax.axhline(sdg_threshold, color='gold', linewidth=1.5, linestyle=':',
                       label=f'SDG 7 proxy ({sdg_threshold})')

        ax.axvline(target_year, color='grey', linewidth=0.8, linestyle=':')
        ax.set_title(f'{country}\nProjected 2030: {proj["ntl_2030"]:.2f}', fontsize=11, fontweight='bold')
        ax.set_xlabel('Year')
        ax.set_ylabel('Mean NTL (nW/cm²/sr)')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(list(yrs) + [target_year])
        ax.set_xticklabels([str(y) for y in yrs] + ['2030'], rotation=45, ha='right')

    fig.suptitle('NTL Trend Projections to 2030 — SDG 7 Analysis', fontsize=13, fontweight='bold')
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


# ── Energy poverty map ────────────────────────────────────────────────────────

def plot_energy_poverty_map(gdf, ntl_col, pop_col,
                            ntl_threshold=1.0, save_path=None):
    """
    Highlight tiles that are both high-population and low-NTL — energy poverty hotspots.

    Category logic:
      - Energy poor: NTL < threshold AND pop > median pop
      - Electrified: NTL >= threshold
      - Low population: NTL < threshold but pop <= median pop
    """
    gdf = gdf.copy()
    pop_med = gdf[pop_col].median()

    def categorise(row):
        if row[ntl_col] >= ntl_threshold:
            return 'Electrified'
        elif row[pop_col] > pop_med:
            return 'Energy poor (high pop, low NTL)'
        else:
            return 'Low pop / low NTL'

    gdf['category'] = gdf.apply(categorise, axis=1)

    palette = {
        'Electrified': '#2c7bb6',
        'Energy poor (high pop, low NTL)': '#d7191c',
        'Low pop / low NTL': '#ffffbf',
    }
    gdf['color'] = gdf['category'].map(palette)

    fig, ax = plt.subplots(figsize=(10, 7))
    gdf.plot(color=gdf['color'], edgecolor='grey', linewidth=0.15, ax=ax)

    patches = [mpatches.Patch(color=c, label=l) for l, c in palette.items()]
    ax.legend(handles=patches, loc='lower right', fontsize=9, framealpha=0.9)
    ax.set_title('Energy Poverty Map\n(NTL threshold = {:.1f} nW/cm²/sr)'.format(ntl_threshold),
                 fontsize=12, fontweight='bold')
    ax.set_axis_off()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


def inequality_summary_table(panel_df, country_col='country',
                              year_col='year', ntl_col='ntl_mean'):
    """
    Year-by-year Gini and Theil T per country. Good for a results table in a paper.
    """
    rows = []
    for (country, year), grp in panel_df.groupby([country_col, year_col]):
        vals = grp[ntl_col].values
        rows.append({
            'Country': country, 'Year': year,
            'Gini': round(gini(vals), 4),
            'Theil T': round(theil_T(vals[vals > 0]) if (vals > 0).any() else np.nan, 4),
            'Mean NTL': round(vals.mean(), 3),
        })
    return pd.DataFrame(rows)
