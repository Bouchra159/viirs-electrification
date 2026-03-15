"""
gen_figures_11_12.py
--------------------
Generate all figures and tables from notebooks 11 (SDG7 progress tracker)
and 12 (validation against official statistics).

Run from project root:  python scripts/gen_figures_11_12.py
"""

from __future__ import annotations

import sys
import warnings
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from shapely.geometry import box
from scipy import stats

sys.path.insert(0, 'scripts')
warnings.filterwarnings('ignore')

from sdg7_tracker import (
    access_rate_timeseries,
    theil_sen_projection,
    sdg7_gap_analysis,
    electrification_index_timeseries,
    urban_rural_disaggregation,
    plot_access_rate_timeseries,
    plot_sdg7_status_dashboard,
    plot_country_comparison,
    SDG7_UNIVERSAL_TARGET,
    NTL_ELECTRIFICATION_THRESHOLD,
)

FIGURES = Path('figures')
FIGURES.mkdir(exist_ok=True)

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 10,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

# ─────────────────────────────────────────────────────────────────────────────
# Shared synthetic panel data (same seeds as notebooks 07–10)
# ─────────────────────────────────────────────────────────────────────────────

def make_panel():
    """Synthetic VIIRS NTL panel 2015–2023 for 3 countries."""
    YEARS = list(range(2015, 2024))
    COUNTRIES_CFG = [
        dict(country='Brazil',  bbox=(-48,-23,-43,-18), ntl_mean=12.5, ntl_std=8.0,  pop_mean=200, seed=1,  annual_gain=0.35),
        dict(country='China',   bbox=(116,29,122,33),   ntl_mean=28.3, ntl_std=14.0, pop_mean=900, seed=2,  annual_gain=0.15),
        dict(country='Morocco', bbox=(-17.1,20.8,-1.0,35.9), ntl_mean=8.1, ntl_std=5.5, pop_mean=120, seed=3, annual_gain=0.55),
    ]

    records = []
    for cfg in COUNTRIES_CFG:
        rng = np.random.default_rng(cfg['seed'])
        minx, miny, maxx, maxy = cfg['bbox']
        cols = rows = 10
        xs = np.linspace(minx, maxx, cols+1)
        ys = np.linspace(miny, maxy, rows+1)
        geoms = [box(xs[j], ys[i], xs[j+1], ys[i+1]) for i in range(rows) for j in range(cols)]
        n = len(geoms)

        centroids = np.array([[g.centroid.x, g.centroid.y] for g in geoms])
        dists = np.linalg.norm(centroids[:,None] - centroids[None,:], axis=-1)
        cov = cfg['ntl_std']**2 * np.exp(-dists / (0.3*(maxx-minx)))
        base_ntl = rng.multivariate_normal(np.full(n, cfg['ntl_mean']), cov)
        base_ntl = np.clip(base_ntl, 0, None)
        pop = rng.lognormal(mean=np.log(cfg['pop_mean']), sigma=0.8, size=n)

        for year in YEARS:
            t = year - 2015
            trend = cfg['annual_gain'] * t
            noise = rng.normal(0, 0.3, n)
            ntl = np.clip(base_ntl + trend + noise, 0, None)
            for i, geom in enumerate(geoms):
                records.append({
                    'country': cfg['country'],
                    'year': year,
                    'tile_id': f"{cfg['country'][:3].upper()}_{i:03d}",
                    'ntl_mean': ntl[i],
                    'pop': pop[i],
                    'geometry': geom,
                })

    return gpd.GeoDataFrame(records, crs='EPSG:4326')


panel = make_panel()
print(f"Panel: {len(panel)} records | {panel['country'].nunique()} countries | {panel['year'].nunique()} years")


# ─────────────────────────────────────────────────────────────────────────────
# NOTEBOOK 11 — SDG 7 Progress Tracker
# ─────────────────────────────────────────────────────────────────────────────

print("\n=== NOTEBOOK 11: SDG 7 Progress Tracker ===")

# 1. Access rate time series
ar_df = access_rate_timeseries(panel, ntl_col='ntl_mean', pop_col='pop',
                                threshold=NTL_ELECTRIFICATION_THRESHOLD)
print("\nAccess rates (2023):")
print(ar_df[ar_df.year==2023][['country','year','access_rate','ntl_mean_pw']].to_string(index=False))

# 2. Theil-Sen projections
proj_results = {}
for country in ar_df.country.unique():
    sub = ar_df[ar_df.country == country].sort_values('year')
    proj_results[country] = theil_sen_projection(
        sub['year'].values, sub['access_rate'].values,
        target_year=2030, n_bootstrap=2000, ci=0.95
    )
    r = proj_results[country]
    print(f"\n{country}: proj2030={r['projected_value']:.3f} [{r['ci_lower']:.3f}–{r['ci_upper']:.3f}]  trend={r['trend']}")

# 3. Gap analysis
gap_df = sdg7_gap_analysis(ar_df)
print("\nSDG 7 Gap Analysis:")
print(gap_df.to_string(index=False))
gap_df.to_csv(FIGURES / 'sdg7_attainment_table.csv', index=False)
print(f"Saved: figures/sdg7_attainment_table.csv")

# 4. Figure: access rate timeline (main publication figure)
fig1 = plot_access_rate_timeseries(ar_df, proj_results, save_path=FIGURES / 'sdg7_progress_timeline.png')
plt.close(fig1)
print("Saved: figures/sdg7_progress_timeline.png")

# 5. Figure: SDG7 status dashboard
fig2 = plot_sdg7_status_dashboard(gap_df, save_path=FIGURES / 'sdg7_status_dashboard.png')
plt.close(fig2)
print("Saved: figures/sdg7_status_dashboard.png")

# 6. Electrification Index (compute first — needed for comparison plot)
ei_df = electrification_index_timeseries(panel, ntl_col='ntl_mean', pop_col='pop')

# Country comparison: EI + Gini dual-axis
fig3 = plot_country_comparison(ei_df, save_path=FIGURES / 'sdg7_country_comparison.png')
plt.close(fig3)
print("Saved: figures/sdg7_country_comparison.png")

# 7. Electrification Index — already computed above, plot trend
fig4, ax4 = plt.subplots(figsize=(10, 5))
colors = {'Brazil': '#2196F3', 'China': '#FF5722', 'Morocco': '#4CAF50'}
for country in ei_df.country.unique():
    sub = ei_df[ei_df.country == country]
    ax4.plot(sub.year, sub.EI, marker='o', linewidth=2, color=colors[country], label=country)
ax4.axhline(0.95, ls='--', color='red', alpha=0.6, label='SDG7 target (0.95)')
ax4.set_xlabel('Year'); ax4.set_ylabel('Composite Electrification Index')
ax4.set_title('Composite Electrification Index (EI = 0.5·access + 0.5·(1−Gini))', fontweight='bold')
ax4.legend(); ax4.set_ylim(0.4, 1.05)
fig4.tight_layout()
fig4.savefig(FIGURES / 'sdg7_electrification_index.png', dpi=180, bbox_inches='tight')
plt.close(fig4)
print("Saved: figures/sdg7_electrification_index.png")

# 8. Urban/rural disaggregation (2023 snapshot)
snap2023 = panel[panel.year == 2023].copy()

ur_rows = []
for country in snap2023['country'].unique():
    csnap = snap2023[snap2023.country == country]
    df_ur = urban_rural_disaggregation(csnap, ntl_col='ntl_mean', pop_col='pop')
    df_ur['country'] = country
    ur_rows.append(df_ur)
ur_df = pd.concat(ur_rows, ignore_index=True)
# Normalise column names to lowercase
ur_df.columns = [c.lower().replace(' ', '_').replace('(%)','').strip('_') for c in ur_df.columns]
# access_rate is in %, convert to fraction for the bar
ur_df['access_frac'] = ur_df['access_rate'] / 100

fig5, ax5 = plt.subplots(figsize=(10, 5))
x = np.arange(len(ur_df))
colors_ur = {'Urban': '#FF5722', 'Rural': '#4CAF50'}
bar_colors = [colors_ur.get(s, '#999') for s in ur_df['stratum']]
bars = ax5.bar(x, ur_df['access_frac'], color=bar_colors, alpha=0.85, edgecolor='white')
ax5.set_xticks(x)
ax5.set_xticklabels(ur_df.apply(lambda r: f"{r['country']}\n{r['stratum']}", axis=1), fontsize=9)
ax5.axhline(0.95, ls='--', color='red', alpha=0.7, label='SDG7 target (95%)')
ax5.set_ylabel('Electricity Access Rate')
ax5.set_title('Urban vs Rural Electricity Access (2023)', fontweight='bold')
ax5.legend()
for bar, val in zip(bars, ur_df['access_frac']):
    ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
             f'{val:.1%}', ha='center', va='bottom', fontsize=8)
fig5.tight_layout()
fig5.savefig(FIGURES / 'sdg7_urban_rural_2023.png', dpi=180, bbox_inches='tight')
plt.close(fig5)
print("Saved: figures/sdg7_urban_rural_2023.png")

# 9. Threshold sensitivity analysis
fig6, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
thresholds = [
    ('Fixed 2.0', 2.0, 'fixed', '#1565C0'),
    ('Fixed 1.0', 1.0, 'fixed', '#F57C00'),
    ('25th Percentile', 25, 'percentile', '#2E7D32'),
]
for ax, (label, thresh, method, color) in zip(axes, thresholds):
    df_t = access_rate_timeseries(panel, ntl_col='ntl_mean', pop_col='pop',
                                   threshold=thresh, method=method)
    for country, cdf in df_t.groupby('country'):
        ls = '-' if country == 'Brazil' else ('--' if country == 'China' else ':')
        ax.plot(cdf.year, cdf.access_rate, ls=ls, linewidth=2, color=color, label=country)
    ax.axhline(0.95, ls='-.', color='red', alpha=0.6)
    ax.set_title(f'Threshold: {label}', fontweight='bold')
    ax.set_xlabel('Year'); ax.set_ylim(0.3, 1.05)
    if ax == axes[0]:
        ax.set_ylabel('Access Rate')
    ax.legend(fontsize=8)
fig6.suptitle('Threshold Sensitivity Analysis', fontsize=13, fontweight='bold')
fig6.tight_layout()
fig6.savefig(FIGURES / 'sdg7_threshold_sensitivity.png', dpi=180, bbox_inches='tight')
plt.close(fig6)
print("Saved: figures/sdg7_threshold_sensitivity.png")


# ─────────────────────────────────────────────────────────────────────────────
# NOTEBOOK 12 — Validation Against Official Statistics
# ─────────────────────────────────────────────────────────────────────────────

print("\n=== NOTEBOOK 12: Validation Against Official Statistics ===")

# Official statistics: World Bank WDI + IEA (% of population with access)
OFFICIAL_STATS = {
    'Brazil': {
        2015: 99.0, 2016: 99.2, 2017: 99.3, 2018: 99.4,
        2019: 99.5, 2020: 99.6, 2021: 99.6, 2022: 99.7, 2023: 99.7,
    },
    'China': {
        2015: 100.0, 2016: 100.0, 2017: 100.0, 2018: 100.0,
        2019: 100.0, 2020: 100.0, 2021: 100.0, 2022: 100.0, 2023: 100.0,
    },
    'Morocco': {
        2015: 96.8, 2016: 97.1, 2017: 97.3, 2018: 97.5,
        2019: 97.7, 2020: 97.9, 2021: 98.1, 2022: 98.3, 2023: 98.5,
    },
}

# Build validation dataframe
val_records = []
for country in ar_df.country.unique():
    for _, row in ar_df[ar_df.country == country].iterrows():
        official = OFFICIAL_STATS[country].get(int(row['year']))
        if official is not None:
            val_records.append({
                'country': country,
                'year': int(row['year']),
                'modeled': row['access_rate'] * 100,   # percent
                'official': official,
            })

val_df = pd.DataFrame(val_records)
val_df['residual'] = val_df['modeled'] - val_df['official']

print(f"\nValidation pairs: {len(val_df)}")
print(val_df.groupby('country')[['modeled','official','residual']].mean().round(2))

# Figure 1: Validation scatter plot
fig7, ax7 = plt.subplots(figsize=(8, 7))
country_styles = {
    'Brazil': dict(color='#2196F3', marker='o'),
    'China':  dict(color='#FF5722', marker='s'),
    'Morocco': dict(color='#4CAF50', marker='^'),
}
for country, cdf in val_df.groupby('country'):
    cs = country_styles[country]
    ax7.scatter(cdf['official'], cdf['modeled'], color=cs['color'], marker=cs['marker'],
                s=70, label=country, zorder=3, alpha=0.85)
    for _, row in cdf.iterrows():
        ax7.annotate(str(int(row['year'])),
                     (row['official'], row['modeled']),
                     fontsize=6.5, alpha=0.7, xytext=(2, 2), textcoords='offset points')

# 1:1 line
lim = [94, 101]
ax7.plot(lim, lim, 'k--', alpha=0.5, linewidth=1.2, label='1:1 line')

# Linear fit
slope, intercept, r, p, se = stats.linregress(val_df['official'], val_df['modeled'])
x_fit = np.linspace(val_df['official'].min(), val_df['official'].max(), 100)
ax7.plot(x_fit, slope * x_fit + intercept, 'r-', alpha=0.7, linewidth=1.5,
         label=f'OLS fit (R²={r**2:.3f}, slope={slope:.3f})')

ax7.set_xlabel('Official Statistic — World Bank WDI/IEA (%)', fontsize=11)
ax7.set_ylabel('VIIRS NTL Model Estimate (%)', fontsize=11)
ax7.set_title('Validation: VIIRS-modelled vs Official Electricity Access\n'
              'Brazil, China, Morocco — 2015–2023', fontsize=12, fontweight='bold')
ax7.legend(fontsize=9)
ax7.set_xlim(lim); ax7.set_ylim([60, 101])

stats_text = (f"N = {len(val_df)}\nR² = {r**2:.3f}\nSlope = {slope:.3f}\n"
              f"MAE = {np.abs(val_df['residual']).mean():.2f}pp\n"
              f"RMSE = {np.sqrt((val_df['residual']**2).mean()):.2f}pp")
ax7.text(0.04, 0.96, stats_text, transform=ax7.transAxes, fontsize=8.5,
         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

fig7.tight_layout()
fig7.savefig(FIGURES / 'validation_scatter.png', dpi=180, bbox_inches='tight')
plt.close(fig7)
print(f"\nValidation R²={r**2:.3f}, MAE={np.abs(val_df['residual']).mean():.2f}pp, RMSE={np.sqrt((val_df['residual']**2).mean()):.2f}pp")
print("Saved: figures/validation_scatter.png")

# Figure 2: Residuals over time
fig8, axes8 = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
for ax, (country, cdf) in zip(axes8, val_df.groupby('country')):
    cdf = cdf.sort_values('year')
    ax.bar(cdf.year, cdf.residual, color=['#E53935' if r > 0 else '#1E88E5' for r in cdf.residual], alpha=0.8)
    ax.axhline(0, color='black', linewidth=1)
    ax.set_title(f'{country}', fontweight='bold')
    ax.set_xlabel('Year')
    if ax == axes8[0]:
        ax.set_ylabel('Residual (Modeled − Official, pp)')
    mean_r = cdf.residual.mean()
    ax.text(0.97, 0.97, f'Mean: {mean_r:+.1f}pp', transform=ax.transAxes,
            ha='right', va='top', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

fig8.suptitle('Residuals: VIIRS Model vs Official Statistics\n'
              'Positive = overestimate, Negative = underestimate',
              fontsize=12, fontweight='bold')
fig8.tight_layout()
fig8.savefig(FIGURES / 'validation_residuals.png', dpi=180, bbox_inches='tight')
plt.close(fig8)
print("Saved: figures/validation_residuals.png")

# Figure 3: Urban vs Rural comparison (modeled vs official proxy)
# Official urban/rural from World Bank (2023 snapshot)
OFFICIAL_UR = pd.DataFrame([
    dict(country='Brazil',  stratum='urban', official=99.9, modeled=None),
    dict(country='Brazil',  stratum='rural', official=97.5, modeled=None),
    dict(country='China',   stratum='urban', official=100.0, modeled=None),
    dict(country='China',   stratum='rural', official=100.0, modeled=None),
    dict(country='Morocco', stratum='urban', official=99.6, modeled=None),
    dict(country='Morocco', stratum='rural', official=96.8, modeled=None),
])

OFFICIAL_UR['modeled'] = OFFICIAL_UR.apply(
    lambda r: ur_df[(ur_df.country == r.country) &
                    (ur_df.stratum == r.stratum.title())]['access_rate'].values[0]
    if len(ur_df[(ur_df.country == r.country) &
                 (ur_df.stratum == r.stratum.title())]) > 0 else np.nan,
    axis=1
)

fig9, ax9 = plt.subplots(figsize=(11, 6))
x = np.arange(len(OFFICIAL_UR))
width = 0.35
bars1 = ax9.bar(x - width/2, OFFICIAL_UR['official'], width, label='Official (WDI/IEA)',
                color='#455A64', alpha=0.85, edgecolor='white')
bars2 = ax9.bar(x + width/2, OFFICIAL_UR['modeled'], width, label='VIIRS NTL Model',
                color='#1976D2', alpha=0.85, edgecolor='white')
ax9.set_xticks(x)
ax9.set_xticklabels(OFFICIAL_UR.apply(lambda r: f"{r['country']}\n({r['stratum'].title()})", axis=1), fontsize=9)
ax9.set_ylabel('Electricity Access Rate (%)')
ax9.set_title('Urban vs Rural Electricity Access: Official vs VIIRS Model (2023)', fontweight='bold')
ax9.legend()
ax9.set_ylim(60, 103)
ax9.axhline(95, ls='--', color='red', alpha=0.5, label='SDG7 target')
for bar in [*bars1, *bars2]:
    ax9.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
             f'{bar.get_height():.1f}', ha='center', va='bottom', fontsize=7.5)
fig9.tight_layout()
fig9.savefig(FIGURES / 'validation_urban_rural.png', dpi=180, bbox_inches='tight')
plt.close(fig9)
print("Saved: figures/validation_urban_rural.png")

# Figure 4: Error metrics summary table (styled)
metrics_rows = []
for country, cdf in val_df.groupby('country'):
    metrics_rows.append({
        'Country': country,
        'N pairs': len(cdf),
        'MAE (pp)': round(np.abs(cdf.residual).mean(), 2),
        'RMSE (pp)': round(np.sqrt((cdf.residual**2).mean()), 2),
        'Mean bias (pp)': round(cdf.residual.mean(), 2),
        'R²': round(stats.pearsonr(cdf.official, cdf.modeled)[0]**2, 3),
        'Correlation r': round(stats.pearsonr(cdf.official, cdf.modeled)[0], 3),
    })
metrics_df = pd.DataFrame(metrics_rows)
metrics_df.to_csv(FIGURES / 'validation_metrics_table.csv', index=False)
print("\nValidation metrics:")
print(metrics_df.to_string(index=False))
print("Saved: figures/validation_metrics_table.csv")

# Aggregated across all countries
all_r2 = stats.pearsonr(val_df['official'], val_df['modeled'])[0]**2
all_mae = np.abs(val_df['residual']).mean()
all_rmse = np.sqrt((val_df['residual']**2).mean())
print(f"\nOverall: R²={all_r2:.3f}, MAE={all_mae:.2f}pp, RMSE={all_rmse:.2f}pp")

print("\nAll figures generated successfully.")
