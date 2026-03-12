"""
gwr_analysis.py
---------------
Geographically Weighted Regression (GWR) for VIIRS electrification data.

GWR relaxes the stationarity assumption of OLS by fitting a local regression
at each observation, allowing coefficients to vary across space. This reveals
*where* a variable drives electrification and where it doesn't.

We use the mgwr package (Multiscale GWR), which extends standard GWR by
allowing each predictor to have its own bandwidth — more flexible and
statistically sound than classic GWR.
"""

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path

from mgwr.gwr import GWR, MGWR
from mgwr.sel_bw import Sel_BW
from libpysal.weights import KNN
import spreg


def prep_coords(gdf):
    """Extract centroid coordinates as a (n, 2) array for GWR."""
    return np.column_stack([
        gdf.geometry.centroid.x,
        gdf.geometry.centroid.y
    ])


def run_gwr(gdf, y_col, x_cols, bw=None, kernel='bisquare', fixed=False):
    """
    Fit a GWR model on a GeoDataFrame.

    bw=None triggers automatic bandwidth selection via AIC.
    Returns the fitted GWR results object and the selected bandwidth.
    """
    coords = prep_coords(gdf)
    y = gdf[y_col].values.reshape(-1, 1)
    X = gdf[x_cols].values

    # Standardise X so coefficients are comparable across features
    X_std = (X - X.mean(axis=0)) / X.std(axis=0)

    if bw is None:
        selector = Sel_BW(coords, y, X_std, kernel=kernel, fixed=fixed)
        bw = selector.search(criterion='AIC')

    model = GWR(coords, y, X_std, bw=bw, kernel=kernel, fixed=fixed)
    results = model.fit()
    return results, bw, X_std


def run_mgwr(gdf, y_col, x_cols, kernel='bisquare', fixed=False):
    """
    Fit a Multiscale GWR model — each predictor gets its own optimal bandwidth.

    This is more appropriate than standard GWR when predictors operate at
    different spatial scales (e.g. infrastructure is local, GDP is regional).
    Returns the MGWR results object and bandwidth list.
    """
    coords = prep_coords(gdf)
    y = gdf[y_col].values.reshape(-1, 1)
    X = gdf[x_cols].values
    X_std = (X - X.mean(axis=0)) / X.std(axis=0)

    # Initial bandwidths from GWR for MGWR starting point
    selector = Sel_BW(coords, y, X_std, multi=True, kernel=kernel, fixed=fixed)
    bws = selector.search(multi_bw_min=[2], criterion='AIC')

    model = MGWR(coords, y, X_std, selector, kernel=kernel, fixed=fixed)
    results = model.fit()
    return results, bws, X_std


def gwr_coefficient_maps(gdf, results, x_cols, ncols=3,
                         title_prefix='GWR Local Coefficient',
                         save_path=None):
    """
    Plot a grid of choropleth maps showing how each GWR coefficient varies spatially.

    Red = strong positive effect, Blue = strong negative effect.
    Tiles where t-value < 1.96 are greyed out (not significant at p < 0.05).
    """
    # params: intercept + one per feature
    n_params = len(x_cols) + 1
    nrows = int(np.ceil(n_params / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows))
    axes = axes.flatten()

    param_names = ['Intercept'] + x_cols

    for i, (name, ax) in enumerate(zip(param_names, axes)):
        coefs = results.params[:, i]
        # t-values for significance filtering
        t_vals = results.tvalues[:, i]
        sig = np.abs(t_vals) >= 1.96

        gdf_plot = gdf.copy()
        gdf_plot['coef'] = coefs
        gdf_plot['sig'] = sig

        vmax = np.quantile(np.abs(coefs[sig]), 0.95) if sig.sum() > 0 else np.abs(coefs).max()
        vmax = max(vmax, 0.01)

        # Non-significant tiles in grey
        # Plot all tiles first (avoids empty-GDF aspect error)
        gdf_plot.plot(color='#dddddd', edgecolor='none', ax=ax)
        if gdf_plot['sig'].sum() > 0:
            gdf_plot[gdf_plot['sig']].plot(
                column='coef', cmap='RdBu_r',
                vmin=-vmax, vmax=vmax,
                edgecolor='none', legend=True,
                legend_kwds={'label': 'Coefficient', 'shrink': 0.6},
                ax=ax
            )
        ax.set_title(f'{title_prefix}\n{name}', fontsize=10, fontweight='bold')
        ax.set_axis_off()

    # Hide unused axes
    for ax in axes[n_params:]:
        ax.set_visible(False)

    fig.suptitle('GWR Local Coefficients (grey = not significant, p > 0.05)',
                 fontsize=12, fontweight='bold')
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


def gwr_r2_map(gdf, results, title='GWR Local R²', save_path=None):
    """Map the local R² from GWR — where does the model fit well vs poorly?"""
    gdf_plot = gdf.copy()
    gdf_plot['local_r2'] = results.localR2

    fig, ax = plt.subplots(figsize=(9, 7))
    gdf_plot.plot(
        column='local_r2', cmap='YlGnBu',
        vmin=0, vmax=1,
        edgecolor='grey', linewidth=0.15, legend=True,
        legend_kwds={'label': 'Local R²', 'shrink': 0.65},
        ax=ax
    )
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_axis_off()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


def gwr_vs_ols_summary(gdf, y_col, x_cols, gwr_results):
    """
    Compare OLS vs GWR: print AIC, R², and bandwidth.
    Useful for justifying why GWR is worth the added complexity.
    """
    y = gdf[y_col].values.reshape(-1, 1)
    X = gdf[x_cols].values
    X_std = (X - X.mean(axis=0)) / X.std(axis=0)

    ols = spreg.OLS(y, X_std, name_y=y_col, name_x=x_cols)

    print("Model Comparison: OLS vs GWR")
    print("=" * 40)
    print(f"  OLS  AIC = {ols.aic:.2f} | R² = {ols.r2:.4f}")
    print(f"  GWR  AIC = {gwr_results.aic:.2f} | Mean local R² = {gwr_results.localR2.mean():.4f}")
    delta_aic = ols.aic - gwr_results.aic
    print(f"  ΔAIC (OLS - GWR) = {delta_aic:.2f}  {'→ GWR preferred' if delta_aic > 2 else '→ OLS comparable'}")
    return {'ols_aic': ols.aic, 'gwr_aic': gwr_results.aic, 'delta_aic': delta_aic}


def bandwidth_summary(x_cols, bws):
    """Print MGWR bandwidths per predictor — shows spatial scale of each effect."""
    print("\nMGWR Bandwidth Summary (larger = more global effect)")
    print("-" * 45)
    names = ['Intercept'] + list(x_cols)
    for name, bw in zip(names, bws):
        bar = '█' * min(int(bw / 5), 30)
        print(f"  {name:20s}: {bw:6.1f}  {bar}")
