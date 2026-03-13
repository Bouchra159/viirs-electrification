"""
bivariate.py
------------
Bivariate choropleth mapping for VIIRS NTL × population density analysis.

Implements the classic 3×3 bivariate colour scheme used in cartographic research.
Variable 1 (x-axis): NTL radiance intensity (electrification proxy)
Variable 2 (y-axis): population density (exposure weight)

Author: Bouchra Daddaoui
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from matplotlib.patches import FancyArrow
from pathlib import Path


# ---------------------------------------------------------------------------
# Classic Cindy Brewer 3×3 bivariate palette
# Columns = NTL classes (1=low → 3=high)
# Rows    = population density classes (1=low → 3=high)
# ---------------------------------------------------------------------------

# [row_pop][col_ntl]
BIVARIATE_PALETTE = {
    (1, 1): '#e8e8e8',  # low pop,  low NTL  → light grey
    (1, 2): '#ace4e4',  # low pop,  mid NTL  → light cyan
    (1, 3): '#5ac8c8',  # low pop,  high NTL → teal
    (2, 1): '#dfb0d6',  # mid pop,  low NTL  → light mauve
    (2, 2): '#a5b3cc',  # mid pop,  mid NTL  → slate
    (2, 3): '#5698b9',  # mid pop,  high NTL → steel blue
    (3, 1): '#be64ac',  # high pop, low NTL  → pink-purple (energy poor)
    (3, 2): '#8c62aa',  # high pop, mid NTL  → purple
    (3, 3): '#3b4994',  # high pop, high NTL → deep indigo (electrified, dense)
}

CLASS_LABELS = {1: 'Low', 2: 'Mid', 3: 'High'}


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

def quantile_classify(series: pd.Series, n: int = 3) -> pd.Series:
    """Assign 1/2/3 classes by equal-count quantiles."""
    q = np.linspace(0, 100, n + 1)
    bins = np.percentile(series.dropna(), q)
    bins[0] -= 1e-9  # include minimum
    return pd.cut(series, bins=bins, labels=range(1, n + 1)).astype(float).astype('Int64')


def assign_bivariate_class(
    gdf: gpd.GeoDataFrame,
    ntl_col: str = 'ntl_mean',
    pop_col: str = 'pop_density',
) -> gpd.GeoDataFrame:
    """
    Add bivariate class columns to a GeoDataFrame.

    Adds:
      ntl_class     : 1–3 (NTL quantile)
      pop_class     : 1–3 (population density quantile)
      biv_class     : (pop_class, ntl_class) tuple key
      biv_colour    : hex colour string
      biv_label     : e.g. 'High pop / Low NTL'
    """
    gdf = gdf.copy()
    gdf['ntl_class'] = quantile_classify(gdf[ntl_col])
    gdf['pop_class'] = quantile_classify(gdf[pop_col])
    gdf['biv_class'] = list(zip(gdf['pop_class'], gdf['ntl_class']))
    gdf['biv_colour'] = gdf['biv_class'].map(BIVARIATE_PALETTE)
    gdf['biv_label'] = gdf.apply(
        lambda r: f"{CLASS_LABELS.get(r['pop_class'], '?')} pop / {CLASS_LABELS.get(r['ntl_class'], '?')} NTL",
        axis=1,
    )
    return gdf


# ---------------------------------------------------------------------------
# Map plotting
# ---------------------------------------------------------------------------

def plot_bivariate_map(
    gdf: gpd.GeoDataFrame,
    country: str,
    ax: plt.Axes,
    ntl_col: str = 'ntl_mean',
    pop_col: str = 'pop_density',
) -> None:
    """Plot bivariate choropleth for one country on ax."""
    gdf = assign_bivariate_class(gdf, ntl_col=ntl_col, pop_col=pop_col)
    gdf.plot(color=gdf['biv_colour'], edgecolor='white', linewidth=0.3, ax=ax)
    ax.set_title(f'{country}', fontsize=11, fontweight='bold')
    ax.set_axis_off()


def add_bivariate_legend(
    fig: plt.Figure,
    ax_legend: plt.Axes,
    ntl_label: str = 'NTL Radiance →',
    pop_label: str = 'Population Density →',
) -> None:
    """Draw a 3×3 colour matrix legend on ax_legend."""
    ax_legend.set_xlim(0, 3)
    ax_legend.set_ylim(0, 3)
    ax_legend.set_aspect('equal')
    ax_legend.set_axis_off()

    for (row_pop, col_ntl), colour in BIVARIATE_PALETTE.items():
        rect = plt.Rectangle(
            (col_ntl - 1, row_pop - 1), 1, 1,
            facecolor=colour, edgecolor='white', linewidth=0.5,
        )
        ax_legend.add_patch(rect)

    # Axis labels
    ax_legend.text(1.5, -0.4, ntl_label, ha='center', va='top', fontsize=8, fontweight='bold')
    ax_legend.text(-0.4, 1.5, pop_label, ha='right', va='center', fontsize=8,
                   fontweight='bold', rotation=90)

    # Low / High tick labels
    for i, label in enumerate(['Low', 'Mid', 'High']):
        ax_legend.text(i + 0.5, -0.15, label, ha='center', va='top', fontsize=6.5)
        ax_legend.text(-0.15, i + 0.5, label, ha='right', va='center', fontsize=6.5)


def plot_bivariate_panel(
    gdfs: dict[str, gpd.GeoDataFrame],
    ntl_col: str = 'ntl_mean',
    pop_col: str = 'pop_density',
    save_path: str | Path | None = None,
) -> plt.Figure:
    """
    Full 3-country bivariate panel with shared legend.

    Parameters
    ----------
    gdfs : dict country → GeoDataFrame (must have ntl_col and pop_col)
    save_path : optional output path for PNG

    Returns
    -------
    matplotlib Figure
    """
    fig = plt.figure(figsize=(18, 7))

    # Country axes
    ax1 = fig.add_axes([0.01, 0.05, 0.28, 0.88])
    ax2 = fig.add_axes([0.30, 0.05, 0.28, 0.88])
    ax3 = fig.add_axes([0.59, 0.05, 0.28, 0.88])
    ax_leg = fig.add_axes([0.89, 0.25, 0.10, 0.45])

    for ax, (country, gdf) in zip([ax1, ax2, ax3], gdfs.items()):
        plot_bivariate_map(gdf, country, ax, ntl_col=ntl_col, pop_col=pop_col)

    add_bivariate_legend(fig, ax_leg)

    fig.suptitle(
        'Bivariate Map: NTL Radiance × Population Density\n'
        'Electrification intensity relative to demographic exposure',
        fontsize=13, fontweight='bold', y=1.01,
    )

    if save_path is not None:
        fig.savefig(save_path, dpi=180, bbox_inches='tight')
        print(f'Saved: {save_path}')

    return fig


# ---------------------------------------------------------------------------
# Per-country summary table
# ---------------------------------------------------------------------------

def bivariate_summary(gdf: gpd.GeoDataFrame, country: str) -> pd.DataFrame:
    """Return tile counts per bivariate class."""
    gdf = assign_bivariate_class(gdf)
    tbl = (
        gdf.groupby('biv_label')
        .size()
        .reset_index(name='n_tiles')
        .assign(pct=lambda d: (d['n_tiles'] / len(gdf) * 100).round(1))
        .sort_values('n_tiles', ascending=False)
    )
    tbl.insert(0, 'country', country)
    return tbl
