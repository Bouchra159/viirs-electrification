"""
spatial_analysis.py
-------------------
Spatial econometrics and autocorrelation analysis for VIIRS nighttime light data.

Provides:
  - Spatial weights construction (Queen/KNN)
  - Global and Local Moran's I
  - Spatial lag / spatial error regression
  - LISA cluster map generation

Author: Bouchra Daddaoui
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import Optional, Tuple

import libpysal
from libpysal.weights import Queen, KNN, lat2W
import esda
from esda.moran import Moran, Moran_Local
import spreg
from splot.esda import moran_scatterplot, lisa_cluster


# ---------------------------------------------------------------------------
# Spatial weights
# ---------------------------------------------------------------------------

def build_queen_weights(gdf: gpd.GeoDataFrame, id_col: str = "tile_id") -> libpysal.weights.W:
    """
    Build Queen contiguity spatial weights from a GeoDataFrame.

    Parameters
    ----------
    gdf : GeoDataFrame with geometry column.
    id_col : column to use as observation IDs.

    Returns
    -------
    W : row-standardised Queen contiguity weights.
    """
    w = Queen.from_dataframe(gdf, idVariable=id_col)
    w.transform = "r"  # row-standardise
    return w


def build_knn_weights(gdf: gpd.GeoDataFrame, k: int = 8) -> libpysal.weights.W:
    """
    Build K-nearest-neighbour spatial weights.

    Parameters
    ----------
    gdf : GeoDataFrame.
    k   : number of neighbours.

    Returns
    -------
    W : row-standardised KNN weights.
    """
    coords = np.column_stack([gdf.geometry.centroid.x, gdf.geometry.centroid.y])
    w = KNN.from_array(coords, k=k)
    w.transform = "r"
    return w


# ---------------------------------------------------------------------------
# Global Moran's I
# ---------------------------------------------------------------------------

def global_moran(
    values: np.ndarray,
    w: libpysal.weights.W,
    permutations: int = 999,
) -> dict:
    """
    Compute Global Moran's I for spatial autocorrelation.

    Parameters
    ----------
    values       : 1-D array of the variable of interest.
    w            : Spatial weights object.
    permutations : Number of MC permutations for pseudo p-value.

    Returns
    -------
    dict with keys: I, E_I, z_norm, p_norm, p_sim
    """
    mi = Moran(values, w, permutations=permutations)
    return {
        "I": mi.I,
        "E_I": mi.EI,
        "z_norm": mi.z_norm,
        "p_norm": mi.p_norm,
        "p_sim": mi.p_sim,
    }


def plot_moran_scatterplot(
    values: np.ndarray,
    w: libpysal.weights.W,
    variable_name: str = "VIIRS NTL",
    ax: Optional[plt.Axes] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Moran scatterplot with spatial lag on y-axis.

    Parameters
    ----------
    values        : 1-D array of the variable.
    w             : Spatial weights.
    variable_name : Label for the x-axis.
    ax            : Matplotlib axes (created if None).
    save_path     : If given, figure is saved here.

    Returns
    -------
    fig : Matplotlib Figure.
    """
    mi = Moran(values, w, permutations=999)
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    else:
        fig = ax.get_figure()
    moran_scatterplot(mi, ax=ax, aspect_equal=True)
    ax.set_xlabel(variable_name)
    ax.set_ylabel(f"Spatial lag of {variable_name}")
    ax.set_title(f"Moran's I = {mi.I:.4f}  (p = {mi.p_sim:.3f})")
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# Local Moran's I (LISA)
# ---------------------------------------------------------------------------

def local_moran(
    values: np.ndarray,
    w: libpysal.weights.W,
    permutations: int = 999,
    significance: float = 0.05,
) -> pd.DataFrame:
    """
    Compute Local Moran's I (LISA) and assign cluster labels.

    Cluster codes
    -------------
    0 : Not significant
    1 : High-High (HH) hotspot
    2 : Low-Low  (LL) coldspot
    3 : High-Low (HL) spatial outlier
    4 : Low-High (LH) spatial outlier

    Parameters
    ----------
    values       : 1-D array.
    w            : Spatial weights.
    permutations : MC permutations.
    significance : p-value threshold for significance.

    Returns
    -------
    DataFrame with columns [Is, p_sim, cluster_code, cluster_label].
    """
    lisa = Moran_Local(values, w, permutations=permutations)

    labels_map = {0: "Not significant", 1: "High-High", 2: "Low-Low",
                  3: "High-Low", 4: "Low-High"}

    sig = lisa.p_sim < significance
    codes = np.where(sig, lisa.q, 0).astype(int)
    labels = [labels_map[c] for c in codes]

    return pd.DataFrame({
        "Is": lisa.Is,
        "p_sim": lisa.p_sim,
        "cluster_code": codes,
        "cluster_label": labels,
    })


def plot_lisa_map(
    gdf: gpd.GeoDataFrame,
    lisa_df: pd.DataFrame,
    title: str = "LISA Cluster Map",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Choropleth map of LISA cluster types.

    Parameters
    ----------
    gdf       : GeoDataFrame (same order as lisa_df).
    lisa_df   : Output of local_moran().
    title     : Map title.
    save_path : Save path for PNG.

    Returns
    -------
    fig : Matplotlib Figure.
    """
    palette = {
        "High-High": "#d7191c",
        "Low-Low":   "#2c7bb6",
        "High-Low":  "#fdae61",
        "Low-High":  "#abd9e9",
        "Not significant": "#eeeeee",
    }

    gdf = gdf.copy()
    gdf["cluster_label"] = lisa_df["cluster_label"].values
    gdf["color"] = gdf["cluster_label"].map(palette)

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    gdf.plot(color=gdf["color"], linewidth=0.3, edgecolor="white", ax=ax)

    patches = [mpatches.Patch(color=c, label=l) for l, c in palette.items()]
    ax.legend(handles=patches, loc="lower right", fontsize=9, framealpha=0.9)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_axis_off()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# Spatial regression
# ---------------------------------------------------------------------------

def run_ols(
    y: np.ndarray,
    X: np.ndarray,
    x_names: list[str],
    y_name: str = "NTL",
    w: Optional[libpysal.weights.W] = None,
) -> spreg.OLS:
    """
    OLS regression with optional spatial diagnostics (Moran, LM tests).

    Parameters
    ----------
    y      : Dependent variable (n,1).
    X      : Independent variables (n, k).
    x_names: Feature names for printing.
    y_name : Label for dependent variable.
    w      : Spatial weights (required for diagnostics).

    Returns
    -------
    OLS model object.
    """
    model = spreg.OLS(
        y, X,
        w=w,
        name_y=y_name,
        name_x=x_names,
        spat_diag=w is not None,
        moran=w is not None,
    )
    return model


def run_spatial_lag(
    y: np.ndarray,
    X: np.ndarray,
    w: libpysal.weights.W,
    x_names: list[str],
    y_name: str = "NTL",
) -> spreg.ML_Lag:
    """
    Maximum-likelihood spatial lag model (SLM).

    Parameters
    ----------
    y, X      : Dependent and independent variables.
    w         : Spatial weights.
    x_names   : Feature labels.
    y_name    : Dependent variable label.

    Returns
    -------
    ML_Lag model object.
    """
    model = spreg.ML_Lag(
        y, X, w=w,
        name_y=y_name,
        name_x=x_names,
    )
    return model


def run_spatial_error(
    y: np.ndarray,
    X: np.ndarray,
    w: libpysal.weights.W,
    x_names: list[str],
    y_name: str = "NTL",
) -> spreg.ML_Error:
    """
    Maximum-likelihood spatial error model (SEM).

    Parameters
    ----------
    y, X    : Dependent and independent variables.
    w       : Spatial weights.
    x_names : Feature labels.
    y_name  : Label for dependent variable.

    Returns
    -------
    ML_Error model object.
    """
    model = spreg.ML_Error(
        y, X, w=w,
        name_y=y_name,
        name_x=x_names,
    )
    return model


def regression_summary_table(models: dict) -> pd.DataFrame:
    """
    Produce a comparison table for OLS, SLM, SEM.

    Parameters
    ----------
    models : Dict mapping model name → fitted model object.

    Returns
    -------
    DataFrame with AIC, log-likelihood, pseudo-R², rho/lambda columns.
    """
    rows = []
    for name, m in models.items():
        row = {"Model": name}
        row["Log-likelihood"] = round(m.logll, 3) if hasattr(m, "logll") else np.nan
        row["AIC"] = round(m.aic, 3) if hasattr(m, "aic") else np.nan
        if hasattr(m, "rho"):
            row["Spatial param (ρ/λ)"] = round(m.rho, 4)
        elif hasattr(m, "lam"):
            row["Spatial param (ρ/λ)"] = round(m.lam, 4)
        else:
            row["Spatial param (ρ/λ)"] = np.nan
        rows.append(row)
    return pd.DataFrame(rows).set_index("Model")
