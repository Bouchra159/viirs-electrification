"""
data_utils.py
-------------
Utility functions for loading and preprocessing GEE-exported data.

Google Earth Engine exports VIIRS/WorldPop/GHSL assets as CSV files
(feature collections) or GeoTIFFs.  These helpers normalise the column
naming, merge layers, apply quality filters, and build the GeoDataFrame
that all downstream notebooks expect.

Usage
-----
    from data_utils import load_gee_exports, build_analysis_gdf

    gdf = build_analysis_gdf(data_dir="data/raw", country="Morocco")

Author: Bouchra Daddaoui
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point, box

logger = logging.getLogger(__name__)

# ── Column alias maps ─────────────────────────────────────────────────────────
# GEE sometimes exports with system: prefixes or camelCase; these dicts
# normalise everything to snake_case before analysis.

_NTL_ALIASES: dict[str, str] = {
    "avg_rad":         "ntl_mean",
    "cf_cvg":          "cloud_cover",
    "system:index":    "tile_id",
    "latitude":        "lat",
    "longitude":       "lon",
    ".geo":            "_geo_json",   # drop later
}

_POP_ALIASES: dict[str, str] = {
    "population":      "pop_total",
    "b1":              "pop_total",
    "system:index":    "tile_id",
}

_GHSL_ALIASES: dict[str, str] = {
    "built_s":         "built_area_m2",
    "system:index":    "tile_id",
}


# ── Loaders ───────────────────────────────────────────────────────────────────

def load_ntl_csv(path: str | Path, year: Optional[int] = None) -> pd.DataFrame:
    """Load a VIIRS DNB CSV export from GEE.

    Parameters
    ----------
    path : path to CSV file
    year : if provided, adds a 'year' column

    Returns
    -------
    DataFrame with normalised column names and no-data rows dropped.
    """
    df = pd.read_csv(path)
    df = df.rename(columns={k: v for k, v in _NTL_ALIASES.items() if k in df.columns})
    df = df.drop(columns=[c for c in df.columns if c.startswith("system:")], errors="ignore")
    df = df.drop(columns=["_geo_json"], errors="ignore")

    # Mask stray-light artefacts: negative radiance → NaN
    if "ntl_mean" in df.columns:
        df.loc[df["ntl_mean"] < 0, "ntl_mean"] = np.nan

    if year is not None:
        df["year"] = year

    df = df.dropna(subset=["ntl_mean"])
    logger.info("Loaded NTL CSV (%s): %d rows, year=%s", Path(path).name, len(df), year)
    return df


def load_pop_csv(path: str | Path) -> pd.DataFrame:
    """Load a WorldPop CSV export from GEE."""
    df = pd.read_csv(path)
    df = df.rename(columns={k: v for k, v in _POP_ALIASES.items() if k in df.columns})
    df = df.drop(columns=[c for c in df.columns if c.startswith("system:")], errors="ignore")
    df["pop_density"] = df["pop_total"]   # rename for consistency with analysis scripts
    return df


def load_ghsl_csv(path: str | Path) -> pd.DataFrame:
    """Load a GHSL built-up surface CSV export from GEE."""
    df = pd.read_csv(path)
    df = df.rename(columns={k: v for k, v in _GHSL_ALIASES.items() if k in df.columns})
    df = df.drop(columns=[c for c in df.columns if c.startswith("system:")], errors="ignore")
    return df


def load_ntl_timeseries(
    data_dir: str | Path,
    country: str,
    years: range | list[int] = range(2014, 2024),
) -> pd.DataFrame:
    """Load annual NTL CSVs for a country and stack into a long DataFrame.

    Expects files named:  data_dir/ntl_{country}_{year}.csv

    Parameters
    ----------
    data_dir : directory containing the GEE CSV exports
    country  : country name (case-insensitive, used in filename)
    years    : iterable of years to load

    Returns
    -------
    Long-format DataFrame with columns [tile_id, lat, lon, ntl_mean, year]
    """
    data_dir = Path(data_dir)
    dfs = []
    for yr in years:
        fname = data_dir / f"ntl_{country.lower()}_{yr}.csv"
        if not fname.exists():
            logger.warning("Missing file: %s — skipped", fname)
            continue
        dfs.append(load_ntl_csv(fname, year=yr))

    if not dfs:
        raise FileNotFoundError(
            f"No NTL CSV files found for '{country}' in {data_dir}. "
            "Run scripts/gee_export.js in GEE to generate them."
        )
    return pd.concat(dfs, ignore_index=True)


# ── GeoDataFrame builder ──────────────────────────────────────────────────────

def build_analysis_gdf(
    data_dir: str | Path,
    country: str,
    year: int = 2020,
    ntl_threshold: float = 2.0,
    tile_size_deg: float = 0.01,
) -> gpd.GeoDataFrame:
    """Build the analysis GeoDataFrame for one country/year.

    Loads NTL (required), then optionally merges WorldPop and GHSL if present.
    Creates square tile polygons from (lat, lon) centroids.

    Parameters
    ----------
    data_dir      : directory containing GEE CSV exports
    country       : country name
    year          : analysis year
    ntl_threshold : NTL radiance below which a tile is considered 'unelectrified'
    tile_size_deg : approximate tile side length in decimal degrees (0.01° ≈ 1 km)

    Returns
    -------
    GeoDataFrame in EPSG:4326 with columns:
        tile_id, ntl_mean, pop_density, electrified, geometry
        + optional: built_area_m2 (if GHSL available)
    """
    data_dir = Path(data_dir)

    # --- NTL (required) ---
    ntl_path = data_dir / f"ntl_{country.lower()}_{year}.csv"
    if not ntl_path.exists():
        raise FileNotFoundError(
            f"NTL file not found: {ntl_path}\n"
            "Run scripts/gee_export.js in the GEE Code Editor and "
            "export the result to your Google Drive, then place it here."
        )
    df = load_ntl_csv(ntl_path, year=year)

    # --- WorldPop (optional) ---
    pop_path = data_dir / f"pop_{country.lower()}_{year}.csv"
    if pop_path.exists():
        pop = load_pop_csv(pop_path)
        merge_cols = [c for c in ["tile_id", "lat", "lon"] if c in pop.columns]
        df = df.merge(pop[merge_cols + ["pop_density"]], on=merge_cols, how="left")
    else:
        logger.warning("WorldPop CSV not found (%s) — pop_density filled with NaN", pop_path)
        df["pop_density"] = np.nan

    # --- GHSL (optional) ---
    ghsl_path = data_dir / f"ghsl_{country.lower()}.csv"
    if ghsl_path.exists():
        ghsl = load_ghsl_csv(ghsl_path)
        merge_cols = [c for c in ["tile_id", "lat", "lon"] if c in ghsl.columns]
        df = df.merge(ghsl[merge_cols + ["built_area_m2"]], on=merge_cols, how="left")

    # --- Electrification label ---
    df["electrified"] = (df["ntl_mean"] >= ntl_threshold).astype(int)

    # --- Build tile polygons from centroids ---
    half = tile_size_deg / 2.0
    if {"lat", "lon"}.issubset(df.columns):
        geoms = [
            box(row.lon - half, row.lat - half, row.lon + half, row.lat + half)
            for row in df.itertuples()
        ]
    else:
        # Fallback: point geometry
        geoms = [Point(row.lon, row.lat) for row in df.itertuples()]

    gdf = gpd.GeoDataFrame(df, geometry=geoms, crs="EPSG:4326")
    gdf["country"] = country
    logger.info("Built GDF for %s %d: %d tiles, %.1f%% electrified",
                country, year, len(gdf),
                100 * gdf["electrified"].mean())
    return gdf


# ── Quality helpers ───────────────────────────────────────────────────────────

def filter_cloud_cover(df: pd.DataFrame, max_cf: float = 0.3) -> pd.DataFrame:
    """Drop tiles where cloud-cover fraction exceeds max_cf."""
    if "cloud_cover" not in df.columns:
        return df
    before = len(df)
    df = df[df["cloud_cover"] <= max_cf].copy()
    logger.info("Cloud filter (cf <= %.2f): %d → %d rows", max_cf, before, len(df))
    return df


def remove_outliers_iqr(df: pd.DataFrame, col: str, k: float = 3.0) -> pd.DataFrame:
    """Remove outliers beyond k × IQR from the median (for NTL)."""
    q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
    iqr = q3 - q1
    mask = (df[col] >= q1 - k * iqr) & (df[col] <= q3 + k * iqr)
    before = len(df)
    df = df[mask].copy()
    logger.info("IQR filter on '%s' (k=%.1f): %d → %d rows", col, k, before, len(df))
    return df


def log_transform_ntl(df: pd.DataFrame, col: str = "ntl_mean") -> pd.DataFrame:
    """Apply log1p transform to NTL radiance (stabilises variance)."""
    df = df.copy()
    df[f"{col}_log"] = np.log1p(df[col].clip(lower=0))
    return df


# ── Summary stats ─────────────────────────────────────────────────────────────

def electrification_summary(gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    """Compute per-country electrification statistics."""
    rows = []
    for country, grp in gdf.groupby("country"):
        rows.append({
            "country":         country,
            "n_tiles":         len(grp),
            "ntl_mean":        grp["ntl_mean"].mean(),
            "ntl_median":      grp["ntl_mean"].median(),
            "ntl_std":         grp["ntl_mean"].std(),
            "pct_electrified": 100 * grp["electrified"].mean(),
            "pop_weighted_ntl": (
                np.average(grp["ntl_mean"], weights=grp["pop_density"])
                if grp["pop_density"].notna().any() else np.nan
            ),
        })
    return pd.DataFrame(rows).set_index("country")
