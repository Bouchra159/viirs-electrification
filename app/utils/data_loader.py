"""
data_loader.py
--------------
Cached synthetic data generation for the Streamlit app.
Mirrors the exact seeds/params used in notebooks 01–12 so figures
match across notebooks and app pages.
"""

from __future__ import annotations

import sys
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import box
from pathlib import Path
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from sdg7_tracker import (
    access_rate_timeseries,
    theil_sen_projection,
    sdg7_gap_analysis,
    electrification_index_timeseries,
    urban_rural_disaggregation,
    NTL_ELECTRIFICATION_THRESHOLD,
)
from spatial_analysis import build_knn_weights, local_moran

YEARS = list(range(2015, 2024))

COUNTRIES_CFG = [
    dict(country="Brazil",  bbox=(-48,-23,-43,-18), ntl_mean=12.5, ntl_std=8.0,  pop_mean=200, seed=1, annual_gain=0.35),
    dict(country="China",   bbox=(116,29,122,33),   ntl_mean=28.3, ntl_std=14.0, pop_mean=900, seed=2, annual_gain=0.15),
    dict(country="Morocco", bbox=(-17.1,20.8,-1.0,35.9), ntl_mean=8.1, ntl_std=5.5, pop_mean=120, seed=3, annual_gain=0.55),
]

COUNTRY_COLORS = {"Brazil": "#2196F3", "China": "#FF5722", "Morocco": "#4CAF50"}


@st.cache_data(show_spinner=False)
def load_panel() -> gpd.GeoDataFrame:
    """Synthetic VIIRS NTL panel 2015–2023."""
    records = []
    for cfg in COUNTRIES_CFG:
        rng = np.random.default_rng(cfg["seed"])
        minx, miny, maxx, maxy = cfg["bbox"]
        cols = rows = 10
        xs = np.linspace(minx, maxx, cols + 1)
        ys = np.linspace(miny, maxy, rows + 1)
        geoms = [box(xs[j], ys[i], xs[j+1], ys[i+1]) for i in range(rows) for j in range(cols)]
        n = len(geoms)
        centroids = np.array([[g.centroid.x, g.centroid.y] for g in geoms])
        dists = np.linalg.norm(centroids[:, None] - centroids[None, :], axis=-1)
        cov = cfg["ntl_std"] ** 2 * np.exp(-dists / (0.3 * (maxx - minx)))
        base_ntl = rng.multivariate_normal(np.full(n, cfg["ntl_mean"]), cov)
        base_ntl = np.clip(base_ntl, 0, None)
        pop = rng.lognormal(mean=np.log(cfg["pop_mean"]), sigma=0.8, size=n)
        for year in YEARS:
            t = year - 2015
            noise = rng.normal(0, 0.3, n)
            ntl = np.clip(base_ntl + cfg["annual_gain"] * t + noise, 0, None)
            for i, geom in enumerate(geoms):
                records.append({
                    "country": cfg["country"],
                    "year": year,
                    "tile_id": f"{cfg['country'][:3].upper()}_{i:03d}",
                    "ntl_mean": ntl[i],
                    "pop": pop[i],
                    "geometry": geom,
                })
    return gpd.GeoDataFrame(records, crs="EPSG:4326")


@st.cache_data(show_spinner=False)
def load_access_rates(_panel=None) -> pd.DataFrame:
    if _panel is None:
        _panel = load_panel()
    return access_rate_timeseries(_panel, ntl_col="ntl_mean", pop_col="pop",
                                   threshold=NTL_ELECTRIFICATION_THRESHOLD)


@st.cache_data(show_spinner=False)
def load_projections(_ar_df=None) -> dict:
    if _ar_df is None:
        _ar_df = load_access_rates()
    results = {}
    for country in _ar_df.country.unique():
        sub = _ar_df[_ar_df.country == country].sort_values("year")
        results[country] = theil_sen_projection(
            sub["year"].values, sub["access_rate"].values,
            target_year=2030, n_bootstrap=2000, ci=0.95,
        )
    return results


@st.cache_data(show_spinner=False)
def load_gap_analysis(_ar_df=None) -> pd.DataFrame:
    if _ar_df is None:
        _ar_df = load_access_rates()
    return sdg7_gap_analysis(_ar_df)


@st.cache_data(show_spinner=False)
def load_ei_timeseries(_panel=None) -> pd.DataFrame:
    if _panel is None:
        _panel = load_panel()
    return electrification_index_timeseries(_panel, ntl_col="ntl_mean", pop_col="pop")


@st.cache_data(show_spinner=False)
def load_snapshot_2023(_panel=None) -> gpd.GeoDataFrame:
    if _panel is None:
        _panel = load_panel()
    return _panel[_panel.year == 2023].copy()


@st.cache_data(show_spinner=False)
def load_lisa_clusters(_snap=None) -> gpd.GeoDataFrame:
    if _snap is None:
        _snap = load_snapshot_2023()
    gdfs = []
    for country in _snap.country.unique():
        gdf = _snap[_snap.country == country].copy().reset_index(drop=True)
        w = build_knn_weights(gdf, k=min(8, len(gdf) - 1))
        lisa_df = local_moran(gdf["ntl_mean"].values, w, significance=0.05)
        gdf["cluster_label"] = lisa_df["cluster_label"].values
        gdfs.append(gdf)
    return pd.concat(gdfs, ignore_index=True)
