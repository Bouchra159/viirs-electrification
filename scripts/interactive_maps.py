"""
interactive_maps.py
-------------------
Helper functions for building interactive folium maps of VIIRS electrification data.

Author: Bouchra Daddaoui
"""

from __future__ import annotations

import json
import numpy as np
import geopandas as gpd
import folium
from folium import GeoJson, GeoJsonTooltip, LayerControl
import matplotlib
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from pathlib import Path


# ---------------------------------------------------------------------------
# Colour palettes
# ---------------------------------------------------------------------------

LISA_COLOURS = {
    'High-High': '#d7191c',
    'Low-Low':   '#2c7bb6',
    'High-Low':  '#fdae61',
    'Low-High':  '#abd9e9',
    'Not significant': '#cccccc',
}

ENERGY_COLOURS = {
    'Electrified':        '#1a78c2',
    'Energy poor':        '#d62728',
    'Low pop / unlit':    '#f5c518',
}


# ---------------------------------------------------------------------------
# Layer builders
# ---------------------------------------------------------------------------

def choropleth_layer(
    gdf: gpd.GeoDataFrame,
    col: str,
    name: str,
    cmap_name: str = 'plasma',
    vmin: float | None = None,
    vmax: float | None = None,
) -> GeoJson:
    """Return a folium GeoJson layer coloured by a continuous column."""
    vmin = vmin if vmin is not None else float(gdf[col].min())
    vmax = vmax if vmax is not None else float(gdf[col].max())
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cmap = cm.get_cmap(cmap_name)

    def style_fn(feature):
        val = feature['properties'].get(col, 0) or 0
        rgba = cmap(norm(val))
        hex_col = mcolors.to_hex(rgba)
        return {'fillColor': hex_col, 'color': 'white', 'weight': 0.4, 'fillOpacity': 0.8}

    geo_json = gdf[[col, 'tile_id', 'country', 'geometry']].copy()
    return GeoJson(
        data=json.loads(geo_json.to_json()),
        name=name,
        style_function=style_fn,
        tooltip=GeoJsonTooltip(fields=['tile_id', 'country', col],
                               aliases=['Tile', 'Country', col.replace('_', ' ').title()]),
    )


def lisa_layer(gdf: gpd.GeoDataFrame, name: str) -> GeoJson:
    """Return a folium GeoJson layer coloured by LISA cluster label."""
    def style_fn(feature):
        label = feature['properties'].get('cluster_label', 'Not significant')
        colour = LISA_COLOURS.get(label, '#cccccc')
        return {'fillColor': colour, 'color': 'white', 'weight': 0.4, 'fillOpacity': 0.85}

    cols = ['tile_id', 'country', 'cluster_label', 'geometry']
    return GeoJson(
        data=json.loads(gdf[cols].to_json()),
        name=name,
        style_function=style_fn,
        tooltip=GeoJsonTooltip(fields=['tile_id', 'country', 'cluster_label'],
                               aliases=['Tile', 'Country', 'LISA Cluster']),
    )


def energy_poverty_layer(gdf: gpd.GeoDataFrame, name: str) -> GeoJson:
    """Return a folium GeoJson layer coloured by energy poverty class."""
    def style_fn(feature):
        label = feature['properties'].get('energy_class', 'Low pop / unlit')
        colour = ENERGY_COLOURS.get(label, '#cccccc')
        return {'fillColor': colour, 'color': 'white', 'weight': 0.4, 'fillOpacity': 0.85}

    cols = ['tile_id', 'country', 'energy_class', 'geometry']
    return GeoJson(
        data=json.loads(gdf[cols].to_json()),
        name=name,
        style_function=style_fn,
        tooltip=GeoJsonTooltip(fields=['tile_id', 'country', 'energy_class'],
                               aliases=['Tile', 'Country', 'Energy Class']),
    )


# ---------------------------------------------------------------------------
# Map construction helpers
# ---------------------------------------------------------------------------

def _base_map(gdf: gpd.GeoDataFrame) -> folium.Map:
    """Create a folium Map centred on gdf bounds."""
    bounds = gdf.total_bounds  # [minx, miny, maxx, maxy]
    center = [(bounds[1] + bounds[3]) / 2, (bounds[0] + bounds[2]) / 2]
    m = folium.Map(location=center, zoom_start=6,
                   tiles='CartoDB positron')
    return m


def _fit_bounds(m: folium.Map, gdf: gpd.GeoDataFrame) -> None:
    bounds = gdf.total_bounds
    m.fit_bounds([[bounds[1], bounds[0]], [bounds[3], bounds[2]]])


def _add_legend_html(m: folium.Map, colours: dict[str, str], title: str) -> None:
    """Inject a simple HTML legend into the map."""
    items = ''.join(
        f'<li><span style="background:{c};width:14px;height:14px;'
        f'display:inline-block;margin-right:6px;border-radius:2px;"></span>{l}</li>'
        for l, c in colours.items()
    )
    html = f"""
    <div style="position:fixed;bottom:30px;left:30px;z-index:1000;background:white;
                padding:10px 14px;border-radius:6px;box-shadow:2px 2px 6px rgba(0,0,0,0.3);
                font-family:sans-serif;font-size:12px;">
      <b style="font-size:13px;">{title}</b><br>
      <ul style="list-style:none;padding:0;margin:6px 0 0 0;">{items}</ul>
    </div>
    """
    m.get_root().html.add_child(folium.Element(html))


def save_map(m: folium.Map, path: str | Path) -> None:
    """Save a folium map to an HTML file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(path))
    print(f'Saved: {path}')


# ---------------------------------------------------------------------------
# Single-theme single-country maps
# ---------------------------------------------------------------------------

def ntl_map(gdf: gpd.GeoDataFrame, country: str) -> folium.Map:
    m = _base_map(gdf)
    layer = choropleth_layer(gdf, 'ntl_mean', f'NTL — {country}', cmap_name='magma')
    layer.add_to(m)
    _fit_bounds(m, gdf)
    return m


def lisa_map(gdf: gpd.GeoDataFrame, country: str) -> folium.Map:
    m = _base_map(gdf)
    layer = lisa_layer(gdf, f'LISA — {country}')
    layer.add_to(m)
    _add_legend_html(m, LISA_COLOURS, 'LISA Clusters')
    _fit_bounds(m, gdf)
    return m


def energy_map(gdf: gpd.GeoDataFrame, country: str) -> folium.Map:
    m = _base_map(gdf)
    layer = energy_poverty_layer(gdf, f'Energy Poverty — {country}')
    layer.add_to(m)
    _add_legend_html(m, ENERGY_COLOURS, 'Energy Classification')
    _fit_bounds(m, gdf)
    return m


# ---------------------------------------------------------------------------
# Combined multi-country map
# ---------------------------------------------------------------------------

def build_combined_map(gdfs: dict[str, gpd.GeoDataFrame], year: int = 2020) -> folium.Map:
    """
    Build one folium Map with LayerControl showing all countries and all themes.

    Parameters
    ----------
    gdfs : dict mapping country name → GeoDataFrame with cluster_label and energy_class columns.
    year : reference year label shown in layer names (default 2020).

    Returns
    -------
    folium.Map with toggleable layers.
    """
    # World centre
    m = folium.Map(location=[20, 10], zoom_start=2, tiles='CartoDB positron')

    for country, gdf in gdfs.items():
        choropleth_layer(gdf, 'ntl_mean', f'NTL — {country} ({year})', cmap_name='magma').add_to(m)
        lisa_layer(gdf, f'LISA — {country} ({year})').add_to(m)
        energy_poverty_layer(gdf, f'Energy Poverty — {country} ({year})').add_to(m)

    LayerControl(collapsed=False).add_to(m)
    _add_legend_html(m, {**LISA_COLOURS, **ENERGY_COLOURS}, 'Legend')
    return m
