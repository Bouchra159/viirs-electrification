"""
qgis_render.py
--------------
Headless PyQGIS renderer for the VIIRS Electrification Analysis project.

Produces a complete, journal-ready figure suite (300 DPI PNG + PDF) following
cartographic conventions for Remote Sensing / GIS research publications.

Layer stack on every map (bottom → top):
  1. CartoDB Positron XYZ basemap  — clean light reference background
  2. World country boundaries       — QGIS built-in world_map.gpkg, thin outline
  3. Analysis layer (80 % opacity) — graduated or categorized symbology

Maps produced (29 files × 2 formats = 58 outputs in figures/qgis_maps/)
------------------------------------------------------------------------
Study area:
    00_study_area.{png,pdf}           — 3 regions on world overview map

NTL radiance choropleth (Magma, 7-class quantile):
    ntl_{country}.{png,pdf}           × 3
    panel_ntl.{png,pdf}

LISA spatial autocorrelation (categorized):
    lisa_{country}.{png,pdf}          × 3
    panel_lisa.{png,pdf}

Energy poverty classification (categorized):
    energy_{country}.{png,pdf}        × 3
    panel_energy.{png,pdf}

NTL trend direction 2014–2023 (categorized):
    trend_{country}.{png,pdf}         × 3
    panel_trend.{png,pdf}

SHAP — infrastructure density effect (RdBu diverging, 7-class):
    shap_infra_{country}.{png,pdf}    × 3
    panel_shap_infra.{png,pdf}

GWR local coefficient — infrastructure density (RdBu diverging, 7-class):
    gwr_infra_{country}.{png,pdf}     × 3
    panel_gwr_infra.{png,pdf}

GWR local R² — goodness of fit (YlOrRd sequential, 7-class):
    gwr_r2_{country}.{png,pdf}        × 3
    panel_gwr_r2.{png,pdf}

Usage
-----
    scripts\\run_qgis_render.bat          (recommended — sets OSGeo4W env)

Prerequisites
-------------
    Run scripts/export_qgis_layers.py first to generate data/processed/*.gpkg

Author: Bouchra Daddaoui
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ── OSGeo4W / QGIS environment ────────────────────────────────────────────────
_QGIS_ROOT = Path(r'C:\Program Files\QGIS 3.40.11')

for _p in [
    str(_QGIS_ROOT / 'apps' / 'qgis-ltr' / 'python'),
    str(_QGIS_ROOT / 'apps' / 'qgis-ltr' / 'python' / 'plugins'),
    str(_QGIS_ROOT / 'apps' / 'Python312' / 'Lib' / 'site-packages'),
]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

os.environ.setdefault('QT_QPA_PLATFORM',  'offscreen')
os.environ.setdefault('QGIS_PREFIX_PATH', str(_QGIS_ROOT / 'apps' / 'qgis-ltr'))
os.environ.setdefault('QT_PLUGIN_PATH', os.pathsep.join([
    str(_QGIS_ROOT / 'apps' / 'qgis-ltr' / 'qtplugins'),
    str(_QGIS_ROOT / 'apps' / 'qt5'      / 'plugins'),
]))

from PyQt5.QtGui  import QFont, QColor                                    # noqa: E402
from PyQt5.QtCore import Qt                                                # noqa: E402

from qgis.core import (                                                    # noqa: E402
    QgsApplication,
    QgsProject,
    QgsCoordinateReferenceSystem,
    QgsVectorLayer,
    QgsRasterLayer,
    QgsRectangle,
    QgsFillSymbol,
    QgsLineSymbol,
    QgsSingleSymbolRenderer,
    QgsRendererCategory,
    QgsCategorizedSymbolRenderer,
    QgsGraduatedSymbolRenderer,
    QgsClassificationQuantile,
    QgsClassificationJenks,
    QgsStyle,
    QgsPrintLayout,
    QgsLayoutSize,
    QgsUnitTypes,
    QgsLayoutPoint,
    QgsLayoutItemMap,
    QgsLayoutItemLabel,
    QgsLayoutItemLegend,
    QgsLayoutItemScaleBar,
    QgsLayoutItemPicture,
    QgsLayoutItemPage,
    QgsLayoutExporter,
    QgsTextFormat,
)

# ── Project paths ─────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR     = PROJECT_ROOT / 'data' / 'processed'
FIGURES_DIR  = PROJECT_ROOT / 'figures' / 'qgis_maps'

_WORLD_GPKG = _QGIS_ROOT / 'apps' / 'qgis-ltr' / 'resources' / 'data' / 'world_map.gpkg'

_BASEMAP_URI = (
    'type=xyz'
    '&url=https://basemaps.cartocdn.com/light_all/{z}/{x}/{y}.png'
    '&zmax=19&zmin=0'
    '&crs=EPSG3857'
)

# ── Colour palettes — categorized maps ────────────────────────────────────────
LISA_COLOURS: Dict[str, str] = {
    'High-High':       '#d7191c',
    'Low-Low':         '#2c7bb6',
    'High-Low':        '#fdae61',
    'Low-High':        '#abd9e9',
    'Not significant': '#eeeeee',
}
ENERGY_COLOURS: Dict[str, str] = {
    'Electrified':     '#1a78c2',
    'Energy poor':     '#d62728',
    'Low pop / unlit': '#f5c518',
}
TREND_COLOURS: Dict[str, str] = {
    'Increasing':  '#1a9641',
    'Stable':      '#ffffbf',
    'Decreasing':  '#d7191c',
}

# ── Country spatial extents (EPSG:4326) ───────────────────────────────────────
COUNTRY_EXTENTS: Dict[str, QgsRectangle] = {
    'brazil':  QgsRectangle(-48.0, -23.0, -43.0, -18.0),
    'china':   QgsRectangle(116.0,  29.0, 122.0,  33.0),
    'morocco': QgsRectangle(-17.1,  20.8,  -1.0,  35.9),
}
COUNTRY_LABELS: Dict[str, str] = {
    'brazil':  'Brazil',
    'china':   'China',
    'morocco': 'Morocco (incl. Western Sahara)',
}
COUNTRIES = ['brazil', 'china', 'morocco']

# World overview extent for study area map
_WORLD_EXTENT = QgsRectangle(-180, -60, 180, 80)

# ── Map type configuration ─────────────────────────────────────────────────────
# renderer: 'categorized' | 'graduated_seq' | 'graduated_div'
MAP_CONFIGS: Dict[str, Dict] = {
    'ntl': {
        'gpkg_tmpl':    'ntl_choropleth_{country}.gpkg',
        'layer_name':   'ntl_choropleth',
        'renderer':     'graduated_seq',
        'field':        'ntl_mean',
        'color_ramp':   'Magma',
        'n_classes':    7,
        'title_tmpl':   'VIIRS NTL Radiance — {label}',
        'panel_title':  'VIIRS Nighttime Light Radiance (annual composite)',
        'legend_title': 'NTL (nW/cm²/sr)',
        'legend_cols':  1,
    },
    'lisa': {
        'gpkg_tmpl':    'lisa_clusters_{country}.gpkg',
        'layer_name':   'lisa_clusters',
        'renderer':     'categorized',
        'field':        'cluster_type',
        'colours':      LISA_COLOURS,
        'title_tmpl':   'LISA Autocorrelation Clusters — {label}',
        'panel_title':  'LISA Spatial Autocorrelation Clusters · VIIRS NTL Radiance',
        'legend_title': 'LISA Cluster',
        'legend_cols':  3,
    },
    'energy': {
        'gpkg_tmpl':    'energy_poverty_{country}.gpkg',
        'layer_name':   'energy_poverty',
        'renderer':     'categorized',
        'field':        'energy_class',
        'colours':      ENERGY_COLOURS,
        'title_tmpl':   'Energy Poverty Classification — {label}',
        'panel_title':  'Energy Poverty Classification · VIIRS NTL Radiance',
        'legend_title': 'Energy Class',
        'legend_cols':  3,
    },
    'trend': {
        'gpkg_tmpl':    'ntl_trend_{country}.gpkg',
        'layer_name':   'ntl_trend',
        'renderer':     'categorized',
        'field':        'trend_dir',
        'colours':      TREND_COLOURS,
        'title_tmpl':   'NTL Trend Direction 2014–2023 — {label}',
        'panel_title':  'NTL Radiance Trend Direction 2014–2023 (Theil-Sen slope)',
        'legend_title': 'Trend',
        'legend_cols':  3,
    },
    'shap_infra': {
        'gpkg_tmpl':    'shap_values_{country}.gpkg',
        'layer_name':   'shap_values',
        'renderer':     'graduated_div',
        'field':        'shap_infra_density',
        'color_ramp':   'RdBu',
        'n_classes':    7,
        'title_tmpl':   'SHAP Effect: Infrastructure Density — {label}',
        'panel_title':  'SHAP Feature Effect: Infrastructure Density (XGBoost)',
        'legend_title': 'SHAP value',
        'legend_cols':  1,
    },
    'gwr_infra': {
        'gpkg_tmpl':    'gwr_coefficients_{country}.gpkg',
        'layer_name':   'gwr_coefficients',
        'renderer':     'graduated_div',
        'field':        'gwr_infra_density',
        'color_ramp':   'RdBu',
        'n_classes':    7,
        'title_tmpl':   'GWR Coefficient: Infrastructure Density — {label}',
        'panel_title':  'GWR Local Coefficient: Infrastructure Density Effect on NTL',
        'legend_title': 'Local β',
        'legend_cols':  1,
    },
    'gwr_r2': {
        'gpkg_tmpl':    'gwr_coefficients_{country}.gpkg',
        'layer_name':   'gwr_coefficients',
        'renderer':     'graduated_seq',
        'field':        'gwr_local_r2',
        'color_ramp':   'YlOrRd',
        'n_classes':    5,
        'title_tmpl':   'GWR Local R² — {label}',
        'panel_title':  'GWR Goodness of Fit (Local R²) — Spatial Model Performance',
        'legend_title': 'Local R²',
        'legend_cols':  1,
    },
}

SOURCE_TEXT = (
    'Data: VIIRS DNB (NASA/NOAA) · GADM v4.1 · Basemap: © CartoDB / OpenStreetMap contributors'
    '  |  Analysis: Bouchra Daddaoui'
)
DPI = 300

# ── QGIS application lifecycle ────────────────────────────────────────────────
_qgs_app: Optional[QgsApplication] = None


def init_qgis() -> QgsApplication:
    """Start a headless QgsApplication instance."""
    global _qgs_app
    _qgs_app = QgsApplication([], False)
    _qgs_app.initQgis()
    return _qgs_app


def exit_qgis() -> None:
    if _qgs_app is not None:
        _qgs_app.exitQgis()


# ── Background layer builders ─────────────────────────────────────────────────

def make_basemap_layer() -> Optional[QgsRasterLayer]:
    """Load CartoDB Positron as an XYZ tile raster layer."""
    layer = QgsRasterLayer(_BASEMAP_URI, 'CartoDB Positron', 'wms')
    if not layer.isValid():
        print('  WARNING: basemap unavailable — rendering without background.')
        return None
    return layer


def make_boundaries_layer() -> Optional[QgsVectorLayer]:
    """
    Load world country outlines from QGIS built-in world_map.gpkg.
    Styled as thin dark grey lines (no fill) for geographic context.
    """
    if not _WORLD_GPKG.exists():
        return None
    layer = QgsVectorLayer(f'{_WORLD_GPKG}|layername=countries',
                           'Country boundaries', 'ogr')
    if not layer.isValid():
        layer = QgsVectorLayer(str(_WORLD_GPKG), 'Country boundaries', 'ogr')
    if not layer.isValid():
        return None
    symbol = QgsLineSymbol.createSimple({'color': '#444444', 'width': '0.4'})
    layer.setRenderer(QgsSingleSymbolRenderer(symbol))
    layer.setOpacity(0.7)
    return layer


# ── Renderer builders ─────────────────────────────────────────────────────────

def make_categorized_renderer(
    field: str,
    colour_map: Dict[str, str],
) -> QgsCategorizedSymbolRenderer:
    """
    Categorized renderer from a {value → hex} palette.
    Fills at 80% opacity; white 0.3pt outlines separate adjacent polygons.
    """
    categories = []
    for value, hex_col in colour_map.items():
        c = QColor(hex_col)
        c.setAlpha(204)                      # 80 % opacity
        symbol = QgsFillSymbol.createSimple({
            'color':         c.name(QColor.HexArgb),
            'outline_color': '#ffffff',
            'outline_width': '0.3',
        })
        categories.append(QgsRendererCategory(value, symbol, value))
    return QgsCategorizedSymbolRenderer(field, categories)


def make_graduated_renderer(
    layer: QgsVectorLayer,
    field: str,
    color_ramp_name: str,
    n_classes: int = 7,
    diverging: bool = False,
) -> QgsGraduatedSymbolRenderer:
    """
    Quantile-classified graduated renderer using a named QGIS colour ramp.

    For diverging ramps (SHAP, GWR coefficients) Jenks natural breaks are
    used instead of quantiles to better capture the zero-crossing structure.
    Uses 80% opacity fills consistent with all other layers.
    """
    style = QgsStyle.defaultStyle()
    ramp  = style.colorRamp(color_ramp_name)

    if ramp is None:
        # Fallback: 'Reds' always ships with QGIS
        ramp = style.colorRamp('Reds')

    renderer = QgsGraduatedSymbolRenderer(field)

    if diverging:
        renderer.setClassificationMethod(QgsClassificationJenks())
    else:
        renderer.setClassificationMethod(QgsClassificationQuantile())

    renderer.updateClasses(layer, n_classes)
    renderer.updateColorRamp(ramp)

    # Apply 80% opacity and white outline to every generated symbol
    for range_item in renderer.ranges():
        sym = range_item.symbol().clone()
        for i in range(sym.symbolLayerCount()):
            sl = sym.symbolLayer(i)
            c = sl.fillColor()
            c.setAlpha(204)
            sl.setFillColor(c)
            sl.setStrokeColor(QColor('#ffffff'))
            sl.setStrokeWidth(0.3)
        range_item.setSymbol(sym)

    return renderer


def load_styled_layer(
    gpkg: Path,
    layer_name: str,
    display_name: str,
    cfg: Dict,
) -> QgsVectorLayer:
    """
    Load a GeoPackage vector layer and apply the renderer described by *cfg*.

    Raises
    ------
    FileNotFoundError / RuntimeError on load failure.
    """
    if not gpkg.exists():
        raise FileNotFoundError(
            f'GeoPackage not found: {gpkg}\n'
            '  → Run scripts/export_qgis_layers.py first.'
        )
    layer = QgsVectorLayer(f'{gpkg}|layername={layer_name}', display_name, 'ogr')
    if not layer.isValid():
        raise RuntimeError(f'Cannot load "{layer_name}" from {gpkg.name}.')

    rtype = cfg.get('renderer', 'categorized')
    if rtype == 'categorized':
        renderer = make_categorized_renderer(cfg['field'], cfg['colours'])
    elif rtype == 'graduated_div':
        renderer = make_graduated_renderer(
            layer, cfg['field'], cfg['color_ramp'],
            cfg.get('n_classes', 7), diverging=True,
        )
    else:  # graduated_seq
        renderer = make_graduated_renderer(
            layer, cfg['field'], cfg['color_ramp'],
            cfg.get('n_classes', 7), diverging=False,
        )

    layer.setRenderer(renderer)
    return layer


# ── Layout geometry helpers ───────────────────────────────────────────────────

def _sz(w: float, h: float) -> QgsLayoutSize:
    return QgsLayoutSize(w, h, QgsUnitTypes.LayoutMillimeters)


def _pt(x: float, y: float) -> QgsLayoutPoint:
    return QgsLayoutPoint(x, y, QgsUnitTypes.LayoutMillimeters)


def _find_north_arrow_svg() -> Optional[str]:
    svg_root = _QGIS_ROOT / 'apps' / 'qgis-ltr' / 'svg'
    for name in ('NorthArrow_04.svg', 'NorthArrow_02.svg', 'NorthArrow_01.svg'):
        for hit in svg_root.rglob(name):
            return str(hit)
    return None


def _add_label(
    layout: QgsPrintLayout,
    text: str,
    x: float, y: float, w: float, h: float,
    font_size: int = 9, bold: bool = False,
    align: Qt.AlignmentFlag = Qt.AlignLeft,
) -> QgsLayoutItemLabel:
    item = QgsLayoutItemLabel(layout)
    item.setText(text)
    tf = QgsTextFormat()
    tf.setFont(QFont('Noto Sans', font_size,
                     QFont.Bold if bold else QFont.Normal))
    tf.setSize(font_size)
    item.setTextFormat(tf)
    try:
        item.setHAlign(align)
    except AttributeError:
        pass
    item.attemptMove(_pt(x, y))
    item.attemptResize(_sz(w, h))
    layout.addLayoutItem(item)
    return item


def _add_north_arrow(layout: QgsPrintLayout,
                     x: float, y: float, size: float = 12.0) -> None:
    svg = _find_north_arrow_svg()
    if svg is None:
        return
    item = QgsLayoutItemPicture(layout)
    item.setPicturePath(svg)
    item.attemptMove(_pt(x, y))
    item.attemptResize(_sz(size, size * 1.3))
    layout.addLayoutItem(item)


def _add_scalebar(
    layout: QgsPrintLayout,
    map_item: QgsLayoutItemMap,
    x: float, y: float, w: float, h: float,
    segment_km: float = 100.0, n_segments: int = 3,
) -> None:
    bar = QgsLayoutItemScaleBar(layout)
    bar.setLinkedMap(map_item)
    bar.setStyle('Single Box')
    try:
        from qgis.core import Qgis
        bar.setUnits(Qgis.DistanceUnit.Kilometers)
    except AttributeError:
        bar.setUnits(QgsUnitTypes.DistanceKilometers)
    bar.setUnitsPerSegment(segment_km)
    bar.setNumberOfSegments(n_segments)
    bar.setNumberOfSegmentsLeft(0)
    bar.attemptMove(_pt(x, y))
    bar.attemptResize(_sz(w, h))
    layout.addLayoutItem(bar)


def _add_legend(
    layout: QgsPrintLayout,
    map_item: QgsLayoutItemMap,
    data_layer: QgsVectorLayer,
    x: float, y: float, w: float, h: float,
    title: str = '',
    n_cols: int = 1,
) -> QgsLayoutItemLegend:
    """Legend showing only the analysis data layer."""
    legend = QgsLayoutItemLegend(layout)
    legend.setLinkedMap(map_item)
    legend.setAutoUpdateModel(False)
    root = legend.model().rootGroup()
    root.clear()
    root.addLayer(data_layer)
    legend.setTitle(title)
    legend.setColumnCount(n_cols)
    try:
        legend.setEqualColumnWidth(True)
    except AttributeError:
        pass
    legend.attemptMove(_pt(x, y))
    legend.attemptResize(_sz(w, h))
    layout.addLayoutItem(legend)
    return legend


def _make_map_item(
    layout: QgsPrintLayout,
    layers: List,
    extent: QgsRectangle,
    x: float, y: float, w: float, h: float,
) -> QgsLayoutItemMap:
    """Map item with given layer stack (index 0 rendered on top)."""
    item = QgsLayoutItemMap(layout)
    item.attemptMove(_pt(x, y))
    item.attemptResize(_sz(w, h))
    item.setLayers([l for l in layers if l is not None])
    item.setExtent(extent)
    item.setFrameEnabled(True)
    layout.addLayoutItem(item)
    return item


# ── Study area overview layout ────────────────────────────────────────────────

def build_study_area_layout(
    project: QgsProject,
    boundaries: Optional[QgsVectorLayer],
    basemap: Optional[QgsRasterLayer],
) -> QgsPrintLayout:
    """
    A4-landscape study area overview map showing the 3 research regions
    on a world base map with labelled bounding box insets.

    This is Figure 1 in a typical journal paper — demonstrates spatial
    context before any analysis results are shown.
    """
    import geopandas as gpd
    from shapely.geometry import box

    # Build a highlight layer for the 3 study bboxes
    records = []
    for country, ext in COUNTRY_EXTENTS.items():
        geom = box(ext.xMinimum(), ext.yMinimum(),
                   ext.xMaximum(), ext.yMaximum())
        records.append({'country': COUNTRY_LABELS[country], 'geometry': geom})

    import tempfile, json
    gdf_boxes = gpd.GeoDataFrame(records, crs='EPSG:4326')
    tmp = tempfile.mktemp(suffix='.gpkg')
    gdf_boxes.to_file(tmp, driver='GPKG', layer='study_areas')

    highlight = QgsVectorLayer(f'{tmp}|layername=study_areas',
                               'Study areas', 'ogr')
    # Red outline, no fill — clearly delineates the study region
    symbol = QgsFillSymbol.createSimple({
        'color':           '255,50,50,0',     # fully transparent fill
        'outline_color':   '#e31a1c',
        'outline_width':   '1.2',
        'outline_style':   'solid',
    })
    highlight.setRenderer(QgsSingleSymbolRenderer(symbol))

    layout = QgsPrintLayout(project)
    layout.initializeDefaults()
    layout.pageCollection().page(0).setPageSize(
        'A4', QgsLayoutItemPage.Landscape
    )

    _add_label(layout,
               'Study Area Overview — VIIRS NTL Electrification Analysis\n'
               'Brazil · China · Morocco (incl. Western Sahara)',
               15, 6, 267, 14, font_size=12, bold=True)

    # World map item — wide enough to show all 3 study regions
    layers = [l for l in [highlight, boundaries, basemap] if l is not None]
    map_item = _make_map_item(layout, layers, _WORLD_EXTENT, 15, 24, 267, 162)

    # Country labels on the map (positioned at bbox centroids)
    label_positions = {
        'brazil':  (58, 88),   # mm from left/top of page
        'china':   (168, 58),
        'morocco': (32, 58),
    }
    for country, (lx, ly) in label_positions.items():
        _add_label(layout, COUNTRY_LABELS[country].split(' ')[0],
                   lx, ly, 40, 7, font_size=8, bold=True)

    _add_north_arrow(layout, 272, 24, size=11)
    _add_scalebar(layout, map_item, 15, 190, 100, 8,
                  segment_km=2000, n_segments=3)
    _add_label(layout, SOURCE_TEXT, 15, 200, 267, 7, font_size=7)

    project.addMapLayer(highlight, False)
    return layout, highlight


# ── Single-country A4 layout ──────────────────────────────────────────────────

def build_single_layout(
    project: QgsProject,
    data_layer: QgsVectorLayer,
    boundaries: Optional[QgsVectorLayer],
    basemap: Optional[QgsRasterLayer],
    title: str,
    extent: QgsRectangle,
    legend_title: str = '',
) -> QgsPrintLayout:
    """
    A4-landscape layout for one country and one map type.

    Page geometry (297 × 210 mm):
      Title     (15, 6,   267, 12)
      Map       (15, 22,  196, 148)
      N-arrow   (193, 22,  12,  16)
      Legend    (216, 22,  70, 130)
      Scale bar (15, 176,  90,   8)
      Source    (15, 191, 267,   7)
    """
    layout = QgsPrintLayout(project)
    layout.initializeDefaults()
    layout.pageCollection().page(0).setPageSize(
        'A4', QgsLayoutItemPage.Landscape
    )

    _add_label(layout, title, 15, 6, 267, 12, font_size=12, bold=True)

    layers = [l for l in [data_layer, boundaries, basemap] if l is not None]
    map_item = _make_map_item(layout, layers, extent, 15, 22, 196, 148)

    _add_north_arrow(layout, 193, 22, size=12)
    _add_legend(layout, map_item, data_layer, 216, 22, 70, 130,
                title=legend_title, n_cols=1)
    _add_scalebar(layout, map_item, 15, 176, 90, 8,
                  segment_km=100, n_segments=3)
    _add_label(layout, SOURCE_TEXT, 15, 191, 267, 7, font_size=7)

    return layout


# ── Three-country A3 panel layout ─────────────────────────────────────────────

def build_panel_layout(
    project: QgsProject,
    data_layers: List[QgsVectorLayer],
    boundaries: Optional[QgsVectorLayer],
    basemap: Optional[QgsRasterLayer],
    panel_title: str,
    extents: List[QgsRectangle],
    legend_title: str = '',
    n_legend_cols: int = 3,
) -> QgsPrintLayout:
    """
    A3-landscape panel (420 × 297 mm) with one column per country.
    Shared legend at the bottom; per-column scale bars and north arrows.
    """
    COL_X = [10.0, 150.0, 290.0]
    COL_W = 120.0
    MAP_Y = 30.0
    MAP_H = 215.0

    layout = QgsPrintLayout(project)
    layout.initializeDefaults()
    layout.pageCollection().page(0).setPageSize(
        'A3', QgsLayoutItemPage.Landscape
    )

    _add_label(layout, panel_title, 10, 5, 400, 12,
               font_size=13, bold=True, align=Qt.AlignHCenter)

    map_items: List[QgsLayoutItemMap] = []
    for data_layer, country_key, extent, x0 in zip(
            data_layers, COUNTRIES, extents, COL_X):

        _add_label(layout, COUNTRY_LABELS[country_key],
                   x0, 20, COL_W, 8, font_size=10, bold=True,
                   align=Qt.AlignHCenter)

        layers = [l for l in [data_layer, boundaries, basemap] if l is not None]
        m = _make_map_item(layout, layers, extent, x0, MAP_Y, COL_W, MAP_H)
        map_items.append(m)

        _add_north_arrow(layout, x0 + COL_W - 13, MAP_Y, size=10)
        _add_scalebar(layout, m, x0, MAP_Y + MAP_H + 2, COL_W, 7,
                      segment_km=50, n_segments=2)

    _add_legend(layout, map_items[0], data_layers[0],
                10, 258, 400, 26, title=legend_title,
                n_cols=n_legend_cols)
    _add_label(layout, SOURCE_TEXT, 10, 287, 400, 7, font_size=7)
    return layout


# ── Export ────────────────────────────────────────────────────────────────────

def export_layout(layout: QgsPrintLayout, stem: Path, dpi: int = DPI) -> None:
    """Export layout to PNG and PDF at *dpi* resolution."""
    stem.parent.mkdir(parents=True, exist_ok=True)
    exporter = QgsLayoutExporter(layout)

    img_cfg = QgsLayoutExporter.ImageExportSettings()
    img_cfg.dpi = dpi
    res = exporter.exportToImage(str(stem.with_suffix('.png')), img_cfg)
    ok = res == QgsLayoutExporter.Success
    print(f'    {"✓" if ok else "✗"}  {stem.with_suffix(".png").name}'
          + (f'  (code {res})' if not ok else ''))

    pdf_cfg = QgsLayoutExporter.PdfExportSettings()
    res = exporter.exportToPdf(str(stem.with_suffix('.pdf')), pdf_cfg)
    ok = res == QgsLayoutExporter.Success
    print(f'    {"✓" if ok else "✗"}  {stem.with_suffix(".pdf").name}'
          + (f'  (code {res})' if not ok else ''))


# ── Render orchestration ──────────────────────────────────────────────────────

def render_all() -> None:
    """
    Render the complete publication map suite:
      - 1 study area overview (A4)
      - 7 map types × (3 individual A4 + 1 panel A3) = 28 additional maps
    """
    project = QgsProject.instance()
    project.setCrs(QgsCoordinateReferenceSystem('EPSG:4326'))

    print('Loading shared background layers...')
    basemap    = make_basemap_layer()
    boundaries = make_boundaries_layer()

    if basemap:
        project.addMapLayer(basemap, False)
        print('  ✓  CartoDB Positron basemap')
    if boundaries:
        project.addMapLayer(boundaries, False)
        print('  ✓  World country boundaries')

    # ── 00: Study area overview ───────────────────────────────────────────────
    print('\n── STUDY AREA OVERVIEW ──────────────────────────────────────')
    try:
        layout, highlight_layer = build_study_area_layout(
            project, boundaries, basemap
        )
        export_layout(layout, FIGURES_DIR / '00_study_area')
        del layout
        project.removeMapLayer(highlight_layer.id())
    except Exception as e:
        print(f'  SKIP study area: {e}')

    # ── Per-type maps ─────────────────────────────────────────────────────────
    for map_key, cfg in MAP_CONFIGS.items():
        print(f'\n── {map_key.upper()} ──────────────────────────────────────')

        panel_layers:  List[QgsVectorLayer] = []
        panel_extents: List[QgsRectangle]   = []
        layer_ids:     List[str]            = []

        for country in COUNTRIES:
            gpkg  = DATA_DIR / cfg['gpkg_tmpl'].format(country=country)
            label = COUNTRY_LABELS[country]
            title = cfg['title_tmpl'].format(label=label)

            try:
                layer = load_styled_layer(
                    gpkg, cfg['layer_name'],
                    f'{map_key}_{country}', cfg,
                )
            except (FileNotFoundError, RuntimeError) as err:
                print(f'  SKIP {country}: {err}')
                continue

            project.addMapLayer(layer, False)
            layer_ids.append(layer.id())

            extent = COUNTRY_EXTENTS[country]
            layout = build_single_layout(
                project, layer, boundaries, basemap, title, extent,
                legend_title=cfg.get('legend_title', ''),
            )
            export_layout(layout, FIGURES_DIR / f'{map_key}_{country}')
            del layout

            panel_layers.append(layer)
            panel_extents.append(extent)

        if len(panel_layers) == 3:
            panel = build_panel_layout(
                project, panel_layers, boundaries, basemap,
                cfg['panel_title'], panel_extents,
                legend_title=cfg.get('legend_title', ''),
                n_legend_cols=cfg['legend_cols'],
            )
            export_layout(panel, FIGURES_DIR / f'panel_{map_key}')
            del panel

        for lid in layer_ids:
            project.removeMapLayer(lid)

    if basemap:
        project.removeMapLayer(basemap.id())
    if boundaries:
        project.removeMapLayer(boundaries.id())


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    print('━' * 62)
    print(' VIIRS Electrification — Headless QGIS Publication Map Suite')
    print(f' QGIS : {_QGIS_ROOT.name}')
    print(f' Out  : {FIGURES_DIR}')
    print('━' * 62)

    init_qgis()
    try:
        render_all()
        png_count = len(list(FIGURES_DIR.glob('*.png')))
        print(f'\n{"━" * 62}')
        print(f' Done — {png_count} PNG maps in figures/qgis_maps/')
        print('━' * 62)
    finally:
        exit_qgis()


if __name__ == '__main__':
    main()
