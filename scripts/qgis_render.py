"""
qgis_render.py
--------------
Headless PyQGIS renderer for the VIIRS Electrification Analysis project.

Generates publication-quality maps (300 DPI PNG + PDF) from GeoPackage
layers exported by scripts/export_qgis_layers.py.

Layer stack (bottom to top):
  1. CartoDB Positron XYZ basemap     — clean light background
  2. World country boundaries          — from QGIS built-in world_map.gpkg
  3. Analysis layer (80 % opacity)    — LISA / energy poverty / NTL trend tiles

Maps produced
-------------
Individual A4-landscape maps (9 total — one per country × map type):
    figures/qgis_maps/lisa_brazil.{png,pdf}
    figures/qgis_maps/lisa_china.{png,pdf}
    figures/qgis_maps/lisa_morocco.{png,pdf}
    figures/qgis_maps/energy_{country}.{png,pdf}
    figures/qgis_maps/trend_{country}.{png,pdf}

Multi-country panel maps (3 total — A3 landscape):
    figures/qgis_maps/panel_lisa.{png,pdf}
    figures/qgis_maps/panel_energy.{png,pdf}
    figures/qgis_maps/panel_trend.{png,pdf}

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
from typing import Dict, List, Optional

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

# QGIS built-in world boundaries (ships with every QGIS installation)
_WORLD_GPKG = _QGIS_ROOT / 'apps' / 'qgis-ltr' / 'resources' / 'data' / 'world_map.gpkg'

# CartoDB Positron — clean light basemap, standard for research cartography
_BASEMAP_URI = (
    'type=xyz'
    '&url=https://basemaps.cartocdn.com/light_all/{z}/{x}/{y}.png'
    '&zmax=19&zmin=0'
    '&crs=EPSG3857'
)

# ── Cartographic colour palettes ──────────────────────────────────────────────
# Consistent with interactive_maps.py and qgis_workflow.md.

LISA_COLOURS: Dict[str, str] = {
    'High-High':       '#d7191c',   # electrification hotspot
    'Low-Low':         '#2c7bb6',   # electrification coldspot
    'High-Low':        '#fdae61',   # isolated bright tile (outlier)
    'Low-High':        '#abd9e9',   # isolated dark tile (outlier)
    'Not significant': '#eeeeee',   # no significant spatial clustering
}

ENERGY_COLOURS: Dict[str, str] = {
    'Electrified':     '#1a78c2',
    'Energy poor':     '#d62728',
    'Low pop / unlit': '#f5c518',
}

TREND_COLOURS: Dict[str, str] = {
    'Increasing':  '#1a9641',   # NTL rising 2014–2023
    'Stable':      '#ffffbf',   # no significant trend
    'Decreasing':  '#d7191c',   # NTL declining
}

# ── Country spatial extents (EPSG:4326) ───────────────────────────────────────
# Morocco extent covers Western Sahara (south to 20.8°N).
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

MAP_CONFIGS: Dict[str, Dict] = {
    'lisa': {
        'gpkg_tmpl':    'lisa_clusters_{country}.gpkg',
        'layer_name':   'lisa_clusters',
        'style_field':  'cluster_type',
        'colours':      LISA_COLOURS,
        'title_tmpl':   'LISA Autocorrelation Clusters — {label}',
        'panel_title':  'LISA Spatial Autocorrelation Clusters · VIIRS NTL Radiance',
        'legend_cols':  3,
    },
    'energy': {
        'gpkg_tmpl':    'energy_poverty_{country}.gpkg',
        'layer_name':   'energy_poverty',
        'style_field':  'energy_class',
        'colours':      ENERGY_COLOURS,
        'title_tmpl':   'Energy Poverty Classification — {label}',
        'panel_title':  'Energy Poverty Classification · VIIRS NTL Radiance',
        'legend_cols':  3,
    },
    'trend': {
        'gpkg_tmpl':    'ntl_trend_{country}.gpkg',
        'layer_name':   'ntl_trend',
        'style_field':  'trend_dir',
        'colours':      TREND_COLOURS,
        'title_tmpl':   'NTL Radiance Trend Direction 2014–2023 — {label}',
        'panel_title':  'NTL Radiance Trend Direction 2014–2023 · VIIRS',
        'legend_cols':  3,
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
    """Shut down QgsApplication cleanly."""
    if _qgs_app is not None:
        _qgs_app.exitQgis()


# ── Background layer builders ─────────────────────────────────────────────────

def make_basemap_layer() -> Optional[QgsRasterLayer]:
    """
    Load CartoDB Positron as an XYZ tile raster layer.

    Returns None if the layer cannot be loaded (e.g. no internet), in which
    case the map renders without a basemap rather than crashing.
    """
    layer = QgsRasterLayer(_BASEMAP_URI, 'CartoDB Positron', 'wms')
    if not layer.isValid():
        print('  WARNING: basemap not available — rendering without background.')
        return None
    return layer


def make_boundaries_layer() -> Optional[QgsVectorLayer]:
    """
    Load world country outlines from QGIS's built-in world_map.gpkg.

    Styled as thin dark-grey lines with no fill — provides geographic
    context without competing with the analysis layer colours.
    """
    if not _WORLD_GPKG.exists():
        print(f'  WARNING: world_map.gpkg not found at {_WORLD_GPKG}')
        return None

    layer = QgsVectorLayer(
        f'{_WORLD_GPKG}|layername=countries',
        'Country boundaries', 'ogr',
    )
    # Fallback: some QGIS builds use a different layer name inside the GPKG
    if not layer.isValid():
        layer = QgsVectorLayer(str(_WORLD_GPKG), 'Country boundaries', 'ogr')
    if not layer.isValid():
        print('  WARNING: could not load world boundaries layer.')
        return None

    # Transparent fill, thin dark outline — standard reference overlay
    symbol = QgsLineSymbol.createSimple({
        'color':        '#444444',
        'width':        '0.4',
        'penstyle':     'solid',
    })
    layer.setRenderer(QgsSingleSymbolRenderer(symbol))
    layer.setOpacity(0.7)
    return layer


# ── Analysis layer builder ────────────────────────────────────────────────────

def make_categorized_renderer(
    field: str,
    colour_map: Dict[str, str],
) -> QgsCategorizedSymbolRenderer:
    """
    Build a categorized renderer from a {value → hex_colour} mapping.

    Symbols use semi-transparent fills (alpha 200/255) so the basemap
    roads and labels remain faintly visible beneath the analysis tiles.
    White 0.3 pt outlines cleanly separate adjacent polygons.
    """
    categories = []
    for value, hex_col in colour_map.items():
        color = QColor(hex_col)
        color.setAlpha(200)                  # ~78% opacity — basemap shows through
        symbol = QgsFillSymbol.createSimple({
            'color':         color.name(QColor.HexArgb),
            'outline_color': '#ffffff',
            'outline_width': '0.3',
        })
        categories.append(QgsRendererCategory(value, symbol, value))
    return QgsCategorizedSymbolRenderer(field, categories)


def load_styled_layer(
    gpkg: Path,
    layer_name: str,
    display_name: str,
    field: str,
    colour_map: Dict[str, str],
) -> QgsVectorLayer:
    """
    Load a vector layer from a GeoPackage file and apply categorized styling.

    Raises
    ------
    FileNotFoundError
        If the GeoPackage file does not exist on disk.
    RuntimeError
        If QGIS cannot load the specified layer.
    """
    if not gpkg.exists():
        raise FileNotFoundError(
            f'GeoPackage not found: {gpkg}\n'
            '  → Run scripts/export_qgis_layers.py to generate processed data.'
        )
    layer = QgsVectorLayer(f'{gpkg}|layername={layer_name}', display_name, 'ogr')
    if not layer.isValid():
        raise RuntimeError(
            f'Could not load layer "{layer_name}" from {gpkg.name}.'
        )
    layer.setRenderer(make_categorized_renderer(field, colour_map))
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


# ── Layout item factories ─────────────────────────────────────────────────────

def _add_label(
    layout: QgsPrintLayout,
    text: str,
    x: float, y: float, w: float, h: float,
    font_size: int = 9,
    bold: bool = False,
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


def _add_north_arrow(
    layout: QgsPrintLayout,
    x: float, y: float,
    size: float = 12.0,
) -> None:
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
    segment_km: float = 100.0,
    n_segments: int = 3,
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
    n_cols: int = 1,
) -> QgsLayoutItemLegend:
    """
    Add a legend showing only the analysis data layer (not basemap/boundaries).
    """
    legend = QgsLayoutItemLegend(layout)
    legend.setLinkedMap(map_item)
    legend.setAutoUpdateModel(False)

    # Clear auto-populated items and add only the analysis layer
    root = legend.model().rootGroup()
    root.clear()
    root.addLayer(data_layer)

    legend.setTitle('')
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
    """
    Add a map item with the given layer stack and geographic extent.

    Layer order (index 0 = top): data_layer, boundaries, basemap.
    QGIS renders index 0 last (on top), so pass layers top-to-bottom.
    """
    map_item = QgsLayoutItemMap(layout)
    map_item.attemptMove(_pt(x, y))
    map_item.attemptResize(_sz(w, h))
    map_item.setLayers(layers)        # layers[0] drawn on top
    map_item.setExtent(extent)
    map_item.setFrameEnabled(True)
    layout.addLayoutItem(map_item)
    return map_item


# ── Single-country A4 layout ──────────────────────────────────────────────────

def build_single_layout(
    project: QgsProject,
    data_layer: QgsVectorLayer,
    boundaries: Optional[QgsVectorLayer],
    basemap: Optional[QgsRasterLayer],
    title: str,
    extent: QgsRectangle,
) -> QgsPrintLayout:
    """
    Build an A4-landscape QgsPrintLayout for one country and one map type.

    Layer stack (top → bottom): data tiles → country boundaries → basemap.

    Page geometry (297 × 210 mm):

        ┌─────────────────────────────────────────────────┐
        │  Title                                (15, 6)   │
        ├──────────────────────────┬──────────────────────┤
        │  Map  (15, 22, 196×148)  │ [N]  Legend          │
        │                          │      (216, 22)        │
        ├──────────────────────────┴──────────────────────┤
        │  Scale bar (15, 176)     Source note  (15, 191) │
        └─────────────────────────────────────────────────┘
    """
    layout = QgsPrintLayout(project)
    layout.initializeDefaults()
    layout.pageCollection().page(0).setPageSize(
        'A4', QgsLayoutItemPage.Landscape
    )

    # Title
    _add_label(layout, title, 15, 6, 267, 12, font_size=12, bold=True)

    # Build layer stack: data on top, then boundaries, then basemap
    layer_stack = [l for l in [data_layer, boundaries, basemap] if l is not None]
    map_item = _make_map_item(layout, layer_stack, extent, 15, 22, 196, 148)

    # North arrow
    _add_north_arrow(layout, 193, 22, size=12)

    # Legend (data layer only)
    _add_legend(layout, map_item, data_layer, 216, 22, 70, 130, n_cols=1)

    # Scale bar
    _add_scalebar(layout, map_item, 15, 176, 90, 8,
                  segment_km=100, n_segments=3)

    # Attribution
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
    n_legend_cols: int = 3,
) -> QgsPrintLayout:
    """
    Build an A3-landscape panel with one column per country.

    Page geometry (420 × 297 mm):

        ┌──────────────────────────────────────────────────────┐
        │  Panel title (centred)                               │
        ├──────────────┬──────────────┬────────────────────────┤
        │ Brazil       │ China        │ Morocco                │
        │ Map (120×215)│ Map (120×215)│ Map (120×215)          │
        ├──────────────┴──────────────┴────────────────────────┤
        │  Shared legend · Source note                         │
        └──────────────────────────────────────────────────────┘
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

        layer_stack = [l for l in [data_layer, boundaries, basemap] if l is not None]
        map_item = _make_map_item(layout, layer_stack, extent,
                                  x0, MAP_Y, COL_W, MAP_H)
        map_items.append(map_item)

        _add_north_arrow(layout, x0 + COL_W - 13, MAP_Y, size=10)
        _add_scalebar(layout, map_item,
                      x0, MAP_Y + MAP_H + 2, COL_W, 7,
                      segment_km=50, n_segments=2)

    # Shared legend using the first data layer's renderer
    _add_legend(layout, map_items[0], data_layers[0],
                10, 258, 400, 26, n_cols=n_legend_cols)

    _add_label(layout, SOURCE_TEXT, 10, 287, 400, 7, font_size=7)

    return layout


# ── Export ────────────────────────────────────────────────────────────────────

def export_layout(layout: QgsPrintLayout, stem: Path, dpi: int = DPI) -> None:
    """Export *layout* to PNG and PDF at *dpi* resolution."""
    stem.parent.mkdir(parents=True, exist_ok=True)
    exporter = QgsLayoutExporter(layout)

    img_cfg     = QgsLayoutExporter.ImageExportSettings()
    img_cfg.dpi = dpi
    res = exporter.exportToImage(str(stem.with_suffix('.png')), img_cfg)
    print(f'    {"✓" if res == QgsLayoutExporter.Success else "✗"}  '
          f'{stem.with_suffix(".png").name}'
          + ('' if res == QgsLayoutExporter.Success else f' (code {res})'))

    pdf_cfg = QgsLayoutExporter.PdfExportSettings()
    res = exporter.exportToPdf(str(stem.with_suffix('.pdf')), pdf_cfg)
    print(f'    {"✓" if res == QgsLayoutExporter.Success else "✗"}  '
          f'{stem.with_suffix(".pdf").name}'
          + ('' if res == QgsLayoutExporter.Success else f' (code {res})'))


# ── Render orchestration ──────────────────────────────────────────────────────

def render_all() -> None:
    """
    Render all 12 maps (9 individual A4 + 3 panel A3).

    Background layers (basemap + boundaries) are loaded once and reused
    across all map types and countries to keep memory usage low.
    """
    project = QgsProject.instance()
    project.setCrs(QgsCoordinateReferenceSystem('EPSG:4326'))

    # ── Load shared background layers once ───────────────────────────────────
    print('Loading background layers...')
    basemap    = make_basemap_layer()
    boundaries = make_boundaries_layer()

    if basemap:
        project.addMapLayer(basemap, False)
        print('  ✓  CartoDB Positron basemap')
    if boundaries:
        project.addMapLayer(boundaries, False)
        print('  ✓  World country boundaries')

    # ── Render each map type ─────────────────────────────────────────────────
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
                    f'{map_key}_{country}',
                    cfg['style_field'], cfg['colours'],
                )
            except (FileNotFoundError, RuntimeError) as err:
                print(f'  SKIP {country}: {err}')
                continue

            project.addMapLayer(layer, False)
            layer_ids.append(layer.id())

            extent = COUNTRY_EXTENTS[country]
            layout = build_single_layout(
                project, layer, boundaries, basemap, title, extent
            )
            export_layout(layout, FIGURES_DIR / f'{map_key}_{country}')
            del layout

            panel_layers.append(layer)
            panel_extents.append(extent)

        if len(panel_layers) == 3:
            panel = build_panel_layout(
                project, panel_layers, boundaries, basemap,
                cfg['panel_title'], panel_extents,
                n_legend_cols=cfg['legend_cols'],
            )
            export_layout(panel, FIGURES_DIR / f'panel_{map_key}')
            del panel

        for lid in layer_ids:
            project.removeMapLayer(lid)

    # Clean up shared layers
    if basemap:
        project.removeMapLayer(basemap.id())
    if boundaries:
        project.removeMapLayer(boundaries.id())


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    """Initialise QGIS headlessly, render all maps, and exit cleanly."""
    print('━' * 60)
    print(' VIIRS Electrification — Headless QGIS Map Renderer')
    print(f' QGIS : {_QGIS_ROOT.name}')
    print(f' Out  : {FIGURES_DIR}')
    print('━' * 60)

    init_qgis()
    try:
        render_all()
        png_count = len(list(FIGURES_DIR.glob('*.png')))
        print(f'\n{"━" * 60}')
        print(f' Done — {png_count} PNG maps in figures/qgis_maps/')
        print('━' * 60)
    finally:
        exit_qgis()


if __name__ == '__main__':
    main()
