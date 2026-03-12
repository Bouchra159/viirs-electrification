# QGIS Workflow Guide
## Electrification Inequality Analysis — Visual & GIS Layer Production

**Author:** Bouchra Daddaoui

This guide documents the QGIS-based cartographic and spatial analysis workflow used to produce
publication-quality maps and validate raster outputs from the Python + GEE pipeline.

---

## Software Requirements

| Tool | Version | Purpose |
|------|---------|---------|
| QGIS | ≥ 3.28 LTR | Main GIS platform |
| Google Earth Engine Plugin | latest | Direct GEE layer preview |
| QuickMapServices | latest | Basemaps (OSM, ESRI Satellite) |
| QGIS Temporal Controller | built-in | Animated NTL time series |

---

## 1. Loading VIIRS GeoTIFF Outputs

After running `gee_export.js` in the GEE Code Editor, download GeoTIFFs from Google Drive.

**Steps:**
1. `Layer → Add Layer → Add Raster Layer`
2. Select e.g. `Morocco_NTL_2023.tif`
3. Symbology: `Singleband pseudocolor`
   - Min: 0, Max: 50 (nW/cm²/sr)
   - Color ramp: `Magma` (matches VIIRS convention)
4. Repeat for each country and year

**Temporal animation:**
1. Enable `Temporal Controller Panel` (View → Panels)
2. Set each NTL layer's temporal range (Jan 1 – Dec 31 of its year)
3. Press Play to animate electrification growth 2014–2023

---

## 2. Country Boundaries (including Western Sahara)

Morocco's official boundary in this project **includes Western Sahara**
(consistent with Morocco's administrative control and recent international recognition trends).

```
Data source: GADM v4.1
Download: https://gadm.org/download_country.html → Morocco (includes Western Sahara)
Layer: gadm41_MAR_0  (national boundary)
```

**In QGIS:**
1. Load `gadm41_MAR_0.gpkg`
2. Set fill to transparent, outline to black (0.5 pt)
3. Overlay on NTL raster for geographic context

---

## 3. LISA Cluster Maps (from Python pipeline)

The Python scripts (`spatial_analysis.py` + notebook `02_spatial_analysis.ipynb`)
export LISA cluster shapefiles. Load these in QGIS for publication maps.

**Expected files:**
- `data/processed/morocco_lisa_2023.gpkg`
- `data/processed/brazil_lisa_2023.gpkg`
- `data/processed/china_lisa_2023.gpkg`

**Symbology (categorised by `cluster_label`):**

| Cluster | Hex colour | Meaning |
|---------|-----------|---------|
| High-High | `#d7191c` | Electrification hotspot |
| Low-Low | `#2c7bb6` | Electrification coldspot |
| High-Low | `#fdae61` | Isolated bright area |
| Low-High | `#abd9e9` | Isolated dark area |
| Not significant | `#eeeeee` | No spatial clustering |

---

## 4. SHAP Spatial Maps

Export spatial SHAP values as GeoJSON from the ML notebook, then visualise in QGIS.

**Steps:**
1. In `03_ml_shap.ipynb`: at end of "Spatial SHAP" section, run:
   ```python
   all_gdf['shap_infra'] = shap_values[:, FEATURES.index('infra_density')]
   all_gdf.to_file('../data/processed/shap_maps.gpkg', driver='GPKG')
   ```
2. In QGIS: load `shap_maps.gpkg`
3. Symbology: `Graduated` on `shap_infra`
   - Color ramp: `RdBu` (diverging, centred at 0)
   - 7 classes, quantile breaks

---

## 5. Population Overlay (WorldPop)

```
Layer: WorldPop/GP/100m/pop (via GEE plugin or downloaded GeoTIFF)
Resampled to 1 km to match NTL resolution
Blend mode: Multiply (over NTL layer) → highlights populated dark areas (energy poverty)
```

---

## 6. Map Layout for Publication

**Print Layout settings:**
- Page size: A4 landscape
- Scale bar: appropriate to country extent
- North arrow: top-right
- Legend: bottom-left (manual ordering: HH, LL, HL, LH, NS)
- Title font: Noto Sans Bold 14pt
- Body font: Noto Sans 9pt
- DPI export: 300 (for journal submission)

**Layer order (bottom to top):**
1. ESRI Satellite basemap (QuickMapServices)
2. NTL raster (Multiply blend, 70% opacity)
3. LISA cluster polygons (80% opacity)
4. Country/admin boundaries
5. SHAP isolines (optional)
6. Scale bar + North arrow + Legend

---

## 7. Morocco Map — Western Sahara Note

All Morocco maps in this project cover the full territory:
- Northern Morocco: 27°N – 36°N, 1°W – 17°W
- Western Sahara: 21°N – 28°N, 9°W – 17°W
- Combined bbox: `EPSG:4326 [-17.10, 20.76, -1.01, 35.93]`

The two sub-regions are **not administratively separated** in this analysis.
NTL data is extracted for the full combined extent in `gee_export.js`.

---

## 8. Exporting Final Maps

From QGIS Print Layout:
```
Layout → Export as Image → PNG, 300 DPI
Filename: maps/morocco_lisa_2023_final.png
```

For vector output (scalable):
```
Layout → Export as PDF (vector)
```
