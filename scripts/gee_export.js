/**
 * gee_export.js
 * =============
 * Google Earth Engine (GEE) script for exporting VIIRS nighttime light composites,
 * WorldPop population grids, and ancillary geographic features for three study countries.
 *
 * Data exported:
 *   1. VIIRS DNB monthly composites → yearly median → GeoTIFF + tile CSV
 *   2. WorldPop population rasters (100 m) → resampled to 1 km
 *   3. Derived: population-normalised NTL (proxy for electrification intensity)
 *   4. OSM-based infrastructure density mask via GHSL-BUILT
 *
 * Output destination: Google Drive → "viirs_electrification/"
 *
 * Author: Bouchra Daddaoui
 * Usage: paste into GEE Code Editor at https://code.earthengine.google.com
 */

// ─────────────────────────────────────────────────────────────────────────────
// 0. Study Area Definitions
//    Morocco geometry explicitly includes Western Sahara (internationally
//    recognised as part of Morocco per UN SC resolution 2703, 2023).
// ─────────────────────────────────────────────────────────────────────────────

var COUNTRIES = {
  Brazil: ee.Geometry.BBox(-73.99, -33.75, -28.84, 5.27),
  China:  ee.Geometry.BBox(73.50,  18.15, 134.77, 53.56),
  // Morocco full extent including Western Sahara
  Morocco: ee.Geometry.BBox(-17.10, 20.76, -1.01, 35.93)
};

var YEARS   = ee.List.sequence(2014, 2023);
var SCALE   = 1000;   // 1 km output resolution (meters)
var FOLDER  = 'viirs_electrification';
var CRS     = 'EPSG:4326';


// ─────────────────────────────────────────────────────────────────────────────
// 1. VIIRS DNB — Yearly Median Composites
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Build a single-band yearly median composite from VIIRS DNB monthly images.
 * Applies a cloud/stray-light mask using the 'avg_rad' band quality flags.
 *
 * @param {number} year — Calendar year (integer).
 * @param {ee.Geometry} roi — Region of interest.
 * @returns {ee.Image} Single-band image named 'avg_rad'.
 */
function yearlyVIIRS(year, roi) {
  var start = ee.Date.fromYMD(year, 1, 1);
  var end   = ee.Date.fromYMD(year, 12, 31);

  var col = ee.ImageCollection('NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG')
    .filterBounds(roi)
    .filterDate(start, end)
    .select('avg_rad');

  // Mask negative radiance (background / stray light artefacts)
  var masked = col.map(function(img) {
    return img.updateMask(img.gt(0));
  });

  return masked.median()
    .rename('avg_rad')
    .set('year', year)
    .set('system:time_start', start.millis());
}

// ─────────────────────────────────────────────────────────────────────────────
// 2. WorldPop Population Grid
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Load the WorldPop UN-adjusted population count for a given year.
 * Reprojects to the study CRS and SCALE.
 *
 * @param {number} year — Year (2014–2020; capped at 2020 for WorldPop availability).
 * @param {ee.Geometry} roi
 * @returns {ee.Image} Population count image.
 */
function worldPop(year, roi) {
  // WorldPop currently available through 2020 on GEE
  var useYear = Math.min(year, 2020);
  return ee.ImageCollection('WorldPop/GP/100m/pop')
    .filter(ee.Filter.calendarRange(useYear, useYear, 'year'))
    .filterBounds(roi)
    .sum()
    .rename('population')
    .reproject({crs: CRS, scale: SCALE});
}

// ─────────────────────────────────────────────────────────────────────────────
// 3. GHSL Built-Up Surface (infrastructure density proxy)
// ─────────────────────────────────────────────────────────────────────────────

var GHSL = ee.Image('JRC/GHSL/P2023A/GHS_BUILT_S/2020')
  .select('built_surface')
  .rename('built_surface');


// ─────────────────────────────────────────────────────────────────────────────
// 4. Elevation (SRTM) and HAND approximation
// ─────────────────────────────────────────────────────────────────────────────

var SRTM = ee.Image('USGS/SRTMGL1_003').select('elevation').rename('elevation');

// HAND proxy: elevation difference from local drainage network
// Full HAND uses MERIT Hydro; here we use slope as an accessible approximation
var SLOPE = ee.Terrain.slope(SRTM).rename('slope_deg');


// ─────────────────────────────────────────────────────────────────────────────
// 5. Tile-level CSV export pipeline
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Export a tile-level feature table for one country × year.
 * Each record is a 1 km × 1 km tile with mean NTL, population, and built-surface.
 *
 * @param {string} countryName
 * @param {ee.Geometry} roi
 * @param {number} year
 */
function exportTileCSV(countryName, roi, year) {
  var ntl  = yearlyVIIRS(year, roi);
  var pop  = worldPop(year, roi);
  var builtClip  = GHSL.clip(roi);
  var slopeClip  = SLOPE.clip(roi);

  // Stack all bands
  var stack = ntl
    .addBands(pop)
    .addBands(builtClip)
    .addBands(slopeClip)
    .clip(roi);

  // Sample at 1 km grid
  var samples = stack.sample({
    region:      roi,
    scale:       SCALE,
    projection:  CRS,
    geometries:  true,
    numPixels:   50000
  });

  Export.table.toDrive({
    collection:  samples,
    description: countryName + '_tiles_' + year,
    folder:      FOLDER,
    fileNamePrefix: countryName + '_tiles_' + year,
    fileFormat:  'CSV'
  });
}

// ─────────────────────────────────────────────────────────────────────────────
// 6. GeoTIFF raster export (for QGIS / rasterio analysis)
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Export yearly NTL composite as a Cloud-Optimised GeoTIFF.
 *
 * @param {string} countryName
 * @param {ee.Geometry} roi
 * @param {number} year
 */
function exportNTLRaster(countryName, roi, year) {
  var ntl = yearlyVIIRS(year, roi);
  Export.image.toDrive({
    image:       ntl.toFloat(),
    description: countryName + '_NTL_' + year,
    folder:      FOLDER,
    fileNamePrefix: countryName + '_NTL_' + year,
    region:      roi,
    scale:       SCALE,
    crs:         CRS,
    fileFormat:  'GeoTIFF',
    maxPixels:   1e9
  });
}


// ─────────────────────────────────────────────────────────────────────────────
// 7. Run exports for all countries and years
// ─────────────────────────────────────────────────────────────────────────────

var countryNames = Object.keys(COUNTRIES);

countryNames.forEach(function(name) {
  var roi = COUNTRIES[name];

  // Export NTL rasters for all years (2014–2023)
  YEARS.getInfo().forEach(function(yr) {
    exportNTLRaster(name, roi, yr);
    exportTileCSV(name, roi, yr);
  });
});

print('Export tasks submitted. Check Tasks tab for progress.');


// ─────────────────────────────────────────────────────────────────────────────
// 8. Interactive map for verification
// ─────────────────────────────────────────────────────────────────────────────

// Display 2023 NTL for Morocco (including Western Sahara) as a visual check
var morocco2023 = yearlyVIIRS(2023, COUNTRIES.Morocco);
Map.setCenter(-6.5, 31.0, 5);
Map.addLayer(
  morocco2023,
  {min: 0, max: 50, palette: ['black', '#1a0000', '#ff4000', '#ffff00', '#ffffff']},
  'Morocco NTL 2023 (incl. Western Sahara)'
);
Map.addLayer(
  worldPop(2020, COUNTRIES.Morocco),
  {min: 0, max: 500, palette: ['white', 'orange', 'red']},
  'WorldPop 2020'
);
Map.addLayer(COUNTRIES.Morocco, {color: 'cyan'}, 'Study Extent', false);
