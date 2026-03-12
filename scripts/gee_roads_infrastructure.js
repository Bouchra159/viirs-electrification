/**
 * gee_roads_infrastructure.js
 * ============================
 * Google Earth Engine script for deriving road density and infrastructure
 * rasters to complement the VIIRS nighttime light analysis.
 *
 * Datasets used:
 *   - OpenStreetMap via GRIP (Global Roads Inventory Project) — road density
 *   - GHSL Settlement Model — urban fabric density
 *   - JRC Global Surface Water — water body mask (exclude from analysis)
 *   - SRTM — elevation + terrain ruggedness (barrier to electrification)
 *
 * All outputs are exported at 1 km resolution, EPSG:4326.
 * Morocco extent includes Western Sahara (bbox [-17.10, 20.76, -1.01, 35.93]).
 *
 * Author: Bouchra Daddaoui
 */

// ─── Study areas ─────────────────────────────────────────────────────────────

var COUNTRIES = {
  Brazil:  ee.Geometry.BBox(-73.99, -33.75, -28.84,  5.27),
  China:   ee.Geometry.BBox( 73.50,  18.15, 134.77, 53.56),
  Morocco: ee.Geometry.BBox(-17.10,  20.76,  -1.01, 35.93)  // includes Western Sahara
};

var SCALE  = 1000;   // 1 km
var CRS    = 'EPSG:4326';
var FOLDER = 'viirs_electrification';


// ─── 1. Road Density from GRIP Global Roads ───────────────────────────────────
//
// GRIP provides roads as polylines at ~100 m.
// We rasterise them and compute density (km of road per km² tile).
// Higher road density = better infrastructure = proxy for electrification access.

var GRIP = ee.FeatureCollection('projects/sat-io/open-datasets/GRIP4/GRIP4_Region');

function roadDensity(roi) {
  // Rasterise road presence (1 = road pixel, 0 = no road)
  var roadRaster = GRIP
    .filterBounds(roi)
    .reduceToImage(['GP_RTP'], ee.Reducer.max())
    .gt(0)
    .rename('road_presence')
    .unmask(0);

  // Convolve with 1 km kernel to get road density
  var kernel = ee.Kernel.circle({radius: 500, units: 'meters'});
  var density = roadRaster.convolve(kernel)
    .reproject({crs: CRS, scale: SCALE})
    .rename('road_density')
    .clip(roi);

  return density;
}


// ─── 2. Urban fabric density (GHSL Settlement Model 2020) ────────────────────

var GHSL_SMOD = ee.Image('JRC/GHSL/P2023A/GHS_SMOD/2020')
  .select('smod_code')
  .rename('urban_class');
// Classes: 30=dense urban, 23=semi-dense, 22=suburban, 21=rural cluster, 11=low-density, 10=very low

function urbanDensity(roi) {
  return GHSL_SMOD.clip(roi).reproject({crs: CRS, scale: SCALE});
}


// ─── 3. Terrain ruggedness (barrier to grid extension) ───────────────────────

var SRTM = ee.Image('USGS/SRTMGL1_003').select('elevation');

function terrainRuggedness(roi) {
  // TRI: mean absolute difference between centre pixel and 8 neighbours
  var tri = SRTM.reduceNeighborhood({
    reducer: ee.Reducer.stdDev(),
    kernel: ee.Kernel.square({radius: 1})
  }).rename('terrain_ruggedness').clip(roi).reproject({crs: CRS, scale: SCALE});
  return tri;
}


// ─── 4. Distance to nearest road (connectivity gap metric) ───────────────────

function distanceToRoad(roi) {
  var roadRaster = GRIP
    .filterBounds(roi)
    .reduceToImage(['GP_RTP'], ee.Reducer.max())
    .gt(0)
    .unmask(0);

  // fastDistanceTransform gives per-pixel distance to nearest road pixel
  var dist = roadRaster.fastDistanceTransform()
    .sqrt()
    .multiply(ee.Image.pixelArea().sqrt())  // convert to metres
    .divide(1000)                            // to km
    .rename('dist_to_road_km')
    .clip(roi)
    .reproject({crs: CRS, scale: SCALE});

  return dist;
}


// ─── 5. Composite feature stack ──────────────────────────────────────────────

function buildFeatureStack(roi, countryName, year) {
  var ntlYearly = ee.ImageCollection('NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG')
    .filterBounds(roi)
    .filterDate(ee.Date.fromYMD(year, 1, 1), ee.Date.fromYMD(year, 12, 31))
    .select('avg_rad')
    .map(function(img) { return img.updateMask(img.gt(0)); })
    .median()
    .rename('avg_rad');

  var pop = ee.ImageCollection('WorldPop/GP/100m/pop')
    .filter(ee.Filter.calendarRange(Math.min(year, 2020), Math.min(year, 2020), 'year'))
    .filterBounds(roi)
    .sum()
    .rename('population')
    .reproject({crs: CRS, scale: SCALE});

  var stack = ntlYearly
    .addBands(pop)
    .addBands(roadDensity(roi))
    .addBands(distanceToRoad(roi))
    .addBands(urbanDensity(roi))
    .addBands(terrainRuggedness(roi))
    .clip(roi);

  return stack;
}


// ─── 6. Export full feature stacks (GeoTIFF for QGIS) ────────────────────────

[2020, 2023].forEach(function(year) {
  Object.keys(COUNTRIES).forEach(function(name) {
    var roi   = COUNTRIES[name];
    var stack = buildFeatureStack(roi, name, year);

    Export.image.toDrive({
      image:          stack.toFloat(),
      description:    name + '_feature_stack_' + year,
      folder:         FOLDER,
      fileNamePrefix: name + '_feature_stack_' + year,
      region:         roi,
      scale:          SCALE,
      crs:            CRS,
      fileFormat:     'GeoTIFF',
      maxPixels:      1e9
    });
  });
});


// ─── 7. Interactive visualisation in GEE Code Editor ─────────────────────────

// Centred on Morocco (includes Western Sahara) — drag to other countries to explore
Map.setCenter(-8.0, 28.0, 6);

// VIIRS NTL 2023
var ntl2023 = ee.ImageCollection('NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG')
  .filterDate('2023-01-01', '2023-12-31')
  .select('avg_rad')
  .map(function(img) { return img.updateMask(img.gt(0)); })
  .median();

Map.addLayer(
  ntl2023.clip(COUNTRIES.Morocco),
  {min: 0, max: 60, palette: ['000000', '1a0000', 'ff4000', 'ffff00', 'ffffff']},
  'Morocco NTL 2023 (incl. W. Sahara)'
);

// Road density
Map.addLayer(
  roadDensity(COUNTRIES.Morocco),
  {min: 0, max: 0.3, palette: ['white', 'orange', 'darkred']},
  'Road Density (Morocco)'
);

// Urban classification
Map.addLayer(
  GHSL_SMOD.clip(COUNTRIES.Morocco),
  {min: 10, max: 30, palette: ['#ffffcc', '#a1dab4', '#41b6c4', '#2c7fb8', '#253494']},
  'GHSL Urban Class'
);

// Distance to road (electrification access gap)
Map.addLayer(
  distanceToRoad(COUNTRIES.Morocco),
  {min: 0, max: 50, palette: ['#2166ac', '#abd9e9', '#ffffbf', '#fdae61', '#d73027']},
  'Distance to Road (km) — Access Gap'
);

print('Layer order: NTL → Roads → Urban → Distance to Road');
print('Morocco bbox includes Western Sahara: [-17.10, 20.76, -1.01, 35.93]');
