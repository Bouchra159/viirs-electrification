# Electrification Inequality from Space
### A Multi-Country Spatial ML Analysis Using VIIRS Nighttime Lights

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://python.org)
[![GEE](https://img.shields.io/badge/Google_Earth_Engine-JS%20%7C%20Python-brightgreen)](https://earthengine.google.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Author:** Bouchra Daddaoui · [bouchra1daddaoui@gmail.com](mailto:bouchra1daddaoui@gmail.com)
**GitHub:** [@Bouchra159](https://github.com/Bouchra159)
**Status:** Active research project — targeting submission to *Computers, Environment and Urban Systems* or *Applied Geography*

---

## Overview

This project investigates **spatial and temporal dynamics of electrification inequality** across
Brazil, China, and Morocco (2014–2023) using VIIRS DNBS nighttime light (NTL) satellite imagery
as a proxy for electricity access. It combines:

- **Google Earth Engine** — cloud-based export of VIIRS/NOAA annual composites and WorldPop population grids
- **Spatial econometrics** — Moran's I autocorrelation, LISA cluster mapping, spatial lag/error models
- **Interpretable ML** — XGBoost regression with SHAP (SHapley Additive exPlanations) spatial maps
- **Temporal analysis** — Mann-Kendall trend test, Theil-Sen estimator, CUSUM change-point detection
- **QGIS cartography** — publication-quality choropleth and cluster maps at 1 km resolution

The analysis is framed around **UN SDG 7** (Affordable and Clean Energy) and contributes
quantitative evidence on which geographic, demographic, and infrastructural factors drive
within-country electrification gaps.

---

## Research Questions

1. Do electrification levels (measured by NTL) exhibit significant **spatial autocorrelation**
   within Brazil, China, and Morocco — and does clustering differ by country context?
2. What geographic and socioeconomic covariates most strongly predict NTL intensity,
   and **where** do their effects vary spatially (spatial SHAP analysis)?
3. Is there evidence of **structural change** in electrification trajectories over 2014–2023,
   consistent with national rural electrification programmes?

---

## Key Results

| Country | Moran's I (NTL) | Best spatial model | Top SHAP feature | Trend (M-K) | Change year |
|---------|----------------|-------------------|-----------------|-------------|-------------|
| Brazil  | 0.62 *** | Spatial Lag (SLM) | Infrastructure density | Increasing (p < 0.01) | — |
| China   | 0.78 *** | Spatial Error (SEM) | GDP proxy | Increasing → decline 2020 | 2020 (COVID) |
| Morocco | 0.54 *** | Spatial Lag (SLM) | Distance to city | Increasing (p < 0.01) | 2017 (rural elec.) |

**** p < 0.001 (999 permutations)*

**XGBoost (Leave-One-Country-Out):** mean RMSE = 3.21 ± 0.94 | mean R² = 0.74 ± 0.08

---

## Figures

| Figure | Description |
|--------|-------------|
| ![Moran scatterplots](figures/moran_scatterplots.png) | Global Moran's I scatterplots |
| ![LISA maps](figures/lisa_cluster_maps.png) | LISA HH/LL/HL/LH cluster maps |
| ![SHAP summary](figures/shap_summary.png) | Global SHAP feature importance |
| ![Spatial SHAP](figures/spatial_shap_maps.png) | Per-country spatial SHAP maps |
| ![Uncertainty](figures/bootstrap_uncertainty_maps.png) | Bootstrap 90% CI width maps |
| ![Trends](figures/ntl_trends.png) | NTL trend lines with Theil-Sen slopes |
| ![CUSUM](figures/cusum_changepoints.png) | CUSUM structural break detection |

---

## Repository Structure

```
viirs-electrification/
│
├── scripts/
│   ├── gee_export.js           # GEE JavaScript: VIIRS + WorldPop export pipeline
│   ├── spatial_analysis.py     # Moran's I, LISA, spatial lag/error models
│   ├── ml_models.py            # XGBoost, SHAP, bootstrap uncertainty
│   ├── temporal_analysis.py    # Mann-Kendall, Theil-Sen, CUSUM
│   └── qgis_workflow.md        # QGIS cartography guide (symbology, print layouts)
│
├── notebooks/
│   ├── 02_spatial_analysis.ipynb   # Spatial autocorrelation + regression
│   ├── 03_ml_shap.ipynb            # XGBoost + SHAP + uncertainty maps
│   └── 04_temporal_trends.ipynb    # Trend detection + change-point analysis
│
├── figures/                    # All generated figures (PNG, 150 DPI)
├── data/
│   └── README.md               # Data download instructions (GEE + GADM)
├── requirements.txt
└── README.md
```

---

## Data Sources

| Dataset | Source | Resolution | Years |
|---------|--------|-----------|-------|
| VIIRS DNB Monthly V1 | NOAA / NASA via GEE `NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG` | ~500 m | 2014–2023 |
| WorldPop UN-adjusted | WorldPop / GEE `WorldPop/GP/100m/pop` | 100 m | 2014–2020 |
| GHSL Built-up Surface | JRC / GEE `JRC/GHSL/P2023A/GHS_BUILT_S/2020` | 100 m | 2020 |
| SRTM Elevation | USGS / GEE `USGS/SRTMGL1_003` | 30 m | — |
| Admin Boundaries | GADM v4.1 | — | — |

> **Morocco extent:** All analyses include **Western Sahara** within the Moroccan study area
> (bbox: [-17.10, 20.76, -1.01, 35.93] EPSG:4326), consistent with current administrative realities.

---

## Methods Summary

### 1. Data Pipeline (GEE → Python)
Raw VIIRS monthly composites are exported via `scripts/gee_export.js` as yearly median
composites at 1 km resolution. Negative radiance values (stray light artefacts) are masked
prior to compositing.

### 2. Spatial Autocorrelation
Global Moran's I is computed with K=8 nearest-neighbour weights (row-standardised) and
tested against 999 random permutations. Local Moran's I (LISA) identifies cluster types
at p < 0.05 significance.

### 3. Spatial Regression
Three models are compared per country:
- **OLS** with Moran test on residuals and Lagrange Multiplier (LM) diagnostics
- **Spatial Lag Model (SLM)** — ML estimation of ρWy
- **Spatial Error Model (SEM)** — ML estimation of λWε

Model selection follows the LM robust decision rule (Anselin 1988).

### 4. XGBoost + SHAP
An XGBoost regressor is trained on multi-country panel data with 5-fold CV and
leave-one-country-out (LOCO) evaluation. SHAP TreeExplainer provides global and
spatial-local feature importance. Bootstrap ensembles (n=200) quantify prediction uncertainty.

### 5. Temporal Analysis
Mann-Kendall trend test (non-parametric, p < 0.05) and Theil-Sen slope estimation
for each country's yearly NTL median. CUSUM change-point detection identifies structural breaks
in the electrification trajectory.

---

## Setup & Reproduction

```bash
# Clone this repository
git clone https://github.com/Bouchra159/viirs-electrification.git
cd viirs-electrification

# Install dependencies
pip install -r requirements.txt

# (Optional) Export data from GEE
# Paste scripts/gee_export.js into https://code.earthengine.google.com and run tasks

# Run analyses (order matters for data dependencies)
jupyter notebook notebooks/02_spatial_analysis.ipynb
jupyter notebook notebooks/03_ml_shap.ipynb
jupyter notebook notebooks/04_temporal_trends.ipynb
```

---

## QGIS Workflow

Publication-quality maps are produced in QGIS 3.28 LTR using GeoTIFF outputs
from the GEE pipeline. See [`scripts/qgis_workflow.md`](scripts/qgis_workflow.md) for:
- Layer loading and symbology (Magma colour ramp for NTL, diverging for SHAP/residuals)
- LISA cluster symbology specification (HH/LL/HL/LH palette)
- Print Layout settings for journal-quality export (300 DPI, A4)
- Temporal animation setup for VIIRS 2014–2023 series

---

## Relevance to Computational Social Science

This project sits at the intersection of **remote sensing, spatial econometrics, and development economics**,
targeting journals such as:
- *Computers, Environment and Urban Systems*
- *Applied Geography*
- *Environment and Planning B: Urban Analytics and City Science*
- *Remote Sensing* (MDPI open access)

Methodological contributions:
1. **Spatial SHAP maps** — novel spatial visualisation of ML feature effects
2. **LOCO cross-country evaluation** — rigorous out-of-distribution test for geospatial ML
3. **Integrated pipeline** — GEE → Python → QGIS in a single reproducible workflow

---

## Citation

If you use this code or methodology in your work, please cite:

```bibtex
@misc{daddaoui2025electrification,
  author    = {Daddaoui, Bouchra},
  title     = {Electrification Inequality from Space: A Multi-Country Spatial ML Analysis
               Using VIIRS Nighttime Lights},
  year      = {2025},
  publisher = {GitHub},
  url       = {https://github.com/Bouchra159/viirs-electrification}
}
```

---

## License

MIT © Bouchra Daddaoui
