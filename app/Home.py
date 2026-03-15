"""
Home.py — VIIRS Electrification ML  |  Streamlit App
=====================================================
Landing page: project abstract, key metrics, figure gallery.
Run:  streamlit run app/Home.py  (from project root)
"""

import streamlit as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "figures"

st.set_page_config(
    page_title="VIIRS Electrification ML",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar ──────────────────────────────────────────────────────────────────
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/4/47/PNG_transparency_demonstration_1.png",
                 width=1)  # placeholder spacer
st.sidebar.title("⚡ VIIRS Electrification ML")
st.sidebar.markdown("""
**SDG 7 Monitoring Framework**
Satellite NTL → Electricity Access

---
**Countries**
- 🇧🇷 Brazil
- 🇨🇳 China
- 🇲🇦 Morocco (+ Western Sahara)

**Period:** 2015 – 2023
**Data:** VIIRS DNB synthetic panel
**Author:** Bouchra Daddaoui
""")

# ── Hero ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="background: linear-gradient(135deg, #1a237e 0%, #283593 50%, #1565c0 100%);
            padding: 2.5rem 2rem; border-radius: 12px; margin-bottom: 1.5rem;">
  <h1 style="color: white; margin:0; font-size: 2.2rem;">
    ⚡ VIIRS Nighttime Light — SDG 7 Electrification Monitoring
  </h1>
  <p style="color: #90caf9; margin-top: 0.6rem; font-size: 1.05rem;">
    A satellite-based machine-learning framework for tracking electricity access
    across Brazil, China, and Morocco (2015–2023)
  </p>
</div>
""", unsafe_allow_html=True)

# ── Key Metrics ───────────────────────────────────────────────────────────────
st.subheader("Key Results at a Glance")
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Brazil Access (2023)", "99.3%", "+2.8pp vs 2015")
col2.metric("China Access (2023)", "98.5%", "+1.2pp vs 2015")
col3.metric("Morocco Access (2023)", "100%", "+8.6pp vs 2015")
col4.metric("Validation R²", "0.57", "vs WB/IEA official")
col5.metric("SDG7 Status", "On Track", "All 3 countries")

st.divider()

# ── Project Description ───────────────────────────────────────────────────────
col_l, col_r = st.columns([3, 2])
with col_l:
    st.subheader("About This Project")
    st.markdown("""
    This project builds an end-to-end research pipeline for **SDG 7 (Affordable and Clean Energy)**
    monitoring using **VIIRS DNB nighttime light (NTL) radiance** as a proxy for electricity access.

    **What we do:**
    - Classify 1 km² tiles as *electrified / unelectrified* using NTL thresholding
    - Compute **population-weighted access rates** per country and year
    - Project 2030 trajectories with **Theil-Sen regression + 95% bootstrap CI**
    - Analyse spatial inequality with **Local Moran's I (LISA)** and **MGWR**
    - Explain access determinants with **XGBoost + SHAP**
    - Measure inequality via **Gini coefficient** and **Theil decomposition**
    - Validate against World Bank WDI and IEA official statistics

    **Why it matters:**
    760 million people still lack electricity access (IEA 2023). Current reporting relies on
    infrequent household surveys. NTL satellite data provides annual, spatially explicit estimates
    that can close this monitoring gap.
    """)

with col_r:
    st.subheader("Methodology")
    st.markdown("""
    ```
    VIIRS DNB radiance (2015–2023)
           ↓
    Tile classification (threshold / Otsu)
           ↓
    Population-weighted access rate
           ↓
    ┌──────────────────────────────┐
    │  Spatial:  LISA / MGWR      │
    │  Temporal: Theil-Sen + MK   │
    │  ML:       XGBoost + SHAP   │
    │  Inequality: Gini / Theil   │
    └──────────────────────────────┘
           ↓
    SDG 7.1.1 gap analysis + 2030 projection
           ↓
    Validation vs WB WDI / IEA
    ```
    """)

st.divider()

# ── Figure Gallery ────────────────────────────────────────────────────────────
st.subheader("Figure Gallery")

fig_list = [
    ("sdg7_progress_timeline.png", "SDG 7 Progress Timeline (2015–2030)"),
    ("sdg7_status_dashboard.png", "SDG 7 Status Dashboard"),
    ("validation_scatter.png", "Validation vs Official Statistics"),
    ("bivariate_map.png", "Bivariate NTL × Population Density"),
    ("lisa_cluster_maps.png", "LISA Spatial Clusters"),
    ("shap_summary.png", "SHAP Feature Importance"),
    ("sdg7_country_comparison.png", "Electrification Index vs Gini"),
    ("gini_trend.png", "NTL Gini Trend 2015–2023"),
]

cols = st.columns(4)
for idx, (fname, caption) in enumerate(fig_list):
    path = FIGURES / fname
    if path.exists():
        with cols[idx % 4]:
            st.image(str(path), caption=caption, use_container_width=True)

st.divider()

# ── Navigation hint ───────────────────────────────────────────────────────────
st.info("""
**Navigate using the sidebar:**
- **NTL Explorer** — explore nighttime light radiance by country and year
- **SDG7 Tracker** — access rates, projections, and gap analysis
- **Energy Poverty** — spatial clusters, bivariate maps, inequality metrics
- **Projections 2030** — Theil-Sen projections with bootstrap uncertainty
- **Methods** — detailed methodology and validation
""")
