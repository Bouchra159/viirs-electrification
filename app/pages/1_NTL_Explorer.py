"""
1_NTL_Explorer.py
-----------------
Interactive NTL radiance explorer: choropleth, histogram, year slider.
"""

import sys
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from app.utils.data_loader import load_panel, COUNTRY_COLORS, YEARS

st.set_page_config(page_title="NTL Explorer", page_icon="🌙", layout="wide")
st.title("🌙 Nighttime Light Explorer")
st.markdown("Explore VIIRS DNB NTL radiance distributions across Brazil, China, and Morocco.")

# ── Controls ─────────────────────────────────────────────────────────────────
col1, col2 = st.columns([1, 3])
with col1:
    selected_year = st.slider("Year", min_value=2015, max_value=2023, value=2022)
    selected_countries = st.multiselect(
        "Countries", options=["Brazil", "China", "Morocco"],
        default=["Brazil", "China", "Morocco"]
    )
    log_scale = st.checkbox("Log scale (NTL)", value=True)

# ── Load data ─────────────────────────────────────────────────────────────────
with st.spinner("Loading panel data..."):
    panel = load_panel()

snap = panel[(panel.year == selected_year) & (panel.country.isin(selected_countries))]

# ── NTL Distribution ──────────────────────────────────────────────────────────
with col2:
    st.subheader(f"NTL Distribution — {selected_year}")
    fig, ax = plt.subplots(figsize=(10, 4))
    for country in selected_countries:
        cdf = snap[snap.country == country]
        ntl = np.log1p(cdf.ntl_mean) if log_scale else cdf.ntl_mean
        ax.hist(ntl.values, bins=30, alpha=0.6, label=country,
                color=COUNTRY_COLORS.get(country, "#999"), edgecolor="white")
    ax.set_xlabel("log(1 + NTL radiance)" if log_scale else "NTL radiance (nW/cm²/sr)")
    ax.set_ylabel("Tile count")
    ax.set_title(f"NTL Radiance Distribution ({selected_year})", fontweight="bold")
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

st.divider()

# ── Temporal trend ────────────────────────────────────────────────────────────
st.subheader("Mean NTL Trend 2015–2023")
fig2, ax2 = plt.subplots(figsize=(10, 4))
for country in selected_countries:
    cdf = panel[panel.country == country].groupby("year")["ntl_mean"].mean().reset_index()
    ax2.plot(cdf.year, cdf.ntl_mean, marker="o", linewidth=2,
             color=COUNTRY_COLORS.get(country, "#999"), label=country)
ax2.axvline(selected_year, ls="--", color="gray", alpha=0.5, label=f"Selected: {selected_year}")
ax2.set_xlabel("Year"); ax2.set_ylabel("Mean NTL (nW/cm²/sr)")
ax2.set_title("Mean NTL Radiance Trend (population-unweighted)", fontweight="bold")
ax2.legend(); ax2.spines[["top", "right"]].set_visible(False)
fig2.tight_layout()
st.pyplot(fig2)
plt.close(fig2)

st.divider()

# ── Stats table ────────────────────────────────────────────────────────────────
st.subheader(f"Summary Statistics — {selected_year}")
rows = []
for country in selected_countries:
    cdf = snap[snap.country == country]["ntl_mean"]
    rows.append({
        "Country": country,
        "N tiles": len(cdf),
        "Mean NTL": f"{cdf.mean():.2f}",
        "Median NTL": f"{cdf.median():.2f}",
        "Std": f"{cdf.std():.2f}",
        "P10": f"{cdf.quantile(0.10):.2f}",
        "P90": f"{cdf.quantile(0.90):.2f}",
        "% >2.0 (electrified)": f"{(cdf > 2.0).mean()*100:.1f}%",
    })
st.dataframe(pd.DataFrame(rows), use_container_width=True)

# ── Interactive map link ───────────────────────────────────────────────────────
html_path = ROOT / "figures" / "map_ntl_choropleth.html"
if html_path.exists():
    st.divider()
    st.subheader("Interactive Folium Map")
    st.markdown("Open the interactive NTL choropleth map in your browser:")
    with open(html_path) as f:
        html_content = f.read()
    st.components.v1.html(html_content, height=500, scrolling=True)
