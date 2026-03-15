"""
2_SDG7_Tracker.py
-----------------
SDG 7.1.1 access rates, Theil-Sen projections, gap analysis table.
"""

import sys
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from app.utils.data_loader import (
    load_panel, load_access_rates, load_projections,
    load_gap_analysis, COUNTRY_COLORS, YEARS,
)
from sdg7_tracker import SDG7_UNIVERSAL_TARGET

st.set_page_config(page_title="SDG7 Tracker", page_icon="⚡", layout="wide")
st.title("⚡ SDG 7 Progress Tracker")
st.markdown("Population-weighted electricity access rates, 2030 projections, and gap analysis.")

# ── Load ──────────────────────────────────────────────────────────────────────
with st.spinner("Computing access rates..."):
    panel = load_panel()
    ar_df = load_access_rates(panel)
    proj  = load_projections(ar_df)
    gap   = load_gap_analysis(ar_df)

# ── Status badges ─────────────────────────────────────────────────────────────
st.subheader("SDG 7 Status (2023)")
cols = st.columns(3)
status_colors = {"On track": "🟢", "At risk": "🟡", "Off track": "🔴"}
for col, (_, row) in zip(cols, gap.iterrows()):
    icon = status_colors.get(row["Status"], "⚪")
    col.metric(
        f"{icon} {row['Country']}",
        f"{row['Access 2023 (%)']:.1f}%",
        f"Proj. 2030: {row['Projected 2030 (%)']:.1f}%",
    )

st.divider()

# ── Access rate timeline ──────────────────────────────────────────────────────
st.subheader("Access Rate Timeline + 2030 Projection")

fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
for ax, country in zip(axes, ["Brazil", "China", "Morocco"]):
    sub = ar_df[ar_df.country == country].sort_values("year")
    color = COUNTRY_COLORS[country]
    ax.plot(sub.year, sub.access_rate * 100, "o-", color=color, linewidth=2.5,
            markersize=6, label="VIIRS estimate", zorder=4)

    r = proj[country]
    proj_years = r["proj_years"]
    ax.plot(proj_years, np.array(r["proj_values"]) * 100, "--", color=color, alpha=0.7, linewidth=1.8)
    ax.fill_between(proj_years, np.array(r["proj_lower"]) * 100,
                    np.array(r["proj_upper"]) * 100, alpha=0.15, color=color)

    ax.axhline(SDG7_UNIVERSAL_TARGET * 100, ls=":", color="red", alpha=0.6, linewidth=1.4)
    ax.axvline(2030, ls=":", color="gray", alpha=0.5)
    ax.set_title(country, fontweight="bold", fontsize=12)
    ax.set_xlabel("Year")
    if ax == axes[0]:
        ax.set_ylabel("Population-weighted access rate (%)")
    ax.set_ylim(60, 103)
    ax.spines[["top", "right"]].set_visible(False)

fig.suptitle("SDG 7.1.1 Electricity Access — VIIRS NTL Estimates (2015–2023) + Projection to 2030",
             fontsize=13, fontweight="bold")
fig.tight_layout()
st.pyplot(fig)
plt.close(fig)

# ── Projection summary ────────────────────────────────────────────────────────
st.subheader("2030 Projection Summary (Theil-Sen + 95% Bootstrap CI)")
proj_rows = []
for country, r in proj.items():
    proj_rows.append({
        "Country": country,
        "Slope (pp/yr)": f"{r['annual_slope_pp']:.3f}",
        "Trend": r["trend"],
        "Projected 2030 (%)": f"{min(r['projected_value']*100, 100):.1f}",
        "95% CI": f"[{min(r['ci_lower']*100,100):.1f} – {min(r['ci_upper']*100,100):.1f}]",
        "p-value (MK)": f"{r['p_value']:.3f}",
    })
st.dataframe(pd.DataFrame(proj_rows), use_container_width=True)

st.divider()

# ── Gap analysis table ─────────────────────────────────────────────────────────
st.subheader("SDG 7 Gap Analysis Table")

def color_status(val):
    if val == "On track":
        return "background-color: #c8e6c9; color: #1b5e20"
    elif val == "At risk":
        return "background-color: #fff9c4; color: #f57f17"
    else:
        return "background-color: #ffcdd2; color: #b71c1c"

styled = gap.style.applymap(color_status, subset=["Status"])
st.dataframe(styled, use_container_width=True)

# ── Download ───────────────────────────────────────────────────────────────────
csv_path = ROOT / "figures" / "sdg7_attainment_table.csv"
if csv_path.exists():
    with open(csv_path) as f:
        st.download_button("Download Gap Analysis CSV", f.read(),
                           file_name="sdg7_attainment_table.csv", mime="text/csv")
