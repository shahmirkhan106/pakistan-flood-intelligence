import json
import pandas as pd
import streamlit as st
import plotly.express as px

st.set_page_config(
    page_title="Pakistan Flood Intelligence System",
    layout="wide"
)

# Load predictions
try:
    with open("flood_predictions.json", "r") as f:
        data = json.load(f)
except FileNotFoundError:
    st.error("⚠️ `flood_predictions.json` not found. Run `python flood_backend.py` first to generate predictions.")
    st.stop()

predictions = data["predictions"]

# Header
st.title("🌊 Pakistan Flood Intelligence System")
st.caption("Real Satellite Data • Sentinel-1 • SMAP • CHIRPS")

# Convert to dataframe
rows = []

for province, values in predictions.items():
    rows.append({
        "Province": province,
        "Risk %": values.get("ensemble_prob", 0),
        "Prediction": values.get("prediction", "N/A"),
        "Rainfall": values.get("rainfall_mm", 0),
        "Soil Moisture": values.get("soil_moisture", 0),
        "SAR Delta": values.get("sar_delta_db", 0),
        "NDVI": values.get("ndvi", 0)
    })

df = pd.DataFrame(rows)

# Top metrics
col1, col2, col3 = st.columns(3)

col1.metric(
    "Highest Risk Province",
    df.sort_values("Risk %", ascending=False).iloc[0]["Province"]
)

col2.metric(
    "Maximum Risk",
    f"{df['Risk %'].max():.1f}%"
)

col3.metric(
    "Average National Risk",
    f"{df['Risk %'].mean():.1f}%"
)

st.divider()

# Table
st.subheader("Province Predictions")
st.dataframe(df, use_container_width=True)

# Risk chart
st.subheader("Flood Risk by Province")

fig_risk = px.bar(
    df.sort_values("Risk %", ascending=False),
    x="Province",
    y="Risk %",
    text="Risk %",
)

st.plotly_chart(fig_risk, use_container_width=True)

# Rainfall chart
st.subheader("Rainfall")

fig_rainfall = px.bar(
    df,
    x="Province",
    y="Rainfall"
)

st.plotly_chart(fig_rainfall, use_container_width=True)

# Soil Moisture chart
st.subheader("Soil Moisture")

fig_soil = px.bar(
    df,
    x="Province",
    y="Soil Moisture"
)

st.plotly_chart(fig_soil, use_container_width=True)

# SAR chart
st.subheader("SAR Delta")

fig_sar = px.bar(
    df,
    x="Province",
    y="SAR Delta"
)

st.plotly_chart(fig_sar, use_container_width=True)

# Raw JSON
with st.expander("View Raw Prediction JSON"):
    st.json(data)
