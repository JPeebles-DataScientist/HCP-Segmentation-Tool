
import streamlit as st
import pandas as pd
import plotly.express as px
import json

st.set_page_config(page_title="HCP Territory Map", layout="wide")
st.title("📍 ZIP3-Level HCP Territory Map")
st.markdown("This map shows territory segmentation using ZIP3 boundaries and displays TRx and HCP metrics per region.")

# Load ZIP3 metrics and geojson
metrics_url = "zip3_territory_metrics.csv"
geojson_url = "zip3_mock_territories.geojson"

@st.cache_data
def load_data():
    metrics_df = pd.read_csv(metrics_url, dtype={"ZIP3": str})
    with open(geojson_url) as f:
        geojson_data = json.load(f)
    return metrics_df, geojson_data

metrics_df, geojson_data = load_data()

# Territory coloring map
st.subheader("🗺️ Territory Map by ZIP3")
fig = px.choropleth_mapbox(
    metrics_df,
    geojson=geojson_data,
    locations="ZIP3",
    color="Territory",
    featureidkey="properties.ZIP3",
    hover_data=["ZIP3", "Territory", "Total_TRx", "HCP_Count"],
    mapbox_style="carto-positron",
    center={"lat": 38.0, "lon": -96.0},
    zoom=3,
    opacity=0.6,
    color_continuous_scale="Viridis",
    height=650
)

st.plotly_chart(fig, use_container_width=True)

st.markdown("🔁 Data from synthetic ZIP3 territory segmentation. Overlay support for HCP dots coming next.")
