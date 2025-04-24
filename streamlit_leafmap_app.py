
import streamlit as st
import pandas as pd
import leafmap.foliumap as leafmap
import numpy as np
import json

st.set_page_config(page_title="HCP Segmentation Map (Leafmap)", layout="wide")
st.title("🗺️ HCP Territory Map with Clustering")
st.caption("Powered by Leafmap • Clustering by Latitude, Longitude, and TRx")

@st.cache_data
def load_data():
    hcp_df = pd.DataFrame({
        "HCP_ID": [f"HCP_{i:05d}" for i in range(5000)],
        "TRx": np.random.randint(50, 500, size=5000),
        "ZIP3": np.random.choice([str(z) for z in range(100, 999)], size=5000),
    })
    hcp_df["Territory"] = np.random.randint(0, 5, size=5000)

    # Generate random lat/lon within the US
    hcp_df["Latitude"] = np.random.uniform(25.0, 49.0, size=5000)
    hcp_df["Longitude"] = np.random.uniform(-124.0, -67.0, size=5000)

    return hcp_df

hcp_df = load_data()

# Load GeoJSON
geojson_path = "zip3_mock_territories.geojson"
with open(geojson_path) as f:
    geojson = json.load(f)

st.subheader("📍 Marker Cluster View + ZIP3 Territory Polygons")
m = leafmap.Map(center=[38, -96], zoom=4)

# Add ZIP3 polygons
m.add_geojson(geojson_path, layer_name="ZIP3 Territories")

# Add HCP clusters
m.add_points_from_xy(
    hcp_df,
    x="Longitude",
    y="Latitude",
    popup=["HCP_ID", "TRx", "ZIP3", "Territory"],
    icon_colors=["blue"] * len(hcp_df),
    layer_name="HCPs",
    clustered=True,
)

# Render map
m.to_streamlit(height=700)

# Table
st.subheader("📄 HCP Detail Table")
st.dataframe(hcp_df[["HCP_ID", "TRx", "ZIP3", "Territory", "Latitude", "Longitude"]])
