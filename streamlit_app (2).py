
import streamlit as st
import pandas as pd
import numpy as np
import leafmap.foliumap as leafmap
import geopandas as gpd
import json
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN

st.set_page_config(page_title="HCP Segmentation - IQVIA Themed", layout="wide")
st.title("🔷 IQVIA-Themed HCP Segmentation Map")
st.caption("Clusters generated using geo-coordinates and TRx, visualized with ZIP3 overlays and IQVIA brand styling.")

st.markdown("### ⚙️ Clustering Options")

algorithm = st.selectbox("Choose Algorithm", ["KMeans", "Agglomerative", "DBSCAN"])
if algorithm != "DBSCAN":
    num_clusters = st.slider("Number of Clusters", min_value=2, max_value=10, value=5)

algo_desc = {
    "KMeans": "📊 KMeans minimizes variance within groups — good for balanced data.",
    "Agglomerative": "🌲 Agglomerative builds nested clusters by merging neighbors.",
    "DBSCAN": "🔍 DBSCAN finds dense clusters — good for natural groups and outliers."
}
st.markdown(algo_desc[algorithm])

if st.button("🔍 Segment Territories"):
    # Load data
    with open("zip3_mock_territories.geojson") as f:
        geojson = json.load(f)

    hcp_df = pd.DataFrame({
        "HCP_ID": [f"HCP_{i:05d}" for i in range(5000)],
        "TRx": np.random.randint(50, 500, size=5000),
        "Latitude": np.random.uniform(25.0, 49.0, 5000),
        "Longitude": np.random.uniform(-124.0, -67.0, 5000)
    })

    features = hcp_df[["Latitude", "Longitude", "TRx"]]
    scaled = StandardScaler().fit_transform(features)

    if algorithm == "KMeans":
        model = KMeans(n_clusters=num_clusters, random_state=42)
        hcp_df["Territory"] = model.fit_predict(scaled)
    elif algorithm == "Agglomerative":
        model = AgglomerativeClustering(n_clusters=num_clusters)
        hcp_df["Territory"] = model.fit_predict(scaled)
    elif algorithm == "DBSCAN":
        model = DBSCAN(eps=0.8, min_samples=10)
        hcp_df["Territory"] = model.fit_predict(scaled)
        hcp_df["Outlier"] = hcp_df["Territory"] == -1

    st.markdown("### 🗺️ Map of Territories")
    m = leafmap.Map(center=[38, -96], zoom=4)
    m.add_geojson("zip3_mock_territories.geojson", layer_name="ZIP3 Territories")

    # Assign IQVIA-style blues to clusters
    colors = [['#015FF1', '#0072CE', '#66B2FF', '#99CCFF', '#B2D9FF', '#CCE6FF'][t % len(['#015FF1', '#0072CE', '#66B2FF', '#99CCFF', '#B2D9FF', '#CCE6FF'])] if t >= 0 else "#CCCCCC" for t in hcp_df["Territory"]]

    m.add_points_from_xy(
        hcp_df,
        x="Longitude",
        y="Latitude",
        popup=["HCP_ID", "TRx", "Territory"],
        icon_colors=colors,
        layer_name="HCPs",
        clustered=True,
    )
    m.to_streamlit(height=700)

    st.markdown("### 📋 HCP Table")
    st.dataframe(hcp_df[["HCP_ID", "TRx", "Latitude", "Longitude", "Territory"]])

    st.download_button(
        label="📥 Download CSV",
        data=hcp_df.to_csv(index=False),
        file_name="hcp_territories_iqvia.csv",
        mime="text/csv"
    )
else:
    st.info("👈 Select your clustering settings and click the button to segment.")
