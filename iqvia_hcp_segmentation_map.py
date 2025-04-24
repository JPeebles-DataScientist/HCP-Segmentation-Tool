
import streamlit as st
import leafmap.foliumap as leafmap
import pandas as pd
import numpy as np
import json
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN

st.set_page_config(page_title="IQVIA HCP Clustering Map", layout="wide")

st.sidebar.title("About")
st.sidebar.info("This app demonstrates marker clustering, territory segmentation, and ZIP3 overlay.\nBased on opengeos/streamlit-map-template")
st.sidebar.image("https://i.imgur.com/UbOXYAU.png")

st.title("📍 IQVIA-Themed HCP Segmentation Map")

# Clustering controls
algorithm = st.sidebar.selectbox("Clustering Algorithm", ["KMeans", "Agglomerative", "DBSCAN"])
if algorithm != "DBSCAN":
    num_clusters = st.sidebar.slider("Number of Clusters", 2, 10, 5)

algo_info = {
    "KMeans": "📊 KMeans: Fast, balances clusters by minimizing variance.",
    "Agglomerative": "🌲 Agglomerative: Builds clusters from the bottom up.",
    "DBSCAN": "🔍 DBSCAN: Finds dense regions, ideal for natural clusters and noise."
}
st.sidebar.markdown(algo_info[algorithm])

if st.sidebar.button("🔍 Segment HCPs"):

    # Generate mock HCP data
    hcp_df = pd.DataFrame({
        "HCP_ID": [f"HCP_{i:05d}" for i in range(5000)],
        "TRx": np.random.randint(50, 500, 5000),
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

    # Create Leafmap
    m = leafmap.Map(center=[38, -96], zoom=4)

    # Add ZIP3 polygons
    m.add_geojson("zip3_mock_territories.geojson", layer_name="ZIP3 Territories")

    # Color logic
    color_scale = ['#015FF1', '#0072CE', '#66B2FF', '#99CCFF', '#B2D9FF', '#CCE6FF']
    icon_colors = [color_scale[t % len(color_scale)] if t >= 0 else "#999999" for t in hcp_df["Territory"]]

    # Add clustered HCP points
    m.add_points_from_xy(
        hcp_df,
        x="Longitude",
        y="Latitude",
        popup=["HCP_ID", "TRx", "Territory"],
        icon_colors=icon_colors,
        spin=False,
        add_legend=True,
        layer_name="HCP Clusters",
        clustered=True,
    )

    m.to_streamlit(height=700)

    # Show HCP Data Table
    st.subheader("📄 HCP Listing")
    st.dataframe(hcp_df[["HCP_ID", "TRx", "Latitude", "Longitude", "Territory"]])
    st.download_button("📥 Download CSV", hcp_df.to_csv(index=False), "hcp_territories.csv", "text/csv")

else:
    st.info("👈 Set your clustering preferences and press 'Segment HCPs' to begin.")
