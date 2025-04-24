
import streamlit as st
import pandas as pd
import numpy as np
import json
import leafmap.foliumap as leafmap
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

st.set_page_config(page_title="HCP Segmentation Tool", layout="wide")
st.title("HCP Segmentation Tool")

st.sidebar.header("Clustering Settings")
st.sidebar.markdown("Uses fixed regional centers for realistic, geographically coherent segmentation.")

if st.sidebar.button("🔍 Segment HCPs"):

    with open("zip3_mock_territories.geojson") as f:
        geojson = json.load(f)

    hcp_df = pd.DataFrame({
        "HCP_ID": [f"HCP_{i:05d}" for i in range(5000)],
        "TRx": np.random.randint(50, 500, 5000),
        "Latitude": np.random.uniform(25.0, 49.0, 5000),
        "Longitude": np.random.uniform(-124.0, -67.0, 5000)
    })

    features = hcp_df[["Latitude", "Longitude", "TRx"]]
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)

    initial_centers = np.array([
        [34.0, -118.2, 250],  # West (LA)
        [40.7, -74.0, 250],   # East (NY)
        [29.8, -95.4, 250],   # South (Houston)
        [41.9, -87.6, 250],   # Midwest (Chicago)
        [33.4, -112.0, 250],  # Southwest (Phoenix)
    ])
    scaled_init = scaler.transform(initial_centers)

    model = KMeans(n_clusters=5, init=scaled_init, n_init=1, random_state=42)
    hcp_df["Territory"] = model.fit_predict(scaled)

    st.subheader("📍 Territory Map")
    m = leafmap.Map(center=[38, -96], zoom=4)
    m.add_geojson("zip3_mock_territories.geojson", layer_name="ZIP3 Territories")

    iqvia_colors = ['#015FF1', '#0072CE', '#66B2FF', '#99CCFF', '#B2D9FF']
    color_map = {i: iqvia_colors[i % len(iqvia_colors)] for i in np.unique(hcp_df["Territory"])}

    for territory_id, group in hcp_df.groupby("Territory"):
        color = color_map[territory_id]
        m.add_points_from_xy(
            group,
            x="Longitude",
            y="Latitude",
            popup=["HCP_ID", "TRx", "Territory"],
            icon_colors=[color] * len(group),
            clustered=False,
            layer_name=f"Territory {territory_id}"
        )

    m.to_streamlit(height=700)

    tab1, tab2 = st.tabs(["📋 HCP Table", "📊 Territory Summary"])

    with tab1:
        st.dataframe(hcp_df[["HCP_ID", "TRx", "Latitude", "Longitude", "Territory"]])

    with tab2:
        summary = hcp_df.groupby("Territory").agg(
            Total_HCPs=("HCP_ID", "count"),
            Total_TRx=("TRx", "sum"),
            Avg_TRx_per_HCP=("TRx", "mean")
        ).reset_index()

        st.dataframe(summary)
        st.plotly_chart(px.bar(summary, x="Territory", y="Total_HCPs", color="Territory", color_discrete_sequence=iqvia_colors), use_container_width=True)
        st.plotly_chart(px.pie(summary, names="Territory", values="Total_TRx", color="Territory", color_discrete_sequence=iqvia_colors), use_container_width=True)

    st.download_button("📥 Download HCP CSV", hcp_df.to_csv(index=False), "hcp_clusters.csv", "text/csv")
else:
    st.info("👈 Click the button to segment HCPs using regional centers.")
