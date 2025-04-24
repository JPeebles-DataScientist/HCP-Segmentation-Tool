
import streamlit as st
import leafmap.foliumap as leafmap
import pandas as pd
import numpy as np
import json
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
import plotly.express as px

st.set_page_config(page_title="HCP Segmentation Tool", layout="wide")

st.sidebar.title("About")
st.sidebar.info("This tool allows users to cluster HCPs into geographic territories using algorithms like KMeans, Agglomerative Clustering, and DBSCAN. Clusters are visualized on an interactive map with ZIP3 overlays.")
st.sidebar.image("https://i.imgur.com/UbOXYAU.png")

st.title("HCP Segmentation Tool")

# Sidebar clustering options
algorithm = st.sidebar.selectbox("Clustering Algorithm", ["KMeans", "Agglomerative", "DBSCAN"])
if algorithm != "DBSCAN":
    num_clusters = st.sidebar.slider("Number of Clusters", 2, 10, 5)

algo_info = {
    "KMeans": "📊 KMeans: Fast, balances clusters by minimizing variance.",
    "Agglomerative": "🌲 Agglomerative: Merges similar records into clusters.",
    "DBSCAN": "🔍 DBSCAN: Detects dense areas and marks outliers."
}
st.sidebar.markdown(algo_info[algorithm])

if st.sidebar.button("🔍 Segment HCPs"):

    # Load or simulate data
    with open("zip3_mock_territories.geojson") as f:
        geojson = json.load(f)

    hcp_df = pd.DataFrame({
        "HCP_ID": [f"HCP_{i:05d}" for i in range(5000)],
        "TRx": np.random.randint(50, 500, 5000),
        "Latitude": np.random.uniform(25.0, 49.0, 5000),
        "Longitude": np.random.uniform(-124.0, -67.0, 5000)
    })

    # Apply clustering
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

    st.markdown("### 📍 Map View (Territories Color Coded)")
    m = leafmap.Map(center=[38, -96], zoom=4)
    m.add_geojson("zip3_mock_territories.geojson", layer_name="ZIP3 Territories")

    # Assign colors (IQVIA blue tones)
    iqvia_colors = ['#015FF1', '#0072CE', '#66B2FF', '#99CCFF', '#B2D9FF', '#CCE6FF', '#A4C8F0', '#89B7EA', '#6CA5E3', '#488FE0']
    icon_colors = [iqvia_colors[t % len(iqvia_colors)] if t >= 0 else "#999999" for t in hcp_df["Territory"]]

    m.add_points_from_xy(
        hcp_df,
        x="Longitude",
        y="Latitude",
        popup=["HCP_ID", "TRx", "Territory"],
        icon_colors=icon_colors,
        clustered=False,  # now true clusters, not visual
        layer_name="Territories"
    )

    m.to_streamlit(height=700)

    # Tabs
    tab1, tab2, tab3 = st.tabs(["📋 HCP Detail", "📊 Territory Summary", "📥 Export"])

    with tab1:
        st.subheader("HCP Listing")
        st.dataframe(hcp_df[["HCP_ID", "TRx", "Latitude", "Longitude", "Territory"]])

    with tab2:
        st.subheader("KPI Summary by Territory")
        summary = hcp_df.groupby("Territory").agg(
            Total_HCPs=("HCP_ID", "count"),
            Total_TRx=("TRx", "sum"),
            Avg_TRx_per_HCP=("TRx", "mean")
        ).reset_index()
        st.dataframe(summary)

        st.markdown("##### 📊 HCPs by Territory")
        fig = px.bar(summary, x="Territory", y="Total_HCPs", color="Territory", color_discrete_sequence=iqvia_colors)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("##### 📈 Total TRx per Territory")
        fig2 = px.pie(summary, names="Territory", values="Total_TRx", color="Territory", color_discrete_sequence=iqvia_colors)
        st.plotly_chart(fig2, use_container_width=True)

    with tab3:
        st.subheader("Download Results")
        st.download_button("📥 Download HCP CSV", hcp_df.to_csv(index=False), "hcp_clusters.csv", "text/csv")
        st.download_button("📥 Download Territory Summary", summary.to_csv(index=False), "territory_summary.csv", "text/csv")

else:
    st.info("👈 Select your algorithm and click Segment HCPs to begin.")
