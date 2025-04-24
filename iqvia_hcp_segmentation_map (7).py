
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN

st.set_page_config(page_title="HCP Segmentation Tool", layout="wide")
st.title("HCP Segmentation Tool")

st.sidebar.header("Clustering Settings")
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

    iqvia_colors = ['#015FF1', '#0072CE', '#66B2FF', '#99CCFF', '#B2D9FF']

    tab1, tab2, tab3 = st.tabs(["📋 HCP Table", "📊 Territory Summary", "📥 Export"])

    with tab1:
        st.subheader("HCP Listing")
        st.dataframe(hcp_df[["HCP_ID", "TRx", "Latitude", "Longitude", "Territory"]])

    with tab2:
        st.subheader("Territory KPI Summary")
        summary = hcp_df.groupby("Territory").agg(
            Total_HCPs=("HCP_ID", "count"),
            Total_TRx=("TRx", "sum"),
            Avg_TRx_per_HCP=("TRx", "mean")
        ).reset_index()
        st.dataframe(summary)

        st.plotly_chart(
            px.bar(summary, x="Territory", y="Total_HCPs", color="Territory", color_discrete_sequence=iqvia_colors),
            use_container_width=True
        )

        st.plotly_chart(
            px.pie(summary, names="Territory", values="Total_TRx", color="Territory", color_discrete_sequence=iqvia_colors),
            use_container_width=True
        )

    with tab3:
        st.subheader("Download Data")
        st.download_button("📥 Download HCP CSV", hcp_df.to_csv(index=False), "hcp_clusters.csv", "text/csv")
        st.download_button("📥 Download Territory Summary", summary.to_csv(index=False), "territory_summary.csv", "text/csv")

else:
    st.info("👈 Set clustering parameters and click to segment HCPs.")
