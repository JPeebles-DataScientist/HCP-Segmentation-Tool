
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
import plotly.express as px

st.set_page_config(page_title="HCP Segmentation Tool", layout="wide")
st.title("🧠 HCP Segmentation Tool")
st.markdown("Segment HCPs into optimized sales territories based on TRx and ZIP3.")

uploaded_file = st.file_uploader("Upload your HCP dataset (.csv)", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
else:
    st.info("Using mock prostate cancer dataset (1000 HCPs). Upload your file to override.")
    df = pd.DataFrame({
        "HCP_ID": [f"HCP_{i:06d}" for i in range(1000)],
        "TRx": np.random.randint(50, 500, size=1000),
        "ZIP3": np.random.choice(range(100, 999), size=1000),
        "Latitude": np.random.uniform(25.0, 49.0, 1000),
        "Longitude": np.random.uniform(-124.0, -67.0, 1000),
        "Specialty": np.random.choice(["Urology", "Oncology", "Internal Medicine"], size=1000)
    })

st.sidebar.header("Clustering Settings")
algorithm = st.sidebar.selectbox("Clustering Algorithm", ["KMeans", "Agglomerative", "DBSCAN"])
num_clusters = st.sidebar.slider("Number of Territories (Clusters)", min_value=2, max_value=10, value=4)
st.sidebar.markdown("ℹ️ **KMeans** = fast, **Agglomerative** = structured, **DBSCAN** = detects dense clusters and outliers.")

if st.button("Cluster HCPs"):
    X = df[["TRx", "ZIP3"]]
    X_scaled = StandardScaler().fit_transform(X)

    if algorithm == "KMeans":
        model = KMeans(n_clusters=num_clusters, random_state=42)
        df["Territory"] = model.fit_predict(X_scaled)
    elif algorithm == "Agglomerative":
        model = AgglomerativeClustering(n_clusters=num_clusters)
        df["Territory"] = model.fit_predict(X_scaled)
    elif algorithm == "DBSCAN":
        model = DBSCAN(eps=0.8, min_samples=10)
        df["Territory"] = model.fit_predict(X_scaled)
        df["Outlier"] = df["Territory"] == -1
        st.warning("⚠️ DBSCAN may assign some HCPs to -1 as outliers.")

    st.success(f"Clustering complete using {algorithm} algorithm.")
    st.dataframe(df.head(50))

    fig = px.scatter_mapbox(
        df,
        lat="Latitude", lon="Longitude",
        color=df["Territory"].astype(str),
        hover_data=["HCP_ID", "TRx", "Specialty"],
        mapbox_style="carto-positron",
        zoom=3,
        height=600,
        title=f"HCP Territories (Algorithm: {algorithm})"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.download_button(
        label="📥 Download Clustered Data as CSV",
        data=df.to_csv(index=False),
        file_name="clustered_hcp_data.csv",
        mime="text/csv"
    )
