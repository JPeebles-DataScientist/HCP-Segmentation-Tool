
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import json
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN

st.set_page_config(page_title="HCP Segmentation by ZIP3", layout="wide")
st.title("🧠 HCP Segmentation & Territory Visualization")
st.caption("Now clustering by geo-coordinates (Lat, Lon) and TRx for compact, field-rep-friendly territories.")

# Sidebar: clustering setup
st.sidebar.header("Segmentation Settings")
algorithm = st.sidebar.selectbox("Clustering Algorithm", ["KMeans", "Agglomerative", "DBSCAN"])

descriptions = {
    "KMeans": "📊 **KMeans** clusters by minimizing variance within each group — fast, best for evenly distributed data.",
    "Agglomerative": "🌲 **Agglomerative Clustering** builds a hierarchy of clusters by successively merging nearest groups — more flexible.",
    "DBSCAN": "🔍 **DBSCAN** groups dense regions and marks outliers — best when data has natural clusters and noise."
}
st.sidebar.markdown(descriptions[algorithm])

num_clusters = st.sidebar.slider("Number of Territories", min_value=2, max_value=10, value=4)
show_hcp_dots = st.sidebar.checkbox("Show HCP dots on map", value=True)

# Load ZIP3 polygons + summary metrics
@st.cache_data
def load_geo_and_metrics():
    metrics = pd.read_csv("zip3_territory_metrics.csv", dtype={"ZIP3": str})
    with open("zip3_mock_territories.geojson") as f:
        geojson = json.load(f)
    return metrics, geojson

zip3_df, geojson_data = load_geo_and_metrics()

# Simulate HCP-level data with geolocation
hcp_df = pd.DataFrame({
    "HCP_ID": [f"HCP_{i:05d}" for i in range(5000)],
    "TRx": np.random.randint(50, 500, size=5000),
    "ZIP3": np.random.choice(zip3_df["ZIP3"], size=5000),
})
hcp_df = pd.merge(hcp_df, zip3_df[["ZIP3"]], on="ZIP3")

# Assign synthetic ZIP3 centroids
zip3_coords = {}
for feature in geojson_data["features"]:
    zip3 = feature["properties"]["ZIP3"]
    coords = np.mean(np.array(feature["geometry"]["coordinates"][0]), axis=0)
    zip3_coords[zip3] = coords

hcp_df["Latitude"] = hcp_df["ZIP3"].map(lambda z: zip3_coords[z][1])
hcp_df["Longitude"] = hcp_df["ZIP3"].map(lambda z: zip3_coords[z][0])

if st.button("🔍 Cluster HCPs"):
    # Clustering
    features = hcp_df[["Latitude", "Longitude", "TRx"]].copy()
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

    # Summary per ZIP3
    zip3_summary = hcp_df.groupby(["ZIP3", "Territory"]).agg(
        HCP_Count=("HCP_ID", "count"),
        Total_TRx=("TRx", "sum")
    ).reset_index()

    # Map
    st.subheader("🗺️ Territory Map by ZIP3")
    fig = px.choropleth_mapbox(
        zip3_summary,
        geojson=geojson_data,
        locations="ZIP3",
        featureidkey="properties.ZIP3",
        color="Territory",
        hover_data=["ZIP3", "Territory", "HCP_Count", "Total_TRx"],
        color_continuous_scale="Viridis",
        mapbox_style="carto-positron",
        zoom=3,
        center={"lat": 38.0, "lon": -96.0},
        opacity=0.6,
        height=650
    )

    if show_hcp_dots:
        fig.add_trace(px.scatter_mapbox(
            hcp_df,
            lat="Latitude",
            lon="Longitude",
            hover_name="HCP_ID",
            hover_data=["TRx", "ZIP3", "Territory"],
            color=hcp_df["Territory"].astype(str),
            size_max=5,
            zoom=3
        ).data[0])

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("📄 HCP Listing by Territory")
    st.dataframe(hcp_df[["HCP_ID", "TRx", "ZIP3", "Territory", "Latitude", "Longitude"]])

    st.subheader("📋 Territory Summary Table")
    summary = hcp_df.groupby("Territory").agg(
        ZIP3s=("ZIP3", lambda x: ', '.join(sorted(set(x.astype(str))))),
        Total_HCPs=("HCP_ID", "count"),
        Total_TRx=("TRx", "sum")
    ).reset_index()
    st.dataframe(summary)
else:
    st.info("👆 Select your clustering settings and click the button to generate territories.")
