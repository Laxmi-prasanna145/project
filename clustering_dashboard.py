import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram, linkage
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt


# ---------- PREPROCESS ----------
def preprocess(df):

    data = df.copy()

    id_cols = [c for c in data.columns if "id" in c.lower()]
    data = data.drop(columns=id_cols, errors="ignore")

    for col in data.select_dtypes(include=["object"]).columns:
        data[col] = LabelEncoder().fit_transform(data[col].astype(str))

    scaler = StandardScaler()
    scaled = scaler.fit_transform(data)

    return scaled, data.columns


# ---------- CLUSTER DEFINITION ----------
def define_clusters(df):

    numeric_df = df.select_dtypes(include=np.number)

    cluster_profiles = numeric_df.groupby(df["Cluster"]).mean()

    cluster_labels = {}

    for cluster in cluster_profiles.index:

        mean_values = cluster_profiles.loc[cluster]
        overall_mean = numeric_df.mean()

        score = (mean_values - overall_mean).mean()

        if score > 0.5:
            label = "High Value / Premium Customers"
        elif score > 0:
            label = "Moderate Engagement Customers"
        else:
            label = "Low Value / Price Sensitive Customers"

        cluster_labels[cluster] = label

    return cluster_profiles, cluster_labels


# ---------- AI VISUAL SUMMARY ----------
def visualization_summary(title):

    summaries = {

        "PCA": """
        This visualization shows how well customer segments are separated in reduced dimensional space.
        Distinct cluster separation indicates strong segmentation quality.
        Overlapping clusters may indicate similar behavioral patterns between segments.
        The distance between clusters reflects customer diversity across segments.
        Clear separation suggests highly differentiated customer groups.
        """,

        "Radar": """
        This chart compares average behavioral characteristics across customer segments.
        It highlights which features dominate each cluster.
        Wide spread between cluster lines indicates strong segment differentiation.
        Similar patterns suggest overlapping customer behavior.
        Helps identify strengths and weaknesses of each segment.
        """,

        "Heatmap": """
        This heatmap reveals relationships between customer features.
        Strong correlations suggest influencing variables for segmentation.
        Helps analysts understand feature dependencies.
        Highly correlated features may indicate redundant variables.
        Identifies key drivers of customer behaviour.
        """,

        "Box": """
        This boxplot shows the distribution of selected features across clusters.
        It helps identify variability, median behaviour and outliers.
        Wide spread indicates diverse customer behaviour.
        Outliers highlight exceptional customer groups.
        Useful for risk and opportunity identification.
        """,

        "Elbow": """
        The elbow plot helps determine the optimal number of clusters.
        The bending point indicates ideal segmentation complexity.
        Too many clusters cause over segmentation.
        Too few clusters hide meaningful customer differences.
        This ensures scientifically justified clustering.
        """,

        "Dendrogram": """
        This hierarchical visualization shows nested cluster relationships.
        It helps understand customer similarity hierarchy.
        Branch length represents behavioural distance.
        Useful for exploratory segmentation validation.
        Shows potential alternative clustering structures.
        """
    }

    return summaries.get(title, "")


# ---------- DASHBOARD ----------
def clustering_dashboard():

    st.title("🔬 Customer Segmentation Analytics")

    df = st.session_state.raw_df.copy()
    k = st.session_state.k
    viz = st.session_state.viz

    scaled, features = preprocess(df)

    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    df["Cluster"] = model.fit_predict(scaled)

    cluster_profiles, cluster_labels = define_clusters(df)

    # ---------- SHOW CLUSTER DEFINITIONS ----------
    st.header("🧠 Cluster Definitions")

    for cluster, label in cluster_labels.items():

        st.subheader(f"Cluster {cluster} : {label}")

        st.write(cluster_profiles.loc[cluster])
        st.info(f"""
        This segment represents customers grouped based on similar behavioural and demographic attributes.
        The profile indicates dominant purchasing characteristics and engagement trends.
        Businesses can tailor marketing and product strategies according to this segment behaviour.
        Segment specific campaigns improve conversion and customer retention.
        This cluster plays a strategic role in overall revenue optimization.
        """)

    st.divider()

    # ---------- PCA ----------
    if "PCA Cluster Scatter" in viz:

        st.header("📍 PCA Cluster Separation")

        pca = PCA(n_components=2)
        pca_data = pca.fit_transform(scaled)

        pca_df = pd.DataFrame(pca_data, columns=["PC1", "PC2"])
        pca_df["Cluster"] = df["Cluster"]

        fig = px.scatter(pca_df, x="PC1", y="PC2",
                         color=pca_df["Cluster"].astype(str))

        st.plotly_chart(fig, use_container_width=True)

        st.info(visualization_summary("PCA"))

    # ---------- RADAR ----------
    if "Radar Cluster Profile" in viz:

        st.header("🕸 Cluster Behaviour Radar")

        fig = go.Figure()

        for cluster in cluster_profiles.index:

            fig.add_trace(go.Scatterpolar(
                r=cluster_profiles.loc[cluster].values,
                theta=cluster_profiles.columns,
                fill='toself',
                name=f"Cluster {cluster}"
            ))

        st.plotly_chart(fig, use_container_width=True)

        st.info(visualization_summary("Radar"))

    # ---------- HEATMAP ----------
    if "Correlation Heatmap" in viz:

        st.header("🔥 Feature Correlation")

        corr = df.select_dtypes(include=np.number).corr()
        fig = px.imshow(corr, text_auto=True)
        st.plotly_chart(fig, use_container_width=True)

        st.info(visualization_summary("Heatmap"))

    # ---------- ELBOW ----------
    if "Elbow Method Analysis" in viz:

        st.header("📉 Optimal Cluster Identification")

        sse = []
        k_range = range(1, 10)

        for i in k_range:
            km = KMeans(n_clusters=i, n_init=10)
            km.fit(scaled)
            sse.append(km.inertia_)

        fig = px.line(x=list(k_range), y=sse)
        st.plotly_chart(fig, use_container_width=True)

        st.info(visualization_summary("Elbow"))

    # ---------- BOX ----------
    if "Cluster Distribution Boxplots" in viz:

        st.header("📦 Cluster Feature Distribution")

        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        feature = st.selectbox("Select Feature", numeric_cols)

        fig = px.box(df, x="Cluster", y=feature)
        st.plotly_chart(fig, use_container_width=True)

        st.info(visualization_summary("Box"))

    # ---------- DENDROGRAM ----------
    if "Hierarchical Dendrogram" in viz:

        st.header("🌳 Hierarchical Cluster Structure")

        linked = linkage(scaled, method='ward')

        fig, ax = plt.subplots(figsize=(10, 5))
        dendrogram(linked)
        st.pyplot(fig)

        st.info(visualization_summary("Dendrogram"))
