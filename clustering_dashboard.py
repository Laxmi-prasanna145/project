import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
import plotly.express as px
import plotly.graph_objects as go


# ---------- Preprocessing + Clustering ----------
def preprocess_and_cluster(df, k):

    data = df.copy()

    id_cols = [c for c in data.columns if "id" in c.lower()]
    data = data.drop(columns=id_cols, errors="ignore")

    for col in data.select_dtypes(include=["object"]).columns:
        data[col] = LabelEncoder().fit_transform(data[col].astype(str))

    scaler = StandardScaler()
    scaled = scaler.fit_transform(data)

    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    df["Cluster"] = model.fit_predict(scaled)

    return df


# ---------- AI Summary Generator ----------
def generate_summary(cluster_df):

    size = len(cluster_df)

    summary = f"""
    🔍 This segment consists of **{size} customers**, representing a unique behavioral group.

    📊 The purchasing behaviour of this cluster shows distinct spending and demographic patterns compared to other clusters.

    💡 Customers in this segment demonstrate specific engagement tendencies which can help businesses target marketing campaigns more effectively.

    📈 The spending and income patterns indicate potential opportunities for revenue optimization and customer retention strategies.

    🎯 Business Recommendation: Personalized offers, loyalty programs, and targeted promotions can significantly improve engagement with this segment.
    """

    return summary


# ---------- Dashboard ----------
def clustering_dashboard():

    st.title("📊 AI Customer Segmentation Platform")

    df = st.session_state.raw_df.copy()
    k = st.session_state.k
    views = st.session_state.viz

    df = preprocess_and_cluster(df, k)

    # ================= REPORT VIEW =================
    if "Report View (Detailed Visual Dashboards)" in views:

        st.header("📑 Report View – Cluster Visual Analysis")

        fig = px.pie(df, names="Cluster", hole=0.4,
                     title="Customer Segment Distribution")
        st.plotly_chart(fig, use_container_width=True)

        tabs = st.tabs([f"Cluster {i}" for i in range(k)])

        for i in range(k):

            cluster_df = df[df["Cluster"] == i]

            with tabs[i]:

                if {"AnnualIncome", "SpendingScore"}.issubset(df.columns):

                    fig = px.scatter(
                        cluster_df,
                        x="AnnualIncome",
                        y="SpendingScore",
                        title=f"Cluster {i} Spending Pattern"
                    )

                    st.plotly_chart(fig, use_container_width=True, key=f"scatter_{i}")

                if "Age" in df.columns:

                    fig = px.histogram(cluster_df, x="Age",
                                       title=f"Cluster {i} Age Distribution")

                    st.plotly_chart(fig, use_container_width=True, key=f"age_{i}")

                st.info(generate_summary(cluster_df))


    # ================= DATA VIEW =================
    if "Data View (Dataset Understanding)" in views:

        st.header("📂 Data View – Dataset Exploration")
        st.dataframe(df)
        st.write("Dataset Shape:", df.shape)
        st.write("Column Data Types:")
        st.write(df.dtypes)


    # ================= MODEL VIEW =================
    if "Model View (Feature Relationship Analysis)" in views:

        st.header("🔗 Model View – Feature Relationships")

        corr = df.select_dtypes(include="number").corr()

        fig = px.imshow(corr, text_auto=True, title="Feature Relationship Heatmap")
        st.plotly_chart(fig, use_container_width=True)


    # ================= POWER QUERY VIEW =================
    if "Power Query View (Data Cleaning Insights)" in views:

        st.header("🧹 Power Query View – Data Cleaning Insights")

        missing = df.isnull().sum().reset_index()
        missing.columns = ["Column", "Missing Values"]

        fig = px.bar(missing, x="Column", y="Missing Values",
                     title="Missing Data Analysis")

        st.plotly_chart(fig, use_container_width=True)

        st.write("Duplicate Rows:", df.duplicated().sum())


    # ================= EXECUTIVE DASHBOARD =================
    if "Executive Dashboard View (Business KPI Overview)" in views:

        st.header("📊 Executive KPI Dashboard")

        total_customers = len(df)
        avg_cluster = df["Cluster"].nunique()

        col1, col2 = st.columns(2)

        col1.metric("Total Customers", total_customers)
        col2.metric("Total Segments", avg_cluster)

        cluster_counts = df["Cluster"].value_counts().reset_index()
        cluster_counts.columns = ["Cluster", "Customers"]

        fig = px.bar(cluster_counts, x="Cluster", y="Customers",
                     title="Customers Per Segment")

        st.plotly_chart(fig, use_container_width=True)
