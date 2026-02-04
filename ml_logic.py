import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
import plotly.express as px


def preprocess_and_cluster(df, k):

    data = df.copy()

    # Remove ID columns
    id_cols = [col for col in data.columns if "id" in col.lower()]
    data = data.drop(columns=id_cols, errors="ignore")

    # Encode categorical
    for col in data.select_dtypes(include=["object"]).columns:
        data[col] = LabelEncoder().fit_transform(data[col])

    # Scaling
    scaler = StandardScaler()
    scaled = scaler.fit_transform(data)

    # KMeans
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    df["Cluster"] = model.fit_predict(scaled)

    return df


def generate_summary(cluster_df):

    size = len(cluster_df)

    if "SpendingScore" in cluster_df.columns:
        avg_spend = cluster_df["SpendingScore"].mean()

        if avg_spend > 50:
            strategy = "Premium Loyalty Programs"
        else:
            strategy = "Discount Marketing Campaigns"

        return f"""
        Segment Size : {size} customers  
        Average Spending Score : {avg_spend:.2f}  
        Suggested Strategy : {strategy}
        """

    return f"Segment contains {size} customers."


def clustering_dashboard():

    st.title("📊 Customer Segmentation Dashboard")

    df = st.session_state.raw_df
    k = st.session_state.k
    viz = st.session_state.viz

    df = preprocess_and_cluster(df, k)

    tabs = st.tabs([f"Cluster {i}" for i in range(k)])

    for i in range(k):

        cluster_df = df[df["Cluster"] == i]

        with tabs[i]:

            st.subheader(f"📌 Cluster {i} Analysis")

            if "Market Share Pie" in viz:
                fig = px.pie(df, names="Cluster", title="Market Share")
                st.plotly_chart(fig, use_container_width=True)

            if "Income vs Spending Scatter" in viz:
                if {"AnnualIncome","SpendingScore"}.issubset(df.columns):
                    fig = px.scatter(
                        cluster_df,
                        x="AnnualIncome",
                        y="SpendingScore",
                        title="Income vs Spending"
                    )
                    st.plotly_chart(fig, use_container_width=True)

            if "Age Distribution" in viz:
                if "Age" in df.columns:
                    fig = px.histogram(cluster_df, x="Age", title="Age Distribution")
                    st.plotly_chart(fig, use_container_width=True)

            if "Feature Correlation Heatmap" in viz:
                corr = cluster_df.select_dtypes(include="number").corr()
                fig = px.imshow(corr, text_auto=True)
                st.plotly_chart(fig, use_container_width=True)

            st.info(generate_summary(cluster_df))
