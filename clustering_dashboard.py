import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
import plotly.express as px


# ---------- Preprocessing + Clustering ----------
def preprocess_and_cluster(df, k):

    data = df.copy()

    # Remove ID-like columns
    id_cols = [col for col in data.columns if "id" in col.lower()]
    data = data.drop(columns=id_cols, errors="ignore")

    # Encode categorical columns
    for col in data.select_dtypes(include=["object"]).columns:
        data[col] = LabelEncoder().fit_transform(data[col].astype(str))

    # Scale data
    scaler = StandardScaler()
    scaled = scaler.fit_transform(data)

    # KMeans clustering
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    df["Cluster"] = model.fit_predict(scaled)

    return df


# ---------- Auto Summary Generator ----------
def generate_summary(cluster_df):

    size = len(cluster_df)

    summary_text = f"Segment Size: {size} customers.\n"

    if "SpendingScore" in cluster_df.columns:
        avg_spend = cluster_df["SpendingScore"].mean()
        summary_text += f"Average Spending Score: {avg_spend:.2f}\n"

        if avg_spend > 60:
            summary_text += "Suggested Strategy: Focus on Premium Retention Programs."
        elif avg_spend > 40:
            summary_text += "Suggested Strategy: Introduce Personalized Marketing."
        else:
            summary_text += "Suggested Strategy: Offer Discount & Engagement Campaigns."

    return summary_text


# ---------- Dashboard ----------
def clustering_dashboard():

    st.title("📊 Customer Segmentation Dashboard")

    df = st.session_state.raw_df.copy()
    k = st.session_state.k
    viz = st.session_state.viz

    # Run clustering
    df = preprocess_and_cluster(df, k)

    # ---------- GLOBAL VISUALIZATION ----------
    if "Market Share Pie" in viz:
        st.subheader("🌍 Overall Market Share Distribution")

        fig = px.pie(
            df,
            names="Cluster",
            title="Customer Distribution Across Segments",
            hole=0.4
        )

        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # ---------- CLUSTER TABS ----------
    st.subheader("📌 Cluster Level Analysis")

    tabs = st.tabs([f"Cluster {i}" for i in range(k)])

    for i in range(k):

        cluster_df = df[df["Cluster"] == i]

        with tabs[i]:

            st.markdown(f"### 🔍 Insights for Cluster {i}")

            # ---------- Scatter ----------
            if "Income vs Spending Scatter" in viz:
                if {"AnnualIncome", "SpendingScore"}.issubset(cluster_df.columns):

                    fig = px.scatter(
                        cluster_df,
                        x="AnnualIncome",
                        y="SpendingScore",
                        title=f"Cluster {i} : Income vs Spending"
                    )

                    st.plotly_chart(
                        fig,
                        use_container_width=True,
                        key=f"scatter_{i}"
                    )

            # ---------- Age Distribution ----------
            if "Age Distribution" in viz:
                if "Age" in cluster_df.columns:

                    fig = px.histogram(
                        cluster_df,
                        x="Age",
                        title=f"Cluster {i} : Age Distribution"
                    )

                    st.plotly_chart(
                        fig,
                        use_container_width=True,
                        key=f"age_{i}"
                    )

            # ---------- Heatmap ----------
            if "Feature Correlation Heatmap" in viz:

                numeric_df = cluster_df.select_dtypes(include="number")

                if len(numeric_df.columns) > 1:
                    corr = numeric_df.corr()

                    fig = px.imshow(
                        corr,
                        text_auto=True,
                        title=f"Cluster {i} : Feature Correlation"
                    )

                    st.plotly_chart(
                        fig,
                        use_container_width=True,
                        key=f"heatmap_{i}"
                    )

            # ---------- AI Summary ----------
            st.info(generate_summary(cluster_df))
