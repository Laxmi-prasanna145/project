import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def get_elbow_data(df):
    numeric_df = df.select_dtypes(include=[np.number]).dropna()
    scaler = StandardScaler()
    scaled = scaler.fit_transform(numeric_df)
    wcss = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42, n_init=10)
        kmeans.fit(scaled)
        wcss.append(kmeans.inertia_)
    return wcss

def show_upload():
    st.title("📂 Step 1: Data Configuration")
    uploaded_file = st.file_uploader("Upload Customer Dataset (CSV)", type="csv")
    
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.session_state.raw_df = df
        st.success("Dataset ready for analysis.")
        
        st.divider()
        st.subheader("Step 2: Business Customization")
        st.session_state.viz_choices = st.multiselect(
            "Select Dashboards to Generate:",
            ["PCA Global Separation", "Correlation Heatmap", "Economic Scatter Plot", "Behavioral Radar Chart", "Spend Box Plots"],
            default=["PCA Global Separation", "Behavioral Radar Chart"]
        )
        
        if st.checkbox("Show Elbow Method Plot (Select Optimal Segments)"):
            wcss = get_elbow_data(df)
            fig = px.line(x=range(1, 11), y=wcss, markers=True, title="Elbow Method: Finding the 'Bend' for Optimal K")
            st.plotly_chart(fig)

        st.session_state.k = st.number_input("Final Number of Segments (K)", 2, 10, 5)
        
        if st.button("GENERATE DASHBOARDS"):
            st.session_state.step = "process"
            st.rerun()
