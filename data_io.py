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
    wcss = [KMeans(n_clusters=i, init='k-means++', random_state=42, n_init=10).fit(scaled).inertia_ for i in range(1, 11)]
    return wcss

def show_upload():
    st.title("📂 Data Configuration")
    # Added unique key to file_uploader
    uploaded_file = st.file_uploader("Upload Customer Dataset (CSV)", type="csv", key="main_uploader")
    
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.session_state.raw_df = df
        
        st.session_state.viz_choices = st.multiselect(
            "Select Dashboards:", 
            ["PCA Separation", "Correlation Heatmap", "Economic Analysis", "Behavioral Radar"],
            default=["PCA Separation", "Behavioral Radar"],
            key="viz_selector"
        )
        
        if st.checkbox("Show Elbow Method Plot", key="elbow_toggle"):
            st.plotly_chart(px.line(x=range(1, 11), y=get_elbow_data(df), markers=True, title="Optimal K Search"))
        
        st.session_state.k = st.number_input("Target Number of Clusters (K)", 2, 10, 5, key="k_input")
        
        if st.button("RUN AI ANALYSIS", key="process_start"):
            st.session_state.step = "process"
            st.rerun()
