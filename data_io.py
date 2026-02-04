import streamlit as st
import pandas as pd

def show_upload():
    st.title("📂 Step 1: Data Acquisition")
    uploaded_file = st.file_uploader("Upload your Customer Dataset (CSV)", type="csv")
    
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.session_state.raw_df = df
        st.success("Dataset Loaded Successfully!")
        
        st.divider()
        st.subheader("Step 2: Analysis Configuration")
        st.session_state.viz_choices = st.multiselect(
            "Select Visualizations for your Dashboard:",
            ["Market Share Pie", "Income vs Spend Scatter", "Age Distribution", "Behavioral Radar Chart"],
            default=["Market Share Pie"]
        )
        
        st.session_state.k = st.slider("Target Number of Segments (K)", 2, 8, 5)
        
        if st.button("Start AI Clustering Analysis"):
            st.session_state.step = "process"
            st.rerun()
