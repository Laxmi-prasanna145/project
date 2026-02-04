import streamlit as st
import pandas as pd

def data_upload_page():

    st.title("📂 Upload Customer Dataset")

    file = st.file_uploader("Upload CSV File", type=["csv"])

    if file:

        df = pd.read_csv(file)
        st.session_state.raw_df = df

        st.success("Dataset Uploaded Successfully")
        st.dataframe(df.head())

        st.subheader("📊 Select Required Visualizations")

        viz_options = [
            "Market Share Pie",
            "Income vs Spending Scatter",
            "Age Distribution",
            "Feature Correlation Heatmap"
        ]

        st.session_state.viz = st.multiselect(
            "Select Visualizations",
            viz_options,
            default=["Market Share Pie"]
        )

        st.session_state.k = st.slider("Select Number of Clusters", 2, 8, 4)

        if st.button("Run Clustering"):
            st.session_state.page = "dashboard"
            st.rerun()
