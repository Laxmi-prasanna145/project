import streamlit as st
import pandas as pd

def data_upload_page():

    st.title("📂 Upload Business Dataset")

    file = st.file_uploader("Upload CSV File", type=["csv"])

    if file:

        df = pd.read_csv(file)
        st.session_state.raw_df = df

        st.success("Dataset Uploaded Successfully")
        st.dataframe(df.head())

        st.subheader("📊 Select Power BI View Type")

        powerbi_views = [
            "Report View (Detailed Visual Dashboards)",
            "Data View (Dataset Understanding)",
            "Model View (Feature Relationship Analysis)",
            "Power Query View (Data Cleaning Insights)",
            "Executive Dashboard View (Business KPI Overview)"
        ]

        st.session_state.viz = st.multiselect(
            "Select Views You Want to Generate",
            powerbi_views,
            default=["Report View (Detailed Visual Dashboards)"]
        )

        st.session_state.k = st.slider("Select Number of Clusters", 2, 8, 4)

        if st.button("Run AI Segmentation"):
            st.session_state.page = "dashboard"
            st.rerun()
