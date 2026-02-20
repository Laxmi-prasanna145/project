import streamlit as st
from auth import show_login
from data_io import show_upload
from ml_logic import run_clustering
from visuals import show_results

st.set_page_config(page_title="Business Analyst")

# -------------------------
# SESSION STATE INITIALIZATION
# -------------------------

if "auth" not in st.session_state:
    st.session_state.auth = False

if "step" not in st.session_state:
    st.session_state.step = "upload"


# -------------------------
# LOGIN CHECK
# -------------------------

if not st.session_state.auth:
    show_login()


# -------------------------
# MAIN APPLICATION
# -------------------------

st.title("Customer Segmentation and Targeting")

# Logout Button
col1, col2 = st.columns([6, 2])

with col2:
    if st.button("Logout"):
        st.session_state.auth = False
        st.session_state.step = "upload"
        st.rerun()


# -------------------------
# STEP FLOW
# -------------------------

if st.session_state.step == "upload":
    show_upload()

elif st.session_state.step == "process":
    with st.spinner("Analyzing customer behaviors..."):
        df, scaled, feats, names_map = run_clustering(
            st.session_state.raw_df,
            st.session_state.k
        )

        st.session_state.results = (df, scaled, feats, names_map)
        st.session_state.step = "results"
        st.rerun()

elif st.session_state.step == "results":
    df, scaled, feats, names_map = st.session_state.results
    show_results(
        df,
        scaled,
        feats,
        st.session_state.viz_choices,
        st.session_state.k,
        names_map
    )
