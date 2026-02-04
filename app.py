import streamlit as st
from auth import show_login
from data_io import show_upload
from ml_logic import run_clustering
from visuals import show_results

# Set page config
st.set_page_config(page_title="ClusterPro AI", layout="wide")

# Initialize session states
if 'auth' not in st.session_state: st.session_state.auth = False
if 'step' not in st.session_state: st.session_state.step = "upload"

def main():
    if not st.session_state.auth:
        show_login()
    elif st.session_state.step == "upload":
        show_upload()
    elif st.session_state.step == "process":
        with st.spinner("AI is analyzing behavioral segments..."):
            # Execute ML Logic
            df, scaled, feats, names_map = run_clustering(st.session_state.raw_df, st.session_state.k)
            st.session_state.results = (df, scaled, feats, names_map)
            st.session_state.step = "results"
            st.rerun()
    elif st.session_state.step == "results":
        df, scaled, feats, names_map = st.session_state.results
        show_results(df, scaled, feats, st.session_state.viz_choices, st.session_state.k, names_map)

if __name__ == "__main__":
    main()
