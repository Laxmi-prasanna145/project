import streamlit as st
from auth import show_login
from data_io import show_upload
from ml_logic import run_clustering
from visuals import show_results

# Initialize session states
if 'auth' not in st.session_state: st.session_state.auth = False
if 'step' not in st.session_state: st.session_state.step = "upload"

def main():
    if not st.session_state.auth:
        show_login()
    elif st.session_state.step == "upload":
        show_upload()
    elif st.session_state.step == "process":
        with st.spinner("AI is calculating segments..."):
            df, scaled, features = run_clustering(st.session_state.raw_df, st.session_state.k)
            st.session_state.results = (df, scaled, features)
            st.session_state.step = "results"
            st.rerun()
    elif st.session_state.step == "results":
        df, scaled, features = st.session_state.results
        show_results(df, scaled, features, st.session_state.viz_choices, st.session_state.k)

if __name__ == "__main__":
    main()
