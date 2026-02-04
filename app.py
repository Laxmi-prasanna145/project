import streamlit as st
from login import login_page
from data_page import data_upload_page
from clustering_dashboard import clustering_dashboard

st.set_page_config(page_title="AI Customer Segmentation", layout="wide")

# Session states
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "page" not in st.session_state:
    st.session_state.page = "login"


def main():

    if not st.session_state.logged_in:
        login_page()

    elif st.session_state.page == "upload":
        data_upload_page()

    elif st.session_state.page == "dashboard":
        clustering_dashboard()


if __name__ == "__main__":
    main()
