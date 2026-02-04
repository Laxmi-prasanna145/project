import streamlit as st


def show_login():

    st.title("🔐 ClusterPro AI Login")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):

        # Trim accidental spaces
        username = username.strip()
        password = password.strip()

        # Dummy authentication (replace later with DB if needed)
        if username == "admin" and password == "admin123":

            st.session_state.auth = True
            st.session_state.step = "upload"

            st.success("Login Successful ✅")
            st.rerun()

        else:
            st.error("Invalid Username or Password ❌")
