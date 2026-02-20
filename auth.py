import streamlit as st


def show_login():
    st.title("Business Analyst Login")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username == "admin" and password == "admin123":
            st.session_state.auth = True
            st.rerun()
        else:
            st.error("Invalid Credentials")

    st.stop()  # 🚨 Stops app until login success
