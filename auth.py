import streamlit as st

# -------------------------
# DEFAULT LOGIN PAGE
# -------------------------

if "auth" not in st.session_state:
    st.session_state.auth = False

if not st.session_state.auth:
    st.title("Business Analyst")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username == "admin" and password == "admin123":
            st.session_state.auth = True
            st.rerun()
        else:
            st.error("Invalid Credentials")

    st.stop()   # This makes login page default

# -------------------------
# MAIN APPLICATION (Only runs after login)
# -------------------------

st.title("Customer Segmentation Dashboard")
st.success("Welcome! You are logged in.")

# Optional logout
if st.button("Logout"):
    st.session_state.auth = False
    st.rerun()
