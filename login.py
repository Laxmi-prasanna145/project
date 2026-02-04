import streamlit as st

def login_page():

    st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg,#1f4037,#99f2c8);
    }

    .login-box {
        background: white;
        padding: 40px;
        border-radius: 15px;
        box-shadow: 0px 10px 30px rgba(0,0,0,0.2);
        text-align:center;
    }
    </style>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1,2,1])

    with col2:
        st.markdown("<div class='login-box'>", unsafe_allow_html=True)

        st.title("🔐 Business Analyst Portal")
        st.write("Login to continue")

        username = st.text_input("Username")
        password = st.text_input("Password", type="password")

        if st.button("Login"):
            if username == "admin" and password == "admin123":
                st.session_state.logged_in = True
                st.session_state.page = "upload"
                st.success("Login Successful")
                st.rerun()
            else:
                st.error("Invalid Credentials")

        st.markdown("</div>", unsafe_allow_html=True)
