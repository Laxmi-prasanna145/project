import streamlit as st

def show_login():
    st.markdown('''
        <style>
        .stApp { background: linear-gradient(to right, #1e3c72, #2a5298); color: white; }
        .login-card { background: white; padding: 40px; border-radius: 15px; box-shadow: 0 10px 25px rgba(0,0,0,0.2); color: #333; max-width: 400px; margin: auto; text-align: center; }
        .stButton>button { background-color: #1e3c72; color: white; width: 100%; border-radius: 5px; }
        </style>
    ''', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown('<div class="login-card">', unsafe_allow_html=True)
        st.title("🚀 Analyst Portal")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.button("Login"):
            if username == "admin" and password == "admin123":
                st.session_state.auth = True
                st.rerun()
            else:
                st.error("Invalid credentials")
        st.markdown('</div>', unsafe_allow_html=True)
