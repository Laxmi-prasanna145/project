import streamlit as st

def show_login():
    st.markdown('''
        <style>
        .stApp { background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%); }
        .login-card { background: white; padding: 50px; border-radius: 15px; box-shadow: 0 10px 30px rgba(0,0,0,0.3); color: #333; max-width: 450px; margin: auto; text-align: center; }
        .stButton>button { background-color: #1e3c72; color: white; width: 100%; border-radius: 8px; font-weight: bold; }
        </style>
    ''', unsafe_allow_html=True)
    
    st.write("#") # Spacer
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown('<div class="login-card">', unsafe_allow_html=True)
        st.title("🛡️ ClusterPro AI")
        st.subheader("Business Analyst Portal")
        user = st.text_input("Username", placeholder="admin")
        pw = st.text_input("Password", type="password", placeholder="admin123")
        if st.button("AUTHENTICATE"):
            if user == "admin" and pw == "admin123":
                st.session_state.auth = True
                st.rerun()
            else:
                st.error("Access Denied: Invalid Credentials")
        st.markdown('</div>', unsafe_allow_html=True)
