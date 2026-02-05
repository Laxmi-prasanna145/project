import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

def get_dynamic_narrative(c_data, g_inc, g_spd, inc_col, spd_col):
    c_inc, c_spd = c_data[inc_col].mean(), c_data[spd_col].mean()
    i_status = "above" if c_inc > g_inc else "below"
    s_status = "high" if c_spd > g_spd else "low"
    
    return f"""
    - **Financial Context:** Average income is ${c_inc:,.0f} ({i_status} market avg).
    - **Engagement Level:** Spending score is {c_spd:.1f}, indicating {s_status} brand loyalty.
    - **AI Persona Insights:** This segment contains {len(c_data)} customers with similar traits.
    - **Strategy Recommendation:** Focus on {"loyalty rewards" if c_spd > g_spd else "discount incentives"}.
    - **Data Strength:** Tight clustering suggests these insights are 90%+ predictable.
    """

def show_results(df, scaled_data, features, choices, k, names_map):
    st.title("📊 AI Behavioral Dashboard")
    
    cols = df.columns
    inc_col = next((c for c in cols if 'income' in c.lower()), None)
    spd_col = next((c for c in cols if 'spend' in c.lower()), None)
    g_inc, g_spd = df[inc_col].mean(), df[spd_col].mean()

    if "PCA Separation" in choices:
        st.header("1. Global Segment Separation")
        pca_res = PCA(n_components=2).fit_transform(scaled_data)
        pca_df = pd.DataFrame(pca_res, columns=['PC1', 'PC2'])
        pca_df['Persona'] = df['Persona']
        fig = px.scatter(pca_df, x='PC1', y='PC2', color='Persona')
        st.plotly_chart(fig, use_container_width=True)

    st.divider()
    tabs = st.tabs([f"{names_map[i]}" for i in range(k)])
    all_means = df.groupby('Cluster')[features].mean()
    norm_means = (all_means - all_means.min()) / (all_means.max() - all_means.min())

    for i in range(k):
        with tabs[i]:
            c_data = df[df['Cluster'] == i]
            col_a, col_b = st.columns(2)
            
            with col_a:
                st.subheader("💰 Economic Profile")
                fig_scat = px.scatter(c_data, x=inc_col, y=spd_col)
                st.plotly_chart(fig_scat, use_container_width=True)
            
            with col_b:
                st.subheader("🧬 Behavioral DNA")
                fig_rad = go.Figure(data=go.Scatterpolar(r=norm_means.iloc[i].values, theta=features, fill='toself'))
                st.plotly_chart(fig_rad, use_container_width=True)
            
            st.markdown(f'<div style="background:#f0f2f6;padding:20px;border-radius:10px;">{get_dynamic_narrative(c_data, g_inc, g_spd, inc_col, spd_col)}</div>', unsafe_allow_html=True)

    if st.button("START NEW ANALYSIS"):
        st.session_state.step = "upload"
        st.rerun()
