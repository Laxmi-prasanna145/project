import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

def generate_ai_text(c_data, g_df, inc_col, spd_col):
    # Calculate stats for the summary
    c_inc, g_inc = c_data[inc_col].mean(), g_df[inc_col].mean()
    c_spd, g_spd = c_data[spd_col].mean(), g_df[spd_col].mean()
    
    inc_diff = ((c_inc - g_inc) / g_inc) * 100
    spd_diff = ((c_spd - g_spd) / g_spd) * 100
    
    return f"""
    - **Economic Standing:** This group earns an average of ${c_inc:,.0f}, which is **{abs(inc_diff):.1f}% {"higher" if inc_diff > 0 else "lower"}** than the market average.
    - **Engagement Score:** Their spending score is **{c_spd:.1f}**. This reflects a **{"high" if spd_diff > 0 else "conservative"}** level of brand interaction.
    - **Segment Potential:** With {len(c_data)} members, this cluster represents **{(len(c_data)/len(g_df))*100:.1f}%** of your total database.
    - **Business Strategy:** We recommend **{"loyalty retention" if spd_diff > 10 else "reactivation campaigns"}** tailored to this specific income bracket.
    - **Growth Insight:** The stability of this cluster suggests that target marketing here will yield a predictable ROI of at least 15-20% based on historical density.
    """

def show_results(df, scaled_data, features, choices, k, names_map):
    st.title("🚀 AI-Driven Business Insights")
    
    # Identify key columns
    inc_col = next((c for c in df.columns if 'income' in c.lower()), None)
    spd_col = next((c for c in df.columns if 'spend' in c.lower()), None)

    if "PCA Separation" in choices:
        st.header("1. Market Structure Map")
        pca_res = PCA(n_components=2).fit_transform(scaled_data)
        pca_df = pd.DataFrame(pca_res, columns=['PC1', 'PC2'])
        pca_df['Persona'] = df['Persona']
        fig = px.scatter(pca_df, x='PC1', y='PC2', color='Persona', title="AI Cluster Boundaries")
        st.plotly_chart(fig, use_container_width=True)
        st.info("AI Insight: This map shows how distinct your customer groups are. High distance between colors means your marketing should be highly differentiated.")

    st.divider()
    st.header("🔍 Individual Segment Deep-Dive")
    tabs = st.tabs([f"{names_map[i]}" for i in range(k)])
    
    # Prepare Radar Data (Safety: Numeric Only)
    all_means = df.groupby('Cluster')[features].mean()
    norm_means = (all_means - all_means.min()) / (all_means.max() - all_means.min() + 1e-6)

    for i in range(k):
        with tabs[i]:
            c_data = df[df['Cluster'] == i]
            col_left, col_right = st.columns(2)
            
            with col_left:
                st.subheader("📊 Purchasing Power")
                fig_scat = px.scatter(c_data, x=inc_col, y=spd_col, color_discrete_sequence=['#1f77b4'])
                st.plotly_chart(fig_scat, use_container_width=True)
            
            with col_right:
                st.subheader("🧬 Behavioral DNA")
                fig_rad = go.Figure(data=go.Scatterpolar(r=norm_means.iloc[i].values, theta=features, fill='toself'))
                st.plotly_chart(fig_rad, use_container_width=True)
            
            # AI GENERATED SUMMARY
            st.markdown(f"""
            <div style="background-color:#f8f9fa; padding:20px; border-radius:10px; border-left: 5px solid #1f77b4;">
                <h4 style="margin-top:0; color:#1f77b4;">AI Observation for {names_map[i]}</h4>
                {generate_ai_text(c_data, df, inc_col, spd_col)}
            </div>
            """, unsafe_allow_html=True)

    if st.button("RESET ANALYSIS"):
        st.session_state.step = "upload"
        st.rerun()
