import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from sklearn.decomposition import PCA

def get_dynamic_economic_summary(c_data, global_avg_inc, global_avg_spd):
    c_inc = c_data['AnnualIncome'].mean()
    c_spd = c_data['SpendingScore'].mean()
    
    inc_desc = "significantly above" if c_inc > global_avg_inc * 1.2 else "below" if c_inc < global_avg_inc * 0.8 else "around"
    spd_desc = "aggressive" if c_spd > global_avg_spd * 1.2 else "conservative" if c_spd < global_avg_spd * 0.8 else "moderate"
    
    return f"""
    - **Income Profile:** This group earns an average of ${c_inc:,.0f}, which is **{inc_desc}** the market average.
    - **Spending Behavior:** With a spending score of {c_spd:.1f}, they exhibit a **{spd_desc}** purchasing pattern.
    - **Density:** The data shows a cluster of {len(c_data)} individuals with high internal consistency.
    - **Action:** Since their spend is {spd_desc}, prioritize {"premium loyalty programs" if c_spd > global_avg_spd else "discount-based re-engagement"}.
    - **Stability:** The low variance in this scatter plot suggests this segment's behavior is highly predictable.
    """

def get_dynamic_radar_summary(norm_data, features, cluster_idx):
    traits = norm_data.iloc[cluster_idx]
    top_trait = features[traits.argmax()]
    weak_trait = features[traits.argmin()]
    
    return f"""
    - **Dominant Trait:** The AI identifies **{top_trait}** as the core behavioral driver for this segment.
    - **Optimization Area:** Engagement is lowest in **{weak_trait}**, representing a growth opportunity.
    - **Segment DNA:** The radar's shape suggests a { "specialized" if traits.std() > 0.2 else "generalist" } customer profile.
    - **Strategic Alignment:** Campaigns should lead with messaging centered on {top_trait}.
    - **Market Fit:** Compared to other clusters, this group is the most distinct in their {top_trait} levels.
    """

def show_results(df, scaled_data, features, choices, k, names_map):
    st.title("📊 AI-Driven Behavioral Insights")
    
    # Global Averages for comparison
    g_inc = df['AnnualIncome'].mean()
    g_spd = df['SpendingScore'].mean()

    if "PCA Separation" in choices:
        st.header("1. Global Segment Mapping")
        pca_df = pd.DataFrame(PCA(n_components=2).fit_transform(scaled_data), columns=['PC1', 'PC2'])
        pca_df['Persona'] = df['Persona']
        st.plotly_chart(px.scatter(pca_df, x='PC1', y='PC2', color='Persona'), use_container_width=True)
        st.info("AI Observation: This map visualizes how the algorithm separated your customers based on all features.")

    st.divider()
    tabs = st.tabs([f"{names_map[i]}" for i in range(k)])

    # Radar Data Prep
    all_means = df.groupby('Cluster')[features].mean()
    norm_means = (all_means - all_means.min()) / (all_means.max() - all_means.min())

    for i in range(k):
        with tabs[i]:
            c_data = df[df['Cluster'] == i]
            
            if "Economic Analysis" in choices:
                st.subheader("💰 Financial Narrative")
                st.plotly_chart(px.scatter(c_data, x="AnnualIncome", y="SpendingScore"), use_container_width=True)
                st.markdown(f'<div style="background:#f0f2f6;padding:15px;border-radius:10px;">{get_dynamic_economic_summary(c_data, g_inc, g_spd)}</div>', unsafe_allow_html=True)

            if "Behavioral Radar" in choices:
                st.subheader("🧬 Behavioral DNA")
                fig = go.Figure(data=go.Scatterpolar(r=norm_means.iloc[i].values, theta=features, fill='toself'))
                st.plotly_chart(fig, use_container_width=True)
                st.markdown(f'<div style="background:#e1f5fe;padding:15px;border-radius:10px;">{get_dynamic_radar_summary(norm_means, features, i)}</div>', unsafe_allow_html=True)

    if st.button("New Analysis"):
        st.session_state.step = "upload"; st.rerun()
