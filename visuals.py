import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from sklearn.decomposition import PCA

def get_dynamic_economic_summary(c_data, global_avg_inc, global_avg_spd):
    c_inc, c_spd = c_data['AnnualIncome'].mean(), c_data['SpendingScore'].mean()
    inc_desc = "significantly above" if c_inc > global_avg_inc * 1.2 else "below" if c_inc < global_avg_inc * 0.8 else "around"
    spd_desc = "aggressive" if c_spd > global_avg_spd * 1.2 else "conservative" if c_spd < global_avg_spd * 0.8 else "moderate"
    return f"""
    - **Income Profile:** This group earns ${c_inc:,.0f} avg, which is **{inc_desc}** the market average.
    - **Spending Behavior:** With a score of {c_spd:.1f}, they exhibit a **{spd_desc}** pattern.
    - **Density:** Represents {len(c_data)} distinct behavioral profiles.
    - **Action:** Prioritize {"premium loyalty" if c_spd > global_avg_spd else "re-engagement incentives"}.
    - **Observation:** High cluster density indicates extremely predictable future purchasing.
    """

def get_dynamic_radar_summary(norm_data, features, cluster_idx):
    traits = norm_data.iloc[cluster_idx]
    top_trait, weak_trait = features[traits.argmax()], features[traits.argmin()]
    return f"""
    - **Dominant Trait:** AI identifies **{top_trait}** as the core driver for this segment.
    - **Opportunity:** Engagement is lowest in **{weak_trait}**, suggesting a market gap.
    - **Segment DNA:** The profile indicates a { "highly specialized" if traits.std() > 0.2 else "balanced" } behavioral set.
    - **Alignment:** Focus messaging exclusively on {top_trait} to maximize ROI.
    - **Fit:** This segment shows the highest distinctiveness in {top_trait} compared to the total database.
    """

def show_results(df, scaled_data, features, choices, k, names_map):
    st.title("📊 AI-Driven Behavioral Insights")
    g_inc, g_spd = df['AnnualIncome'].mean(), df['SpendingScore'].mean()

    if "PCA Separation" in choices:
        st.header("1. Global Segment Mapping")
        pca_df = pd.DataFrame(PCA(n_components=2).fit_transform(scaled_data), columns=['PC1', 'PC2'])
        pca_df['Persona'] = df['Persona']
        st.plotly_chart(px.scatter(pca_df, x='PC1', y='PC2', color='Persona', key="pca_chart"), use_container_width=True)

    st.divider()
    tabs = st.tabs([f"{names_map[i]}" for i in range(k)])
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

    if st.button("START NEW ANALYSIS", key="reset_app"):
        st.session_state.step = "upload"
        st.rerun()
