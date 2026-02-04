import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from sklearn.decomposition import PCA


def pca_summary(df):
    return """
    The PCA visualization shows how customers are grouped based on multi-dimensional behaviour.
    Clear separation between clusters indicates strong segmentation quality.
    Overlapping regions suggest customers share similar behavioural traits.
    Dense cluster formations represent dominant customer types.
    This helps businesses visually understand customer diversity and targeting opportunities.
    """


def show_results(df, scaled_data, features, choices, k, names_map):

    st.title("📊 AI-Driven Behavioral Insights")

    if "PCA Separation" in choices:

        st.header("Global Segment Mapping")

        pca_df = pd.DataFrame(
            PCA(n_components=2).fit_transform(scaled_data),
            columns=['PC1', 'PC2']
        )

        pca_df['Persona'] = df['Persona']

        fig = px.scatter(
            pca_df,
            x='PC1',
            y='PC2',
            color='Persona'
        )

        st.plotly_chart(fig, use_container_width=True, key="pca_plot")

        st.info(pca_summary(df))

    st.divider()

    tabs = st.tabs([f"{names_map[i]}" for i in range(k)])

    all_means = df.groupby('Cluster')[features].mean()

    norm_means = (all_means - all_means.min()) / (
        (all_means.max() - all_means.min()).replace(0, 1)
    )

    for i in range(k):

        with tabs[i]:

            c_data = df[df['Cluster'] == i]

            if "Correlation Heatmap" in choices:

                st.subheader("Feature Correlation")

                corr = c_data[features].corr()

                fig = px.imshow(corr, text_auto=True)

                st.plotly_chart(fig, use_container_width=True)

                st.info(
                    "Heatmap highlights relationships between customer features. "
                    "Strong correlations suggest linked purchasing behaviours."
                )

            if "Behavioral Radar" in choices:

                st.subheader("Behavioral DNA")

                fig = go.Figure(
                    data=go.Scatterpolar(
                        r=norm_means.loc[i].values,
                        theta=features,
                        fill='toself'
                    )
                )

                st.plotly_chart(fig, use_container_width=True)

                st.info(
                    "Radar chart shows behavioural strengths and weaknesses "
                    "of this customer segment across all features."
                )

    if st.button("START NEW ANALYSIS", key="reset_app"):
        st.session_state.step = "upload"
        st.rerun()
