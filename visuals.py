import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

def show_results(df, scaled_data, features, choices, k):
    st.title("📊 Strategic Segmentation Dashboard")
    
    if "Market Share Pie" in choices:
        st.header("1. Market Share Distribution")
        st.plotly_chart(px.pie(df, names='Cluster', hole=0.4, title="Customer Weightage"), use_container_width=True)
        st.info("Global Summary: This identifies core vs niche segments.")

    st.divider()
    st.header("📍 Per-Cluster Deep-Dive")
    tabs = st.tabs([f"Segment {i}" for i in range(k)])
    
    for i in range(k):
        with tabs[i]:
            c_data = df[df['Cluster'] == i]
            if "Income vs Spend Scatter" in choices:
                st.plotly_chart(px.scatter(c_data, x="AnnualIncome", y="SpendingScore", title=f"Segment {i} Economy"), use_container_width=True)
            if "Age Distribution" in choices:
                st.plotly_chart(px.histogram(c_data, x="Age", title=f"Segment {i} Demographics"), use_container_width=True)

            st.markdown(f'''
                <div style="background-color: #e3f2fd; padding: 15px; border-radius: 10px; border-left: 5px solid #2196f3;">
                <b>🤖 AI Insight for Cluster {i}:</b> This segment contains {len(c_data)} customers. 
                Average Spending Score is {c_data['SpendingScore'].mean():.1f}. 
                Strategy: Focus on {"Premium Loyalty" if c_data['SpendingScore'].mean() > 50 else "Discount Incentives"}.
                </div>
            ''', unsafe_allow_html=True)
