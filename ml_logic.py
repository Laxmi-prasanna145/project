import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder


# ---------------------------------------------------
# Auto Persona Definition Based on Cluster Behaviour
# ---------------------------------------------------
def auto_define_personas(df):

    # Select numeric columns only
    numeric_df = df.select_dtypes(include=np.number)

    # Remove cluster column
    numeric_features = [c for c in numeric_df.columns if c != "Cluster"]

    # Cluster summary
    summary = df.groupby("Cluster")[numeric_features].mean()

    overall_avg = summary.values.mean()

    names_map = {}

    for cluster in summary.index:

        cluster_avg = summary.loc[cluster].mean()

        if cluster_avg >= overall_avg:
            names_map[cluster] = "High Value Customers"

        elif cluster_avg >= overall_avg * 0.75:
            names_map[cluster] = "Growth Potential Customers"

        elif cluster_avg >= overall_avg * 0.50:
            names_map[cluster] = "Moderate Customers"

        else:
            names_map[cluster] = "Low Engagement Customers"

    return names_map


# ---------------------------------------------------
# Main Clustering Pipeline
# ---------------------------------------------------
def run_clustering(df, k):

    data = df.copy()

    # 🔹 Remove ID columns
    ids = [c for c in data.columns if "id" in c.lower()]
    data = data.drop(columns=ids, errors="ignore")

    # 🔹 Encode categorical columns
    for col in data.select
