import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
import numpy as np


# ------------------------------
# Auto Persona Definition
# ------------------------------
def auto_define_personas(df):

    # Select only numeric columns
    numeric_df = df.select_dtypes(include=np.number)

    # Remove Cluster column from averaging
    numeric_features = [c for c in numeric_df.columns if c != "Cluster"]

    summary = df.groupby("Cluster")[numeric_features].mean()

    names_map = {}

    for cluster in summary.index:

        row = summary.loc[cluster]

        # Simple scoring logic
        score = row.mean()

        if score >= summary.values.mean():
            names_map[cluster] = "High Value Customers"
        elif score >= summary.values.mean() * 0.75:
            names_map[cluster] = "Growth Potential Customers"
        elif score >= summary.values.mean() * 0.50:
            names_map[cluster] = "Moderate Customers"
        else:
            names_map[cluster] = "Low Engagement Customers"

    return names_map


# ------------------------------
# Main Clustering Function
# ------------------------------
def run_clustering(df, k):

    data = df.copy()

    # Remove ID columns
    ids = [c for c in data.columns if "id" in c.lower()]
    data = data.drop(columns=ids)

    # Encode categorical columns
    for col in data.select_dtypes(include=["object"]).columns:
        data[col] = LabelEncoder().fit_transform(data[col].astype(str))

    # Select numeric only for ML
    numeric_cols = data.select_dtypes(include=np.number).columns.tolist()

    # Scaling
    scaler = StandardScaler()
    scaled = scaler.fit_transform(data[numeric_cols])

    # KMeans
    model = KMeans(
        n_clusters=k,
        init="k-means++",
        random_state=42,
        n_init=10
    )

    df["Cluster"] = model.fit_predict(scaled)

    # Generate persona labels
    names_map = auto_define_personas(df)

    return df, scaled, numeric_cols, names_map
