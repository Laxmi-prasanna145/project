import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder


# --------------------------------------------
# Auto Persona Definition
# --------------------------------------------
def auto_define_personas(df):

    # Select numeric columns only
    numeric_df = df.select_dtypes(include=np.number)

    # Remove Cluster column
    numeric_features = [col for col in numeric_df.columns if col != "Cluster"]

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


# --------------------------------------------
# Main Clustering Pipeline
# --------------------------------------------
def run_clustering(df, k):

    data = df.copy()

    # Remove ID columns safely
    ids = [c for c in data.columns if "id" in c.lower()]
    data = data.drop(columns=ids, errors="ignore")

    # Encode categorical columns
    for col in data.select_dtypes(include=["object"]).columns:
        encoder = LabelEncoder()
        data[col] = encoder.fit_transform(data[col].astype(str))

    # Select numeric columns
    numeric_cols = data.select_dtypes(include=np.number).columns.tolist()

    # Scaling
    scaler = StandardScaler()
    scaled = scaler.fit_transform(data[numeric_cols])

    # Apply KMeans
    model = KMeans(
        n_clusters=k,
        init="k-means++",
        random_state=42,
        n_init=10
    )

    df["Cluster"] = model.fit_predict(scaled)

    # Generate persona labels
    names_map = auto_define_personas(df)

    # Add Persona column
    df["Persona"] = df["Cluster"].map(names_map).fillna("Unknown Segment")

    return df, scaled, numeric_cols, names_map
