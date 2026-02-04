import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder


def auto_define_personas(df, features):

    summary = df.groupby('Cluster')[features].mean()
    global_mean = summary.mean()

    names = {}

    for i in summary.index:

        high_features = summary.loc[i] > global_mean

        if high_features.sum() >= len(features) * 0.6:
            names[i] = "High Value Customers"

        elif high_features.sum() <= len(features) * 0.3:
            names[i] = "Low Engagement Customers"

        else:
            names[i] = "Potential Growth Segment"

    return names


def run_clustering(df, k):

    data = df.copy()

    proc = data.drop(
        columns=[c for c in data.columns if 'id' in c.lower()],
        errors="ignore"
    )

    for col in proc.select_dtypes(include=['object']).columns:
        proc[col] = LabelEncoder().fit_transform(proc[col])

    features = proc.select_dtypes(include=['number']).columns.tolist()

    scaler = StandardScaler()
    scaled = scaler.fit_transform(proc[features])

    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)

    data['Cluster'] = kmeans.fit_predict(scaled)

    names_map = auto_define_personas(data, features)

    data['Persona'] = data['Cluster'].map(names_map)

    return data, scaled, features, names_map
