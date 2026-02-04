import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder

def run_clustering(df, k):
    data = df.copy()
    # 1. Automatic Cleaning
    ids = [c for c in data.columns if 'id' in c.lower()]
    data = data.drop(columns=ids)
    # 2. Encoding
    for col in data.select_dtypes(include=['object']).columns:
        data[col] = LabelEncoder().fit_transform(data[col])
    # 3. Scaling & Clustering
    scaler = StandardScaler()
    scaled = scaler.fit_transform(data)
    model = KMeans(n_clusters=k, init='k-means++', random_state=42, n_init=10)
    df['Cluster'] = model.fit_predict(scaled)
    return df, scaled, data.columns.tolist()
