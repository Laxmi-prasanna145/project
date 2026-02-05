import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder

def auto_define_personas(df):
    # Only use numeric columns for the summary comparison
    summary = df.select_dtypes(include=[np.number]).groupby('Cluster').mean()
    cols = summary.columns
    
    # Flexible column matching for Income and Spending
    inc_col = next((c for c in cols if 'income' in c.lower()), None)
    spd_col = next((c for c in cols if 'spend' in c.lower()), None)
    
    if not inc_col or not spd_col:
        return {i: f"Segment {i}" for i in range(len(summary))}

    m_inc, m_spd = summary[inc_col].mean(), summary[spd_col].mean()
    names = {}
    for i in range(len(summary)):
        inc, spd = summary.loc[i, inc_col], summary.loc[i, spd_col]
        if inc > m_inc and spd > m_spd: names[i] = "Platinum Champions"
        elif inc > m_inc: names[i] = "Reserved Wealth"
        elif spd > m_spd: names[i] = "High-Engagement Strivers"
        else: names[i] = "Budget Cautious"
    return names

def run_clustering(df, k):
    data = df.copy()
    # Remove ID-like columns for clustering
    proc = data.drop(columns=[c for c in data.columns if 'id' in c.lower()], errors='ignore')
    
    # Store categorical columns to encode them in the main data too
    cat_cols = proc.select_dtypes(include=['object']).columns
    le = LabelEncoder()
    for col in cat_cols:
        # Encode in the main data so mean() works later
        data[col] = le.fit_transform(data[col].astype(str))
        proc[col] = data[col] # Keep them synced
    
    # Scale and Cluster
    scaled = StandardScaler().fit_transform(proc)
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    data['Cluster'] = kmeans.fit_predict(scaled)
    
    # Map Names
    names_map = auto_define_personas(data)
    data['Persona'] = data['Cluster'].map(names_map)
    
    return data, scaled, proc.columns.tolist(), names_map
