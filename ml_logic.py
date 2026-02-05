import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder

def auto_define_personas(df):
    summary = df.groupby('Cluster').mean(numeric_only=True)
    cols = summary.columns
    # Robustly find column names
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
    proc = data.drop(columns=[c for c in data.columns if 'id' in c.lower()], errors='ignore')
    
    for col in proc.select_dtypes(include=['object']).columns:
        proc[col] = LabelEncoder().fit_transform(proc[col].astype(str))
    
    scaled = StandardScaler().fit_transform(proc)
    data['Cluster'] = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(scaled)
    
    names_map = auto_define_personas(data)
    data['Persona'] = data['Cluster'].map(names_map)
    return data, scaled, proc.columns.tolist(), names_map
