import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder

def auto_define_personas(df):
    summary = df.groupby('Cluster').mean(numeric_only=True)
    m_inc, m_spd = summary['AnnualIncome'].mean(), summary['SpendingScore'].mean()
    names = {}
    for i in range(len(summary)):
        inc, spd = summary.loc[i, 'AnnualIncome'], summary.loc[i, 'SpendingScore']
        if inc > m_inc and spd > m_spd: names[i] = "Platinum Champions"
        elif inc > m_inc: names[i] = "Reserved Wealth"
        elif spd > m_spd: names[i] = "High-Engagement Strivers"
        else: names[i] = "Budget Cautious"
    return names

def run_clustering(df, k):
    data = df.copy()
    proc = data.drop(columns=[c for c in data.columns if 'id' in c.lower()])
    for col in proc.select_dtypes(include=['object']).columns:
        proc[col] = LabelEncoder().fit_transform(proc[col])
    
    scaled = StandardScaler().fit_transform(proc)
    data['Cluster'] = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(scaled)
    names_map = auto_define_personas(data)
    data['Persona'] = data['Cluster'].map(names_map)
    return data, scaled, proc.columns.tolist(), names_map
