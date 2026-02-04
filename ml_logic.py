import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder

def auto_define_personas(df):
    summary = df.groupby('Cluster').mean(numeric_only=True)
    names = {}
    m_income = summary['AnnualIncome'].mean()
    m_spend = summary['SpendingScore'].mean()
    
    for i in range(len(summary)):
        inc = summary.loc[i, 'AnnualIncome']
        spd = summary.loc[i, 'SpendingScore']
        if inc > m_income and spd > m_spend: names[i] = "Platinum Champions"
        elif inc > m_income and spd <= m_spend: names[i] = "Reserved Wealth"
        elif inc <= m_income and spd > m_spend: names[i] = "High-Engagement Strivers"
        elif inc < m_income and spd < m_spend: names[i] = "Budget Cautious"
        else: names[i] = "Core Market"
    return names

def run_clustering(df, k):
    data = df.copy()
    # Clean IDs
    ids = [c for c in data.columns if 'id' in c.lower()]
    data_proc = data.drop(columns=ids)
    
    # Encode
    for col in data_proc.select_dtypes(include=['object']).columns:
        data_proc[col] = LabelEncoder().fit_transform(data_proc[col])
    
    # Scale & KMeans
    scaler = StandardScaler()
    scaled = scaler.fit_transform(data_proc)
    model = KMeans(n_clusters=k, init='k-means++', random_state=42, n_init=10)
    data['Cluster'] = model.fit_predict(scaled)
    
    # Map Personas
    names_map = auto_define_personas(data)
    data['Persona'] = data['Cluster'].map(names_map)
    
    return data, scaled, data_proc.columns.tolist(), names_map
