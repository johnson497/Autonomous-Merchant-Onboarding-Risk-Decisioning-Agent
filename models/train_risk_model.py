import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

def generate_synthetic_data():
    np.random.seed(42)
    n = 1000
    domain_age_days = np.random.randint(1, 1000, n)
    has_refund_policy = np.random.choice([0, 1], n, p=[0.3, 0.7])
    has_contact_info = np.random.choice([0, 1], n, p=[0.2, 0.8])
    prohibited_keyword_count = np.random.poisson(0.5, n)
    industry_risk_score = np.random.uniform(0, 1, n)
    
    risk_score = (
        (domain_age_days < 30) * 0.3 + 
        (1 - has_refund_policy) * 0.25 + 
        prohibited_keyword_count * 0.2 + 
        industry_risk_score * 0.25
    )
    is_high_risk = (risk_score > 0.45).astype(int)
    
    return pd.DataFrame({
        'domain_age_days': domain_age_days,
        'has_refund_policy': has_refund_policy,
        'has_contact_info': has_contact_info,
        'prohibited_keyword_count': prohibited_keyword_count,
        'industry_risk_score': industry_risk_score,
        'is_high_risk': is_high_risk
    })

if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    df = generate_synthetic_data()
    X = df.drop(columns=['is_high_risk'])
    y = df['is_high_risk']
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    joblib.dump(model, "models/risk_model.joblib")
    print("Risk model trained successfully and saved to models/risk_model.joblib")