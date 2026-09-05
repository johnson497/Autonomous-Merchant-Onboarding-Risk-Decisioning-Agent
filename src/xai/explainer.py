import shap
import joblib
import pandas as pd

class RiskExplainer:
    def __init__(self, model_path="models/risk_model.joblib"):
        self.model = joblib.load(model_path)
        self.explainer = shap.TreeExplainer(self.model)

    def explain_merchant(self, merchant_feature_df):
        shap_values = self.explainer.shap_values(merchant_feature_df)
        prob = self.model.predict_proba(merchant_feature_df)[0][1]
        
        if isinstance(shap_values, list):
            sv = shap_values[1][0]
        else:
            sv = shap_values[0]
            
        feature_importance = dict(zip(merchant_feature_df.columns, sv))
        return prob, feature_importance