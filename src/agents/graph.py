from typing import TypedDict, Dict, Any
from langgraph.graph import StateGraph, END
import requests
import pandas as pd
from src.xai.explainer import RiskExplainer

# 1. Define MerchantState FIRST so functions can reference it
class MerchantState(TypedDict):
    website_url: str
    industry_category: str
    scraped_data: Dict[str, Any]
    features: Dict[str, Any]
    risk_score: float
    shap_explanations: Dict[str, float]
    decision: str

# 2. Define worker nodes
def scrape_merchant_site(state: MerchantState) -> MerchantState:
    url = state["website_url"]
    try:
        resp = requests.get(url, timeout=5)
        text = resp.text.lower()
        has_refund = 1 if "refund" in text or "cancellation" in text else 0
        has_contact = 1 if "contact" in text or "email" in text else 0
        prohibited_words = sum(text.count(w) for w in ["gambling", "casino", "crypto", "replica"])
    except Exception:
        has_refund, has_contact, prohibited_words = 0, 0, 2

    state["scraped_data"] = {
        "has_refund_policy": has_refund,
        "has_contact_info": has_contact,
        "prohibited_keyword_count": prohibited_words
    }
    return state

def evaluate_risk(state: MerchantState) -> MerchantState:
    explainer = RiskExplainer()
    cat_risk = 0.8 if state["industry_category"] in ["Gaming", "Crypto"] else 0.2
    
    feat_dict = {
        'domain_age_days': 45,
        'has_refund_policy': state["scraped_data"]["has_refund_policy"],
        'has_contact_info': state["scraped_data"]["has_contact_info"],
        'prohibited_keyword_count': state["scraped_data"]["prohibited_keyword_count"],
        'industry_risk_score': cat_risk
    }
    
    df_feat = pd.DataFrame([feat_dict])
    prob, shap_exp = explainer.explain_merchant(df_feat)
    
    state["features"] = feat_dict
    state["risk_score"] = float(prob.item() if hasattr(prob, "item") else prob)
    
    state["shap_explanations"] = {
        k: float(v.item() if hasattr(v, "item") else v[0]) 
        for k, v in shap_exp.items()
    }
    
    if prob > 0.7:
        state["decision"] = "REJECTED"
    elif prob > 0.35:
        state["decision"] = "FLAGGED_FOR_HUMAN_REVIEW"
    else:
        state["decision"] = "APPROVED"
        
    return state

# 3. Define graph compiler
def create_risk_agent_graph():
    workflow = StateGraph(MerchantState)
    workflow.add_node("scraper", scrape_merchant_site)
    workflow.add_node("risk_evaluator", evaluate_risk)
    
    workflow.set_entry_point("scraper")
    workflow.add_edge("scraper", "risk_evaluator")
    workflow.add_edge("risk_evaluator", END)
    
    return workflow.compile()