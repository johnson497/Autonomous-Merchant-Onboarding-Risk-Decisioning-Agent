import streamlit as st
import pandas as pd
from src.agents.graph import create_risk_agent_graph

st.set_page_config(page_title="Razorpay AI Risk Manager", layout="wide")
st.title("🛡️ Autonomous Merchant Onboarding & Risk Agent")

col1, col2 = st.columns([1, 2])

with col1:
    st.header("Merchant Application")
    url = st.text_input("Merchant Website URL", "https://example.com")
    category = st.selectbox("Category", ["E-Commerce", "SaaS", "Gaming", "Crypto", "Services"])
    submit = st.button("Evaluate Merchant Risk")

if submit:
    agent = create_risk_agent_graph()
    initial_state = {
        "website_url": url,
        "industry_category": category,
        "scraped_data": {},
        "features": {},
        "risk_score": 0.0,
        "shap_explanations": {},
        "decision": ""
    }
    
    with st.spinner("Agent running web scraping and SHAP evaluation..."):
        result = agent.invoke(initial_state)

    with col2:
        st.header("Risk Assessment Results")
        dec = result["decision"]
        if dec == "APPROVED":
            st.success(f"Status: {dec}")
        elif dec == "FLAGGED_FOR_HUMAN_REVIEW":
            st.warning(f"Status: {dec}")
        else:
            st.error(f"Status: {dec}")
            
        st.metric("Predicted Risk Probability", f"{result['risk_score']*100:.1f}%")
        
        st.subheader("Explainable AI (SHAP) Factor Attribution")
        shap_df = pd.DataFrame(
            list(result["shap_explanations"].items()), 
            columns=["Feature", "SHAP Impact"]
        )
        st.bar_chart(shap_df.set_index("Feature"))