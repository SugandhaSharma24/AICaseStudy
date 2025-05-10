import streamlit as st
st.set_page_config(page_title="MultitoolApp", layout="wide")
import chatbot
import merged_code
import eligibility_app



tabs = st.tabs(["Eligibility App", "AI Chatbot"])

# Tab 1: Merged eligibility decision app
with tabs[0]:
    if hasattr(merged_code, 'run'):
        merged_code.run()
    else:
        st.error("Eligibility module (merged_code) not configured properly.")

# Tab 2: AI Chatbot
with tabs[1]:
    if hasattr(chatbot, 'run'):
        chatbot.run()
    else:
        st.error("Chatbot module not configured properly.")
# Tab 3: Knowledge Graph
with tabs[1]:
    if hasattr(eligibility_app, 'run'):
        eligibility_app.run()
    else:
        st.error("Knowledge graph module not configured properly.")        
       
