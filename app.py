import streamlit as st
import chatbot
import merged_code
import eligibility_app
import Mcp_agent
import sys
import pysqlite3
sys.modules["sqlite3"] = pysqlite3
st.set_page_config(page_title="MultitoolApp", layout="wide")

# Sidebar navigation instead of tabs
st.sidebar.title("🧭 Navigation")
page = st.sidebar.radio("Go to:", ["Eligibility App", "AI Chatbot", "Knowledge Graph", "MCP"])

# Page routing
if page == "Eligibility App":
    if hasattr(merged_code, 'run'):
        merged_code.run()
    else:
        st.error("Eligibility module (merged_code) not configured properly.")

elif page == "AI Chatbot":
    if hasattr(chatbot, 'run'):
        chatbot.run()
    else:
        st.error("Chatbot module not configured properly.")

elif page == "Knowledge Graph":
    if hasattr(eligibility_app, 'run'):
        eligibility_app.run()
    else:
        st.error("Knowledge graph module not configured properly.")

elif page == "MCP":
    if hasattr(Mcp_agent, 'run'):
        Mcp_agent.run()
    else:
        st.error("MCP agent module not configured properly.")
