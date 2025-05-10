import streamlit as st
import pandas as pd
import os
from PyPDF2 import PdfReader
from PIL import Image
from openai import OpenAI
from langgraph.graph import StateGraph, END
from typing import TypedDict, Optional


# Load data
df = pd.read_csv("data.csv")

# Configure page
st.title("🌟 AI-Powered Social Welfare Assistant")

# ================= Model Context Protocol Implementation =================
class ModelContext(TypedDict):
    """Standardized protocol for model interactions"""
    model_name: str
    temperature: float
    max_tokens: Optional[int]
    system_prompt: str
    api_key: str

class ApplicationState(TypedDict):
    applicant_data: dict
    processed_files: dict
    errors: list
    decision: str
    model_context: ModelContext

def initialize_model_context() -> ModelContext:
    """Default model configuration"""
    return {
        "model_name": "gpt-4-turbo",
        "temperature": 0.0,
        "max_tokens": None,
        "system_prompt": """Analyze welfare applications using these strict rules:
        1. ID must be verified
        2. Income must be under $400
        3. At least 2 dependents""",
        "api_key": ""
    }

def get_client(model_context: ModelContext) -> OpenAI:
    """Client factory using model context"""
    return OpenAI(api_key=model_context["api_key_mcp"] or os.environ.get("OPENAI_API_KEY_MCP", ""))

# ================= Workflow Nodes =================
def process_files(state: ApplicationState) -> ApplicationState:
    """Process uploaded files with model context"""
    state = state.copy()
    model_context = state["model_context"]
    processed = {'pdf_text': '', 'id_verified': False, 'transcript': '', 'sentiment': 'Neutral'}
    
    try:
        # PDF Processing
        if pdf_file := state['processed_files'].get('pdf'):
            reader = PdfReader(pdf_file)
            processed['pdf_text'] = " ".join([page.extract_text() for page in reader.pages])[:500] + "..."
        
        # Image Verification
        if img_file := state['processed_files'].get('image'):
            Image.open(img_file).verify()
            processed['id_verified'] = True
            
        # Audio Processing
        if audio_file := state['processed_files'].get('audio'):
            client = get_client(model_context)
            audio_file.seek(0)
            transcript = client.audio.transcriptions.create(
                file=("audio.wav", audio_file.read(), "audio/wav"),
                model="whisper-1",
                temperature=model_context["temperature"],
                response_format="text"
            )
            processed['transcript'] = transcript[:500] + "..."
            processed['sentiment'] = 'Positive' if 'help' in transcript.lower() else 'Neutral'

    except Exception as e:
        state['errors'].append(str(e))
    
    state['processed_files'] = processed
    return state

def make_decision(state: ApplicationState) -> ApplicationState:
    """Make final decision using model context"""
    state = state.copy()
    model_context = state["model_context"]
    
    if state['errors']:
        state['decision'] = "Processing errors occurred"
        return state
    
    try:
        client = get_client(model_context)
        response = client.chat.completions.create(
            model=model_context["model_name"],
            temperature=model_context["temperature"],
            max_tokens=model_context["max_tokens"],
            messages=[{
                "role": "system",
                "content": model_context["system_prompt"]
            }, {
                "role": "user",
                "content": f"""
                Applicant: {state['applicant_data']['name']}
                Income: ${state['applicant_data']['income']}
                Dependents: {state['applicant_data']['dependents']}
                ID Verified: {state['processed_files']['id_verified']}
                Documents: {state['processed_files']['pdf_text'][:200]}
                Sentiment: {state['processed_files']['sentiment']}
                """
            }]
        )
        state['decision'] = response.choices[0].message.content
    except Exception as e:
        state['errors'].append(f"Decision Error: {str(e)}")
    
    return state

# ================= Workflow Setup =================
workflow = StateGraph(ApplicationState)
workflow.add_node("process_files", process_files)
workflow.add_node("make_decision", make_decision)
workflow.set_entry_point("process_files")
workflow.add_edge("process_files", "make_decision")
workflow.add_edge("make_decision", END)
app = workflow.compile()

# ================= Streamlit UI =================
with st.sidebar:
    st.header("⚙️ Model Configuration")
    model_config = initialize_model_context()
    model_config["model_name"] = st.selectbox(
        "Model",
        ["gpt-4-turbo", "gpt-3.5-turbo"],
        index=0
    )
    model_config["temperature"] = st.slider(
        "Temperature",
        0.0, 1.0, 0.0
    )
    model_config["system_prompt"] = st.text_area(
        "System Prompt",
        value=model_config["system_prompt"],
        height=200
    )
    model_config["api_key_mcp"] = st.text_input(
        "🔑 OpenAI API Key Mcp",
        type="password"
    )
    
    st.header("📥 Application Inputs")
    selected_applicant = st.selectbox("👤 Select Applicant", df['name'].unique())
    pdf_file = st.file_uploader("📑 Supporting Document (PDF)", type=["pdf"])
    img_file = st.file_uploader("🖼️ ID Verification", type=["jpg", "jpeg", "png"])
    audio_file = st.file_uploader("🔊 Voice Explanation", type=["mp3", "wav"])
    analyze_btn = st.button("🚀 Analyze Application")

if analyze_btn:
    # Initialize state with model context
    initial_state = {
        "model_context": model_config,
        "applicant_data": df[df['name'] == selected_applicant].iloc[0].to_dict(),
        "processed_files": {
            "pdf": pdf_file,
            "image": img_file,
            "audio": audio_file
        },
        "errors": [],
        "decision": ""
    }
    
    # Execute workflow
    result = app.invoke(initial_state)
    
    # Display Results
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("👤 Applicant Profile")
        applicant = result['applicant_data']
        st.markdown(f"""
        **Name:** {applicant['name']}  
        **Income:** ${applicant['income']}  
        **Dependents:** {applicant['dependents']}
        """)
        
        if img_file:
            st.image(Image.open(img_file), width=250)
            status = "✅ Verified" if result['processed_files']['id_verified'] else "❌ Unverified"
            st.markdown(f"**ID Status:** {status}")

    with col2:
        if pdf_file:
            st.subheader("📄 Document Excerpt")
            st.write(result['processed_files']['pdf_text'])
        
        if audio_file:
            st.subheader("🗣️ Voice Analysis")
            st.write(result['processed_files']['transcript'])
            st.markdown(f"**Sentiment:** {result['processed_files']['sentiment']}")

    st.markdown("---")
    st.subheader("📝 Final Decision")
    st.markdown(result['decision'])
    
    st.markdown("---")
    st.subheader("⚙️ Model Context Used")
    st.json({
        "model": result['model_context']['model_name'],
        "temperature": result['model_context']['temperature'],
        "system_prompt": result['model_context']['system_prompt'][:100] + "..."
    })

else:
    st.info("💡 Select an applicant and upload documents to begin analysis")