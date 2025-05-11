import streamlit as st
import pandas as pd
import os
from PyPDF2 import PdfReader
from PIL import Image
from openai import OpenAI
from crewai import Agent, Task, Crew  # For Crew AI
from langgraph.graph import StateGraph, END  # For LangGraph
from typing import TypedDict

def run():
    # Load data
    df = pd.read_csv("data.csv")

    # Configure page
    st.title("🌟 AI-Powered Social Welfare Assistant")

    # Common document processing function
    def process_documents(pdf_file, img_file, audio_file):
        """Process uploaded files and extract relevant information"""
        processed = {
            'pdf_text': '',
            'id_verified': False,
            'transcript': '',
            'sentiment': 'Neutral'
        }

        if pdf_file:
            try:
                reader = PdfReader(pdf_file)
                processed['pdf_text'] = " ".join([page.extract_text() for page in reader.pages])[:500] + "..."
            except Exception as e:
                st.error(f"PDF Error: {str(e)}")

        if img_file:
            try:
                Image.open(img_file).verify()
                processed['id_verified'] = True
            except Exception as e:
                processed['id_verified'] = False
                st.error(f"Image Error: {str(e)}")

        if audio_file:
            try:
                client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
                audio_file.seek(0)
                transcript = client.audio.transcriptions.create(
                    file=("audio.wav", audio_file.read(), "audio/wav"),
                    model="whisper-1",
                    response_format="text"
                )
                processed['transcript'] = transcript[:500] + "..."
                processed['sentiment'] = 'Positive' if 'help' in transcript.lower() else 'Neutral'
            except Exception as e:
                st.error(f"Audio Error: {str(e)}")

        return processed

    # Crew AI decision-making
    def crew_ai_decision(applicant_data, processed_data):
        """Create decision task for Crew AI framework"""
        decision_agent = Agent(
            role="Strict Eligibility Officer",
            goal="Make decisions based ONLY on provided data",
            backstory="Expert in welfare policy adherence",
            verbose=True,
            allow_delegation=False,
            max_iter=3,
            temperature=0.1
        )

        task = Task(
            description=f"""
            Analyze ONLY this data:
            - Name: {applicant_data['name']}
            - Income: ${applicant_data['income']}
            - Dependents: {applicant_data['dependents']}
            - ID Verified: {processed_data['id_verified']}
            - Document Content: {processed_data['pdf_text'][:200]}
            - Voice Sentiment: {processed_data['sentiment']}
            
            Apply EXACTLY these rules:
            1. ID must be verified
            2. Income must be under $400
            3. At least 2 dependents
            """,
            agent=decision_agent,
            expected_output="""
            Structured decision in this format:
            ## Eligibility Result
            **Status:** Approved/Rejected  
            **Reason:** [concise reason based on rules]
            
            ## Verification Summary
            - ID Verified: Yes/No
            - Income: $[amount] [meets/doesn't meet] requirement
            - Dependents: [number] [meets/doesn't meet] requirement
            """,
            context=[],
            config={"temperature": 0.0}
        )

        crew = Crew(agents=[decision_agent], tasks=[task])
        return crew.kickoff()

    # LangGraph decision-making
    def langgraph_decision(applicant_data, processed_data):
        """Make eligibility decision using LangGraph"""
        class ApplicationState(TypedDict):
            applicant_data: dict
            processed_files: dict
            errors: list
            decision: str

        def make_decision(state: ApplicationState) -> ApplicationState:
            """Make final eligibility decision"""
            state = state.copy()
            if state['errors']:
                state['decision'] = "Errors occurred during processing:\n- " + "\n- ".join(state['errors'])
                return state

            data = {
                **state['applicant_data'],
                **state['processed_files']
            }

            client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
            response = client.chat.completions.create(
                model="gpt-4",
                temperature=0.0,
                messages=[{
                    "role": "system",
                    "content": f"""
                    Analyze ONLY this data:
                    - Name: {data['name']}
                    - Income: ${data['income']}
                    - Dependents: {data['dependents']}
                    - ID Verified: {data['id_verified']}
                    - Document Content: {data['pdf_text'][:200]}
                    - Voice Sentiment: {data['sentiment']}

                    Apply EXACTLY these rules:
                    1. ID must be verified
                    2. Income must be under $400
                    3. At least 2 dependents
                    
                    Structured response format:
                    ## Eligibility Result
                    **Status:** Approved/Rejected  
                    **Reason:** [concise reason]

                    ## Verification Summary
                    - ID Verified: Yes/No
                    - Income: $[amount] [meets/doesn't meet]
                    - Dependents: [number] [meets/doesn't meet]
                    """
                }]
            )

            state['decision'] = response.choices[0].message.content
            return state

        workflow = StateGraph(ApplicationState)
        workflow.add_node("process_files", lambda state: state)  # Placeholder function
        workflow.add_node("make_decision", make_decision)
        workflow.set_entry_point("process_files")
        workflow.add_edge("process_files", "make_decision")
        workflow.add_edge("make_decision", END)
        app = workflow.compile()

        initial_state = {
            "applicant_data": applicant_data,
            "processed_files": processed_data,
            "errors": [],
            "decision": ""
        }

        result = app.invoke(initial_state)
        return result['decision']

    # Streamlit UI
    with st.sidebar:
        st.header("📥 Application Inputs")
        openai_key = st.text_input("🔑 OpenAI API Key", type="password",key="api_key_input")
        selected_applicant = st.selectbox("👤 Select Applicant", df['name'].unique(), key="eligibility_applicant_selectbox")
        pdf_file = st.file_uploader("📑 Supporting Document (PDF)", type=["pdf"], key="eligibility_pdf_uploader")
        img_file = st.file_uploader("🖼️ ID Verification", type=["jpg", "jpeg", "png"], key="eligibility_image_uploader" )
        audio_file = st.file_uploader("🔊 Voice Explanation", type=["mp3", "wav"], key="eligibility_audio_uploader" )
        framework_choice = st.selectbox("Select Framework", ["Crew AI", "LangGraph"], key="eligibility_framework_selectbox")
        analyze_btn = st.button("🚀 Analyze Application", key="eligibility_analyze_button")

    if analyze_btn:
        if not openai_key:
            st.error("🔑 API key required")
            st.stop()

        os.environ["OPENAI_API_KEY"] = openai_key
        applicant = df[df['name'] == selected_applicant].iloc[0].to_dict()
        processed_data = process_documents(pdf_file, img_file, audio_file)

        if framework_choice == "Crew AI":
            decision_result = crew_ai_decision(applicant, processed_data)
        elif framework_choice == "LangGraph":
            decision_result = langgraph_decision(applicant, processed_data)

        # Display Results
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("👤 Applicant Profile")
            st.markdown(f"""
            **Name:** {applicant['name']}  
            **Income:** ${applicant['income']}  
            **Dependents:** {applicant['dependents']}
            """)
            
            if img_file:
                st.image(Image.open(img_file), width=250)
                status = "✅ Verified" if processed_data['id_verified'] else "❌ Unverified"
                st.markdown(f"**ID Status:** {status}")

        with col2:
            if pdf_file:
                st.subheader("📄 Document Excerpt")
                st.write(processed_data['pdf_text'])
            
            if audio_file:
                st.subheader("🗣️ Voice Analysis")
                st.write(processed_data['transcript'])
                st.markdown(f"**Sentiment:** {processed_data['sentiment']}")

        st.markdown("---")
        st.subheader("📝 Final Decision")
        st.markdown(decision_result)

    else:
        st.info("💡 Select an applicant and upload documents to begin analysis")
