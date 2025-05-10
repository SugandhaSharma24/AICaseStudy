import streamlit as st
import pandas as pd
from py2neo import Graph
import pytesseract
import pdfplumber
import joblib
import speech_recognition as sr
from pydub import AudioSegment
import re
def run():
    # --- Neo4j Connection ---
    graph = Graph("bolt://localhost:7687", auth=("neo4j", "test1234"))

    # --- Load and Clean Data ---
    def load_and_clean_data():
        user_df = pd.read_excel("data/user.xlsx").drop_duplicates()
        bank_df = pd.read_excel("data/bank.xlsx").drop_duplicates()
        credit_df = pd.read_excel("data/credit_score.xlsx").drop_duplicates()

        if 'Balance' in bank_df.columns:
            bank_df['Balance'] = bank_df['Balance'].replace('[\$,]', '', regex=True).astype(float)

        return user_df, bank_df, credit_df

    # --- Build Knowledge Graph ---
    def build_knowledge_graph(graph, user_df, bank_df, credit_df):
        tx = graph.begin()

        for _, row in user_df.iterrows():
            tx.run("""
                MERGE (p:Person {id: $id})
                SET p.name = $name, p.address = $address,
                    p.employment = $employment, p.family_members = $family_members
            """, id=row['ID'], name=row['Name'], address=row['Address'],
                employment=row['Employment'], family_members=int(row['Family Members']))

        for _, row in bank_df.iterrows():
            tx.run("""
                MATCH (p:Person {id: $id})
                MERGE (b:BankAccount {account_number: $account_number})
                SET b.bank_name = $bank_name, b.balance = $balance
                MERGE (p)-[:HAS_ACCOUNT]->(b)
            """, id=row['ID'], account_number=row['Account Number'],
                bank_name=row['Bank Name'], balance=float(row['Balance']))

        for _, row in credit_df.iterrows():
            tx.run("""
                MATCH (p:Person {id: $id})
                MERGE (c:CreditScore {id: $id})
                SET c.score = $score
                MERGE (p)-[:HAS_CREDIT_SCORE]->(c)
            """, id=row['ID'], score=int(row['Credit Score']))

        tx.commit()

    # --- Text Extraction Functions ---
    def extract_text_from_pdf(pdf_file):
        with pdfplumber.open(pdf_file) as pdf:
            text = ""
            for page in pdf.pages:
                text += page.extract_text()
        return text

    def extract_text_from_image(image_file):
        image = Image.open(image_file)
        text = pytesseract.image_to_string(image)
        return text

    def extract_text_from_audio(audio_file):
        audio = AudioSegment.from_file(audio_file)
        audio.export("temp.wav", format="wav")
        
        recognizer = sr.Recognizer()
        with sr.AudioFile("temp.wav") as source:
            audio_data = recognizer.record(source)
        text = recognizer.recognize_google(audio_data)
        return text

    # --- Extract ID Function ---
    def extract_id_from_text(text):
        id_match = re.search(r'\bID(?: No)?:\s*([\d\-]+)', text, re.IGNORECASE)
        return id_match.group(1).strip() if id_match else None

    # --- Function to match data by ID in Neo4j ---
    def match_by_id(id_no):
        query = """
        MATCH (p:Person {id: $id_no})-[:HAS_ACCOUNT]->(b:BankAccount)
        RETURN p.name AS name, b.balance AS balance, p.employment AS employment, p.family_members AS family_members
        """
        result = graph.run(query, id_no=id_no).data()

        if result:
            return result[0]
        return None
    # Load the trained machine learning model
    model = joblib.load('eligibility_model.pkl')  # Replace with the correct path if necessary

    # Use this model for prediction in your eligibility function
    def predict_eligibility(balance, family_members, employment):
        # Convert employment status to numerical if needed (assuming it's encoded)
        employment_numeric = 1 if employment.lower() == 'employed' else 0
        
        # Prepare features for prediction
        features = [[balance, family_members, employment_numeric]]
        
        # Predict eligibility using the loaded model
        prediction = model.predict(features)  # Model returns [0 or 1]
        
        # Return True if eligible (prediction is 1), otherwise False
        return prediction[0] == 1


    # --- Streamlit App ---
    st.set_page_config(page_title="Social Welfare Knowledge Graph", layout="wide")
    st.title("📊 Social Welfare Knowledge Graph")

    st.markdown("---")
    st.subheader("1. Load and Display Merged Data")

    user_df, bank_df, credit_df = load_and_clean_data()

    # Merge DataFrames
    merged_df = user_df.merge(bank_df, on="ID", how="left")
    merged_df = merged_df.merge(credit_df, on="ID", how="left")

    st.dataframe(merged_df, use_container_width=True)

    # Button to build Knowledge Graph
    if st.button("📌 Build Knowledge Graph"):
        try:
            build_knowledge_graph(graph, user_df, bank_df, credit_df)
            st.success("✅ Knowledge Graph successfully built in Neo4j")
        except Exception as e:
            st.error(f"❌ An error occurred: {e}")

    st.markdown("---")
    st.subheader("2. Upload Document for Analysis")

    # File uploader for PDF, Image, and Audio files
    uploaded_file = st.file_uploader("Upload a PDF, Image, or Audio file", type=["pdf", "jpg", "jpeg", "png", "mp3", "wav"])

    if uploaded_file is not None:
        if uploaded_file.type == "application/pdf":
            extracted_text = extract_text_from_pdf(uploaded_file)
        elif uploaded_file.type in ["image/jpg", "image/jpeg", "image/png"]:
            extracted_text = extract_text_from_image(uploaded_file)
        elif uploaded_file.type in ["audio/mp3", "audio/wav"]:
            extracted_text = extract_text_from_audio(uploaded_file)

        st.write("📄 Extracted Text:")
        st.code(extracted_text)

        # Extract ID from text
        extracted_id = extract_id_from_text(extracted_text)
        st.write(f"🔎 Extracted ID: {extracted_id}")

        if extracted_id:
            # Match ID in Neo4j graph
            matched_data = match_by_id(extracted_id)
            if matched_data:
                st.write(f"Balance: {matched_data['balance']}")
                st.write(f"Employment: {matched_data['employment']}")
                st.write(f"Family Members: {matched_data['family_members']}")

                # Analyze eligibility (optional)
                if st.button("Analyze"):
                    balance = matched_data['balance']
                    employment = matched_data['employment']
                    family_members = matched_data['family_members']
                    eligibility = predict_eligibility(balance, family_members, employment)
                    if eligibility:
                        st.success("✅ The applicant is eligible!")
                        st.markdown("**Recommended Support Package:**")
                        st.markdown("- Direct cash transfer: $300/month  \n- Job training program  \n- Healthcare benefits")
                    else:
                        st.error("❌ The applicant is not eligible.")
            else:
                st.error("❌ No person found with this ID in the knowledge graph.")
        else:
            st.warning("⚠️ Could not extract ID from the uploaded document.")
