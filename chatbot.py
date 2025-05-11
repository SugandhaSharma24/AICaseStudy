import pysqlite3
sys.modules["sqlite3"] = pysqlite3
import spacy
import faiss
import os
import numpy as np
import networkx as nx
import streamlit as st
from sentence_transformers import SentenceTransformer
import torch
from langdetect import detect
import PyPDF2
from langfuse import Langfuse
from llm_guard.input_scanners import PromptInjection
from llm_guard.input_scanners.prompt_injection import MatchType
from dotenv import load_dotenv
import re

# ---------- Setup ----------
def run():
    load_dotenv()
    scanner = PromptInjection(threshold=0.5, match_type=MatchType.FULL)
    os.environ["STREAMLIT_WATCHER_TYPE"] = "none"

    # ---------- Langfuse Setup ----------
    langfuse = None
    try:
        langfuse = Langfuse(
            public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
            secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
            host=os.getenv("LANGFUSE_HOST")
        )
    except Exception as e:
        st.warning(f"Langfuse setup failed: {e}")

    st.title("📚 PDF Clause Chatbot (English & Arabic)")

    # ---------- Model Loaders ----------
    @st.cache_resource
    def load_spacy_model(lang):
        if lang == 'en':
            return spacy.load("en_core_web_sm")
        elif lang == 'ar':
            return spacy.load("xx_sent_ud_sm")
        else:
            return spacy.load("en_core_web_sm")

    @st.cache_resource
    def load_sentence_model():
        return SentenceTransformer("distiluse-base-multilingual-cased-v1", device="cpu")

    # ---------- Text Extraction ----------
    def extract_text_from_pdf(file_path):
        with open(file_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            return " ".join([page.extract_text() or "" for page in reader.pages])

    def load_documents(folder_path):
        texts = []
        for file in os.listdir(folder_path):
            if file.endswith(".pdf"):
                full_path = os.path.join(folder_path, file)
                texts.append(extract_text_from_pdf(full_path))
        return texts

    # ---------- Clause Chunking ----------
    def chunk_text_by_clauses(text):
        pattern = r'(Clause \d+:)'
        parts = re.split(pattern, text)
        clauses = []
        i = 1
        while i < len(parts) - 1:
            clause_title = parts[i].strip()
            clause_text = parts[i + 1].strip()
            full_clause = f"{clause_title} {clause_text}"
            clauses.append(full_clause)
            i += 2
        return clauses

    def fallback_chunk(text, size=100):
        words = text.split()
        return [" ".join(words[i:i + size]) for i in range(0, len(words), size)]

    # ---------- Graph + Embeddings ----------
    def extract_entities(text_chunk, nlp):
        doc = nlp(text_chunk)
        return [(ent.text.strip(), ent.label_) for ent in doc.ents]

    def build_knowledge_graph(chunks, nlp):
        G = nx.Graph()
        for i, chunk in enumerate(chunks):
            ents = extract_entities(chunk, nlp)
            sentence_node = f"chunk_{i}"
            G.add_node(sentence_node, type="chunk")
            for ent_text, label in ents:
                if ent_text:
                    G.add_node(ent_text, type=label)
                    G.add_edge(sentence_node, ent_text)
        return G

    def find_chunks_by_entity(query, graph, chunks):
        matches = [node for node in graph.nodes if query.lower() in node.lower()]
        related_chunks = set()
        for node in matches:
            for neighbor in graph.neighbors(node):
                if neighbor.startswith("chunk_"):
                    idx = int(neighbor.replace("chunk_", ""))
                    related_chunks.add(chunks[idx])
        return list(related_chunks)

    def create_embeddings(chunks, model):
        return model.encode(chunks)

    def retrieve_relevant_chunks(query, model, index, chunks, top_k=3):
        query_emb = model.encode([query])
        D, I = index.search(np.array(query_emb), top_k)
        return [chunks[i] for i in I[0]]

    def detect_language(text):
        try:
            lang = detect(text)
            return 'ar' if lang == 'ar' else 'en'
        except:
            return 'en'

    # ---------- Main UI ----------
    query = st.text_input("Ask a question:")

    if query:
        sanitized_prompt, is_valid, risk_score = scanner.scan(query)
        if not is_valid:
            st.error("🚫 Unsafe input detected.")
            st.stop()

        trace = None
        if langfuse:
            try:
                trace = langfuse.trace(
                    name="user_query",
                    input=sanitized_prompt,
                    metadata={"risk_score": risk_score}
                )
            except Exception as e:
                st.warning(f"Langfuse trace error: {e}")

        language = detect_language(query)
        st.markdown(f"🌐 Detected language: **{'Arabic' if language == 'ar' else 'English'}**")

        folder_path = f"data/{'arabic' if language == 'ar' else 'english'}"
        documents = load_documents(folder_path)

        if not documents:
            st.error("❌ No documents found.")
        else:
            full_text = " ".join(documents)

            # Clause-specific chunking
            chunks = chunk_text_by_clauses(full_text)
            if not chunks:
                chunks = fallback_chunk(full_text)

            nlp = load_spacy_model(language)
            sentence_model = load_sentence_model()
            embeddings = create_embeddings(chunks, sentence_model)

            index = faiss.IndexFlatL2(embeddings.shape[1])
            index.add(np.array(embeddings))

            graph = build_knowledge_graph(chunks, nlp)
            entity_results = find_chunks_by_entity(query, graph, chunks)
            semantic_results = retrieve_relevant_chunks(query, sentence_model, index, chunks)

            combined_results = list(set(entity_results + semantic_results))[:3]

            st.markdown("### 🔍 Top Relevant Clauses:")
            if combined_results:
                for i, result in enumerate(combined_results):
                    st.markdown(f"**{i+1}.** {result}")
            else:
                st.info("🔎 No matching clauses found.")

            if trace:
                try:
                    trace.update(output={
                        "entity_matches": entity_results,
                        "semantic_matches": semantic_results,
                        "final_results": combined_results
                    })
                except Exception as e:
                    st.warning(f"Langfuse trace update failed: {e}")
