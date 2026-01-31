import sys, os

os.environ.setdefault("PYTHONIOENCODING", "utf-8")

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except:
    pass

import streamlit as st
import os
from src.document_processor import process_document
from src.embedding import get_embedding
from src.database import setup_database, store_embeddings, retrieve_similar_documents, get_db_connection
from src.generator import generate_answer
from src.evaluator import evaluate_answer_quality, format_evaluation_display
from src.config import COLLECTION_NAME
import logging

# Setup logging for the Streamlit app
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    filename='logs/rag_app.log',
    filemode='a',
    encoding='utf-8'
)

st.set_page_config(page_title="DocQA RAG Assistant", layout="centered")
st.title("🧠 DocQA RAG Assistant")

# --- Sidebar for setup and ingestion ---
st.sidebar.header("⚙️ Setup & Ingestion")

# Check DB connection status
@st.cache_resource
def get_db_status():
    try:
        conn = get_db_connection()
        conn.close()
        return True
    except Exception:
        return False

db_connected = get_db_status()
if not db_connected:
    st.sidebar.error("⚠️ Database connection failed. Check .env and PostgreSQL.")
else:
    st.sidebar.success("✅ Database connected.")

# Check database content status
@st.cache_resource
def get_db_content_status():
    if not db_connected:
        return "No connection"
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute(f"SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = '{COLLECTION_NAME}');")
        table_exists = cur.fetchone()[0]
        if not table_exists:
            return "No table"
        cur.execute(f"SELECT COUNT(*) FROM {COLLECTION_NAME};")
        count = cur.fetchone()[0]
        cur.close()
        conn.close()
        return f"Table exists with {count} chunks"
    except Exception as e:
        return f"Error: {e}"

db_content_status = get_db_content_status()
if db_content_status == "No connection":
    st.sidebar.warning("⚠️ Cannot check database content status.")
elif db_content_status == "No table":
    st.sidebar.info("ℹ️ No data table found. Ready for first ingestion.")
elif "Error:" in db_content_status:
    st.sidebar.error(f"⚠️ Error checking content: {db_content_status}")
else:
    st.sidebar.success(f"✅ {db_content_status}")

# Document Ingestion
with st.sidebar.expander("Upload & Ingest Documents"): # Changed from file_uploader to expannder with a fixed path
    st.write("**Note:** `document.pdf` from the `documents/` folder will be used for ingestion.")
    # Removed file_uploader for simplicity, using pre-defined document.pdf
    if st.button("Ingest Document", key="ingest_btn"):
        if db_connected:
            pdf_path = os.path.join(os.path.dirname(__file__), 'documents', 'document.pdf')
            if not os.path.exists(pdf_path):
                st.error(f"Document not found at {pdf_path}. Please ensure it exists.")
                logging.error(f"Ingestion failed: document not found at {pdf_path}")
            else:
                try:
                    with st.spinner("Processing and embedding document..."):
                        # 1. Process document into chunks
                        chunks = process_document(pdf_path)

                        # 2. Generate embeddings for each chunk
                        contents_with_embeddings = []
                        for i, chunk in enumerate(chunks):
                            embedding = get_embedding(chunk)
                            contents_with_embeddings.append((chunk, embedding))

                        # 3. Store embeddings in the database
                        store_embeddings(contents_with_embeddings)
                    
                    # Update status after ingestion
                    st.cache_resource.clear() # Clear cache to refresh status
                    db_content_status = get_db_content_status()
                    
                    if db_content_status == "No table":
                        st.info("ℹ️ This is the first ingestion. Data has been added to a new table.")
                    elif "chunks" in db_content_status:
                        st.info(f"✅ Data added successfully. {db_content_status}")
                    else:
                        st.warning(f"⚠️ Ingestion completed but status unclear: {db_content_status}")
                    
                    logging.info("Document ingested via Streamlit app.")
                except Exception as e:
                    st.error(f"Error during ingestion: {e}")
                    logging.error(f"Error during ingestion: {e}")
        else:
            st.error("Cannot ingest: Database not connected.")

# --- Main Q&A Interface ---
st.header("Ask a Question")

user_query = st.text_input("Your Question:", "What is the main topic of the document?")

if st.button("Get Answer", key="get_answer_btn"):
    if not db_connected:
        st.error("Cannot answer: Database not connected. Please check your setup.")
        logging.error("Q&A failed: Database not connected.")
    else:
        if user_query:
            with st.spinner("Retrieving and generating answer..."):
                try:
                    # 1. Retrieve relevant document chunks
                    query_embedding = get_embedding(user_query) # Need to generate embedding for the query
                    relevant_chunks = retrieve_similar_documents(query_embedding)

                    if not relevant_chunks:
                        st.warning("No relevant document parts found. Try ingesting documents first.")
                        logging.warning(f"No relevant chunks for query: {user_query}")
                    else:
                        # 2. Generate answer
                        answer = generate_answer(user_query, relevant_chunks)
                        
                        # 3. Evaluate answer quality
                        evaluation_result = evaluate_answer_quality(user_query, relevant_chunks, answer)
                        
                        # 4. Display results
                        st.subheader("Answer:")
                        st.write(answer)
                        
                        # Display evaluation score
                        st.subheader("Answer Quality Assessment")
                        
                        if evaluation_result['evaluation_passed']:
                            # Display score with visual indicators
                            overall_score = evaluation_result['overall_score']
                            
                            # Create columns for score display
                            col1, col2 = st.columns([1, 2])
                            
                            with col1:
                                # Large score display with color coding
                                if overall_score >= 4.5:
                                    st.metric("Quality Score", f"{overall_score}/5.0", delta=None, delta_color="normal")
                                elif overall_score >= 3.5:
                                    st.metric("Quality Score", f"{overall_score}/5.0", delta=None, delta_color="normal")
                                elif overall_score >= 2.5:
                                    st.metric("Quality Score", f"{overall_score}/5.0", delta=None, delta_color="normal")
                                elif overall_score >= 1.5:
                                    st.metric("Quality Score", f"{overall_score}/5.0", delta=None, delta_color="normal")
                                else:
                                    st.metric("Quality Score", f"{overall_score}/5.0", delta=None, delta_color="normal")
                            
                            with col2:
                                # Detailed evaluation
                                st.write(format_evaluation_display(evaluation_result))
                        else:
                            # Display evaluation failure with reasons
                            st.error("❌ Answer Quality Assessment Failed")
                            st.write("**Reasons for failure:**")
                            for reason in evaluation_result.get('failure_reasons', []):
                                st.write(f"- {reason}")
                        
                        # Show retrieved context
                        with st.expander("See Retrieved Context"):
                            for i, chunk in enumerate(relevant_chunks):
                                st.write(f"**Chunk {i+1}:**")
                                st.info(chunk)
                        
                        logging.info(f"Answered query '{user_query}' successfully with quality score {evaluation_result.get('overall_score', 'N/A')}/5.0.")

                except Exception as e:
                    st.error(f"Error during Q&A: {e}")
                    logging.error(f"Error during Q&A for query '{user_query}': {e}")
        else:
            st.warning("Please enter a question.")
            logging.warning("User tried to get answer with empty query.")

# python -m streamlit run C:\Users\Forever\Desktop\test\rag_project\app.py --server.port 8501
