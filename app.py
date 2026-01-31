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
from src.database import setup_database, store_embeddings, get_db_connection
from src.retriever import retrieve_documents
from src.generator import generate_answer
from src.query_expansion import expand_query_with_strategy, get_query_expansion_strategies
from src.evaluator import evaluate_answer_quality, format_evaluation_display, evaluate_chunks_relevance, format_chunk_evaluation_display
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
with st.sidebar.expander("Upload & Ingest Documents"):
    st.write("**Select documents to ingest from the documents folder:**")
    
    # Get list of PDF files in documents directory
    documents_dir = os.path.join(os.path.dirname(__file__), 'documents')
    pdf_files = []
    if os.path.exists(documents_dir):
        pdf_files = [f for f in os.listdir(documents_dir) if f.endswith('.pdf')]
    
    if not pdf_files:
        st.warning("No PDF files found in the documents folder.")
    else:
        # Create checkboxes for each document
        selected_files = []
        for pdf_file in pdf_files:
            if st.checkbox(f"✓ {pdf_file}", key=f"checkbox_{pdf_file}"):
                selected_files.append(pdf_file)
        
        # Option to select all
        if st.checkbox("✓ Select All", key="select_all"):
            selected_files = pdf_files
        
        # Ingestion button
        if st.button("Ingest Selected Documents", key="ingest_btn"):
            if db_connected:
                if not selected_files:
                    st.warning("Please select at least one document to ingest.")
                else:
                    try:
                        total_chunks = 0
                        with st.spinner("Processing and embedding documents..."):
                            # Process each selected document
                            for pdf_file in selected_files:
                                pdf_path = os.path.join(documents_dir, pdf_file)
                                
                                # 1. Process document into chunks
                                chunks_with_metadata = process_document(pdf_path)
                                total_chunks += len(chunks_with_metadata)

                                # 2. Generate embeddings for each chunk
                                contents_with_embeddings = []
                                for i, (chunk, source_file) in enumerate(chunks_with_metadata):
                                    embedding = get_embedding(chunk)
                                    contents_with_embeddings.append((chunk, embedding, source_file))

                                # 3. Store embeddings in the database
                                store_embeddings(contents_with_embeddings)
                                st.write(f"✅ Processed {pdf_file}: {len(chunks_with_metadata)} chunks")
                        
                        # Update status after ingestion
                        st.cache_resource.clear() # Clear cache to refresh status
                        db_content_status = get_db_content_status()
                        
                        if db_content_status == "No table":
                            st.info(f"ℹ️ This is the first ingestion. {total_chunks} chunks have been added to a new table.")
                        elif "chunks" in db_content_status:
                            st.info(f"✅ Successfully ingested {len(selected_files)} documents. Total chunks: {total_chunks}. {db_content_status}")
                        else:
                            st.warning(f"⚠️ Ingestion completed but status unclear: {db_content_status}")
                        
                        logging.info(f"Documents ingested via Streamlit app: {selected_files}")
                    except Exception as e:
                        st.error(f"Error during ingestion: {e}")
                        logging.error(f"Error during ingestion: {e}")
            else:
                st.error("Cannot ingest: Database not connected.")

# --- Main Q&A Interface ---
st.header("Ask a Question")

# File selection for retrieval
st.write("**Select files to search in (optional):**")
documents_dir = os.path.join(os.path.dirname(__file__), 'documents')
pdf_files = []
if os.path.exists(documents_dir):
    pdf_files = [f for f in os.listdir(documents_dir) if f.endswith('.pdf')]

selected_files = []
if pdf_files:
    # Create a more compact horizontal layout for file selection
    cols = st.columns(len(pdf_files) + 1)  # +1 for "Select All" checkbox
    
    # "Select All" checkbox in the first column
    with cols[0]:
        select_all = st.checkbox("✓ All", key="search_all_files", help="Select all files")
        if select_all:
            selected_files = pdf_files
    
    # Individual file checkboxes in subsequent columns
    for i, pdf_file in enumerate(pdf_files):
        with cols[i + 1]:
            if st.checkbox(f"✓ {pdf_file}", key=f"search_{pdf_file}", help=f"Search in {pdf_file}"):
                if pdf_file not in selected_files:
                    selected_files.append(pdf_file)
    
    # Check if selected files have been ingested and show status
    if selected_files and db_connected:
        try:
            conn = get_db_connection()
            cur = conn.cursor()
            
            # Check which selected files have data in the database
            cur.execute(f"SELECT DISTINCT source_file FROM {COLLECTION_NAME} WHERE source_file = ANY(%s);", (selected_files,))
            ingested_files = [row[0] for row in cur.fetchall()]
            
            # Find files that haven't been ingested
            not_ingested_files = [f for f in selected_files if f not in ingested_files]
            
            if not_ingested_files:
                st.warning(f"⚠️ The following selected files have not been ingested into the database: {', '.join(not_ingested_files)}")
                st.info("Please go to the sidebar, select these files, and click 'Ingest Selected Documents' to add them to the database.")
            else:
                st.success(f"✅ All selected files are available for search: {', '.join(selected_files)}")
            
            cur.close()
            conn.close()
        except Exception as e:
            logging.error(f"Error checking file ingestion status: {e}")
            st.error(f"Error checking file status: {e}")
else:
    st.info("No PDF files found. Please upload documents first.")

# Query Expansion Configuration
st.sidebar.header("🔍 Query Expansion")

# Query Expansion Options
use_query_expansion = st.sidebar.checkbox("Enable Query Expansion", value=False, help="Expand the original query to improve retrieval results")

if use_query_expansion:
    expansion_strategy = st.sidebar.selectbox(
        "Expansion Strategy",
        options=list(get_query_expansion_strategies().keys()),
        format_func=lambda x: x.replace('_', ' ').title(),
        help="Choose how to expand the query"
    )
    max_expanded_queries = st.sidebar.slider(
        "Max Expanded Queries",
        min_value=1,
        max_value=5,
        value=3,
        help="Number of expanded queries to generate"
    )

user_query = st.text_input("Your Question:", "What is the main topic of the document?")

if st.button("Get Answer", key="get_answer_btn"):
    if not db_connected:
        st.error("Cannot answer: Database not connected. Please check your setup.")
        logging.error("Q&A failed: Database not connected.")
    else:
        if user_query:
            with st.spinner("Retrieving and generating answer..."):
                try:
                    # Query Expansion Logic
                    if use_query_expansion:
                        st.info(f"🔍 Expanding query using {expansion_strategy.replace('_', ' ').title()} strategy...")
                        expanded_queries = expand_query_with_strategy(
                            user_query, 
                            strategy=expansion_strategy, 
                            max_expanded_queries=max_expanded_queries
                        )
                        
                        # Display original query
                        st.write(f"**Original Query:** {user_query}")
                        
                        # Display expanded queries with checkboxes for user selection
                        st.write(f"**Select Queries to Use:**")
                        selected_queries = []
                        
                        # Always include original query as an option
                        original_checked = st.checkbox(f"✓ Original: {user_query}", value=True, key="query_original")
                        if original_checked:
                            selected_queries.append(user_query)
                        
                        # Display expanded queries with checkboxes
                        for i, query in enumerate(expanded_queries[1:], 1):  # Skip original query (index 0)
                            query_checked = st.checkbox(f"✓ Expanded {i}: {query}", value=True, key=f"query_{i}")
                            if query_checked:
                                selected_queries.append(query)
                        
                        # Show summary of selected queries
                        if selected_queries:
                            st.success(f"✅ Selected {len(selected_queries)} queries for retrieval")
                            st.write("**Selected Queries:**")
                            for j, query in enumerate(selected_queries, 1):
                                st.write(f"{j}. {query}")
                        else:
                            st.warning("⚠️ No queries selected. Using original query only.")
                            selected_queries = [user_query]
                    else:
                        selected_queries = [user_query]
                    
                    # Retrieve documents for selected queries and combine results
                    all_relevant_chunks = []
                    all_status_info = []
                    
                    for query in selected_queries:
                        source_files = selected_files if selected_files else None
                        relevant_chunks, status_info = retrieve_documents(query, top_k=5, source_files=source_files)
                        all_relevant_chunks.extend(relevant_chunks)
                        all_status_info.append(status_info)
                    
                    # Remove duplicates while preserving order
                    seen = set()
                    unique_chunks = []
                    for chunk in all_relevant_chunks:
                        if chunk not in seen:
                            seen.add(chunk)
                            unique_chunks.append(chunk)
                    
                    # Determine overall status
                    overall_status = "success"
                    for status_info in all_status_info:
                        if status_info['status'] == 'error':
                            overall_status = 'error'
                            break
                        elif status_info['status'] == 'no_files_found':
                            overall_status = 'no_files_found'
                        elif status_info['status'] == 'partial_success' and overall_status != 'error':
                            overall_status = 'partial_success'
                    
                    # 处理检索状态
                    if overall_status == 'no_files_found':
                        st.warning("No files found for any of the selected queries.")
                        st.info(f"数据库中可用的文件: {', '.join(all_status_info[0]['available_files']) if all_status_info and all_status_info[0]['available_files'] else '无'}")
                        logging.warning(f"No files found for queries: {selected_queries}. Available files: {all_status_info[0]['available_files'] if all_status_info else []}")
                    elif overall_status == 'partial_success':
                        st.warning(f"⚠️ 部分文件可用")
                        missing_files = set()
                        for status_info in all_status_info:
                            if 'missing_files' in status_info:
                                missing_files.update(status_info['missing_files'])
                        if missing_files:
                            st.info(f"以下文件不存在: {', '.join(missing_files)}")
                        logging.info(f"Partial success for queries: {selected_queries}. Missing files: {missing_files}")
                    elif overall_status == 'error':
                        st.error("检索过程中发生错误")
                        logging.error(f"Retrieval error for queries: {selected_queries}")
                    else:
                        # 正常检索成功
                        if not unique_chunks:
                            st.warning("No relevant document parts found. Try ingesting documents first.")
                            logging.warning(f"No relevant chunks for queries: {selected_queries}")
                        else:
                            # 2. Generate answer using all relevant chunks
                            answer = generate_answer(user_query, unique_chunks)
                            
                            # 3. Evaluate answer quality
                            evaluation_result = evaluate_answer_quality(user_query, unique_chunks, answer)
                            
                            # 4. Display results
                            st.subheader("Answer:")
                            st.write(answer)
                            
                            # Display evaluation score in collapsible section
                            with st.expander("Answer Quality Assessment", expanded=False):
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
                            
                            # Evaluate and rank chunks by relevance
                            with st.spinner("Evaluating chunk relevance..."):
                                chunk_evaluations = evaluate_chunks_relevance(user_query, unique_chunks)
                            
                            # Display ranked chunks with compact visualization
                            st.subheader("Document Chunks Ranked by Relevance")
                            
                            # Create a more compact display with visual indicators
                            for i, eval_result in enumerate(chunk_evaluations):
                                score = eval_result['score']
                                chunk = eval_result['chunk']
                                
                                # Create visual score indicator
                                score_percentage = score / 10.0
                                
                                # Determine color based on score
                                if score >= 8.0:
                                    color = "🟢"  # Green for high scores
                                elif score >= 6.0:
                                    color = "🟡"  # Yellow for medium scores
                                elif score >= 4.0:
                                    color = "🟠"  # Orange for low-medium scores
                                else:
                                    color = "🔴"  # Red for low scores
                                
                                # Create score bar visualization
                                filled_bars = int(score_percentage * 10)
                                empty_bars = 10 - filled_bars
                                score_bar = "█" * filled_bars + "░" * empty_bars
                                
                                # Create expandable section for each chunk
                                with st.expander(f"{color} Chunk {i+1} - Score: {score}/10.0"):
                                    st.write(f"**Score Breakdown:**")
                                    if eval_result['details']['evaluation_passed']:
                                        metrics = eval_result['details']['metric_scores']
                                        for metric_name, metric_score in metrics.items():
                                            st.write(f"- {metric_name.replace('_', ' ').title()}: {metric_score}/10.0")
                                    
                                    st.write(f"**Content:**")
                                    st.info(chunk)
                            
                            logging.info(f"Answered query '{user_query}' successfully with quality score {evaluation_result.get('overall_score', 'N/A')}/5.0.")

                except Exception as e:
                    st.error(f"Error during Q&A: {e}")
                    logging.error(f"Error during Q&A for query '{user_query}': {e}")
        else:
            st.warning("Please enter a question.")
            logging.warning("User tried to get answer with empty query.")

# python -m streamlit run C:\Users\Forever\Desktop\test\rag_project\app.py --server.port 8501
