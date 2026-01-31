import os
import sys
import argparse

# Add the src directory to the Python path so we can import modules
sys.path.append(os.path.join(os.path.dirname(__file__)))

from document_processor import process_document, process_all_documents
from embedding import get_embedding
from database import setup_database, store_embeddings, retrieve_similar_documents, get_db_connection
from generator import generate_answer
from retriever import retrieve_documents
from config import COLLECTION_NAME
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    filename='rag_project/logs/rag_app.log',
    filemode='a',
    encoding='utf-8'
)

def ingest_documents(documents_dir='documents'):
    logging.info(f"Starting document ingestion for all PDFs in {documents_dir}")

    # 1. Setup database (create table if not exists)
    setup_database()

    # 2. Process all documents into chunks with metadata
    all_chunks_with_metadata = process_all_documents(documents_dir)

    if not all_chunks_with_metadata:
        logging.warning("No documents were processed. Exiting.")
        return

    # 3. Generate embeddings for each chunk and prepare for storage with metadata
    contents_with_embeddings_and_metadata = []
    for i, (chunk, source_file) in enumerate(all_chunks_with_metadata):
        try:
            embedding = get_embedding(chunk)
            # Create metadata dictionary
            metadata = {
                "source_file": source_file,
                "chunk_index": i
            }
            contents_with_embeddings_and_metadata.append((chunk, embedding, metadata))
            logging.info(f"Generated embedding for chunk {i+1}/{len(all_chunks_with_metadata)} from {source_file}.")
        except Exception as e:
            logging.error(f"Error embedding chunk {i+1} from {source_file}: {e}")
            continue

    # 4. Store embeddings and metadata in the database
    store_embeddings(contents_with_embeddings_and_metadata)
    logging.info("Document ingestion complete.")

def main():
    parser = argparse.ArgumentParser(description="DocQA RAG CLI")
    parser.add_argument("--ingest", action="store_true", help="Ingest all PDF documents from the documents folder into the vector store.")
    parser.add_argument("--query", type=str, help="Ask a question against the documents.")
    parser.add_argument("--documents-dir", type=str, default="documents", help="Directory containing PDF documents to ingest (default: documents).")
    args = parser.parse_args()

    if args.ingest:
        ingest_documents(args.documents_dir)
        print(f"Document ingestion finished. Check logs/rag_app.log for details.")
    elif args.query:
        if not get_db_connection(): # Check if DB is accessible before querying
            print("Error: Database not accessible. Please check your .env configuration and PostgreSQL setup.")
            return
        logging.info(f"Starting Q&A for query: {args.query}")
        query_text = args.query

        # 1. Retrieve relevant document chunks
        relevant_chunks, relevant_metadata = retrieve_documents(query_text)

        if not relevant_chunks:
            print("No relevant documents found. Please ingest documents first or try a different query.")
            logging.warning("No relevant documents found for query.")
            return

        # 2. Generate answer
        answer = generate_answer(query_text, relevant_chunks, relevant_metadata)

        print(f"\nQuestion: {query_text}")
        print(f"Answer: {answer}")
        logging.info("Q&A session complete.")
    else:
        print("Please use --ingest to process documents or --query 'your question' to ask questions.")
        print("Example: python src/main.py --ingest")
        print("Example: python src/main.py --query 'What is the main topic of the document?'")

if __name__ == "__main__":
    main()
