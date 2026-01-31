import os
import argparse
from src.document_processor import process_document
from src.embedding import get_embedding
from src.database import setup_database, store_embeddings, retrieve_similar_documents, get_db_connection
from src.generator import generate_answer
from src.retriever import retrieve_documents
from src.config import COLLECTION_NAME
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    filename='rag_project/logs/rag_app.log',
    filemode='a',
    encoding='utf-8'
)

def ingest_documents(pdf_path):
    logging.info(f"Starting document ingestion for {pdf_path}")

    # 1. Setup database (create table if not exists)
    setup_database()

    # 2. Process document into chunks
    chunks = process_document(pdf_path)

    # 3. Generate embeddings for each chunk
    contents_with_embeddings = []
    for i, chunk in enumerate(chunks):
        try:
            embedding = get_embedding(chunk)
            contents_with_embeddings.append((chunk, embedding))
            logging.info(f"Generated embedding for chunk {i+1}/{len(chunks)}.")
        except Exception as e:
            logging.error(f"Error embedding chunk {i+1}: {e}")
            continue

    # 4. Store embeddings in the database
    store_embeddings(contents_with_embeddings)
    logging.info("Document ingestion complete.")

def main():
    parser = argparse.ArgumentParser(description="DocQA RAG CLI")
    parser.add_argument("--ingest", action="store_true", help="Ingest documents into the vector store.")
    parser.add_argument("--query", type=str, help="Ask a question against the documents.")
    args = parser.parse_args()

    if args.ingest:
        pdf_path = os.path.join(os.path.dirname(__file__), '..', 'documents', 'document.pdf')
        if not os.path.exists(pdf_path):
            logging.error(f"Document not found at {pdf_path}. Please ensure it exists.")
            print(f"Error: Document not found at {pdf_path}. Please ensure it exists.")
            return
        ingest_documents(pdf_path)
        print("Document ingestion finished. Check logs/rag_app.log for details.")
    elif args.query:
        if not get_db_connection(): # Check if DB is accessible before querying
            print("Error: Database not accessible. Please check your .env configuration and PostgreSQL setup.")
            return
        logging.info(f"Starting Q&A for query: {args.query}")
        query_text = args.query

        # 1. Retrieve relevant document chunks
        relevant_chunks = retrieve_documents(query_text)

        if not relevant_chunks:
            print("No relevant documents found. Please ingest documents first or try a different query.")
            logging.warning("No relevant documents found for query.")
            return

        # 2. Generate answer
        answer = generate_answer(query_text, relevant_chunks)

        print(f"\nQuestion: {query_text}")
        print(f"Answer: {answer}")
        logging.info("Q&A session complete.")
    else:
        print("Please use --ingest to process documents or --query 'your question' to ask questions.")
        print("Example: python src/main.py --ingest")
        print("Example: python src/main.py --query 'What is the main topic of the document?'")

if __name__ == "__main__":
    main()
