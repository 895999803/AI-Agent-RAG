import os
import argparse
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    filename='rag_project/logs/rag_app.log',
    filemode='a',
    encoding='utf-8'
)


    # 1. Setup database (create table if not exists)
    setup_database()


        try:
            embedding = get_embedding(chunk)
        except Exception as e:
            continue

    logging.info("Document ingestion complete.")

def main():
    parser = argparse.ArgumentParser(description="DocQA RAG CLI")
    parser.add_argument("--query", type=str, help="Ask a question against the documents.")
    args = parser.parse_args()

    if args.ingest:
    elif args.query:
        if not get_db_connection(): # Check if DB is accessible before querying
            print("Error: Database not accessible. Please check your .env configuration and PostgreSQL setup.")
            return
        logging.info(f"Starting Q&A for query: {args.query}")
        query_text = args.query

        # 1. Retrieve relevant document chunks

        if not relevant_chunks:
            print("No relevant documents found. Please ingest documents first or try a different query.")
            logging.warning("No relevant documents found for query.")
            return

        # 2. Generate answer

        print(f"\nQuestion: {query_text}")
        print(f"Answer: {answer}")
        logging.info("Q&A session complete.")
    else:
        print("Please use --ingest to process documents or --query 'your question' to ask questions.")
        print("Example: python src/main.py --ingest")
        print("Example: python src/main.py --query 'What is the main topic of the document?'")

if __name__ == "__main__":
    main()
