import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    filename='rag_project/logs/rag_app.log',
    filemode='a',
    encoding='utf-8'
)

    logging.info(f"Retrieving documents for query: {query}")
    query_embedding = get_embedding(query)
    logging.info(f"Retrieved {len(relevant_chunks)} relevant chunks.")
