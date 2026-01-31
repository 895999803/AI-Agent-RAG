from src.embedding import get_embedding
from src.database import retrieve_similar_documents
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    filename='rag_project/logs/rag_app.log',
    filemode='a',
    encoding='utf-8'
)

def retrieve_documents(query, top_k=5):
    logging.info(f"Retrieving documents for query: {query}")
    query_embedding = get_embedding(query)
    relevant_chunks = retrieve_similar_documents(query_embedding, top_k=top_k)
    logging.info(f"Retrieved {len(relevant_chunks)} relevant chunks.")
    return relevant_chunks
