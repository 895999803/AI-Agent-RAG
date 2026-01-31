from sentence_transformers import SentenceTransformer
from src.config import EMBEDDING_MODEL
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    filename='rag_project/logs/rag_app.log',
    filemode='a',
    encoding='utf-8'
)

# Initialize the sentence transformer model (loaded once)
model = None

def get_embedding(text):
    global model
    if model is None:
        try:
            logging.info(f"Loading sentence-transformers model: {EMBEDDING_MODEL}")
            model = SentenceTransformer(EMBEDDING_MODEL)
            logging.info("Sentence-transformers model loaded successfully.")
        except Exception as e:
            logging.error(f"Failed to load sentence-transformers model: {e}")
            raise

    try:
        text = text.replace("\n", " ") # Replace newlines with spaces for better embedding
        embedding = model.encode(text).tolist()  # Convert numpy array to list
        logging.debug(f"Generated embedding for text snippet.")
        return embedding
    except Exception as e:
        logging.error(f"Error generating embedding: {e}")
        raise
