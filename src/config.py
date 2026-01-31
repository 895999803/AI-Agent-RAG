import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENAI_API_KEY")  # Using the same key for now, can be changed later
DB_USER = os.getenv("DB_USER", "user")
DB_PASSWORD = os.getenv("DB_PASSWORD", "password")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "rag_db")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "document_embeddings")

# Embedding model parameters
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Sentence-transformers model
EMBEDDING_DIMENSION = 384  # Dimension for all-MiniLM-L6-v2

# LLM model parameters
LLM_MODEL = "allenai/molmo-2-8b:free"
TEMPERATURE = 0.7

# Document processing parameters
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
