import psycopg2
from pgvector.psycopg2 import register_vector
from src.config import DB_USER, DB_PASSWORD, DB_HOST, DB_PORT, DB_NAME, COLLECTION_NAME, EMBEDDING_DIMENSION
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    filename='rag_project/logs/rag_app.log',
    filemode='a',
    encoding='utf-8'
)

def get_db_connection():
    try:
        conn = psycopg2.connect(
            user=DB_USER,
            password=DB_PASSWORD,
            host=DB_HOST,
            port=DB_PORT,
            dbname=DB_NAME,
            options="-c client_encoding=utf8"
        )
        register_vector(conn)
        logging.info("Successfully connected to the database.")
        return conn
    except Exception as e:
        logging.error(f"Database connection failed: {e}")
        raise

def setup_database():
    conn = None
    try:
        conn = get_db_connection()
        cur = conn.cursor()

        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        conn.commit()
        logging.info("pgvector extension ensured.")

        # Check if table exists
        cur.execute(f"SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = '{COLLECTION_NAME}');")
        table_exists = cur.fetchone()[0]

        if table_exists:
            logging.info(f"Table {COLLECTION_NAME} already exists. Skipping table creation.")
        else:
            # Create table if it doesn't exist
            logging.info(f"EMBEDDING_DIMENSION is set to: {EMBEDDING_DIMENSION}")
            cur.execute(f"""
                CREATE TABLE {COLLECTION_NAME} (
                    id serial PRIMARY KEY,
                    content TEXT NOT NULL,
                );
            """)
            conn.commit()
            logging.info(f"Table {COLLECTION_NAME} created with {EMBEDDING_DIMENSION} dimensions.")

        # Verify table was created correctly
        cur.execute(f"SELECT COUNT(*) FROM {COLLECTION_NAME};")
        count = cur.fetchone()[0]
        logging.info(f"Table {COLLECTION_NAME} has {count} rows after setup.")

        cur.close()
    except Exception as e:
        logging.error(f"Database setup failed: {e}")
        raise
    finally:
        if conn:
            conn.close()

def store_embeddings(contents_with_embeddings):
    conn = None
    try:
        conn = get_db_connection()
        cur = conn.cursor()

        # Check if table exists
        cur.execute(f"SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = '{COLLECTION_NAME}');")
        table_exists = cur.fetchone()[0]

        if not table_exists:
            # Create table if it doesn't exist
            logging.info(f"Table {COLLECTION_NAME} does not exist. Creating table...")
            cur.execute(f"""
                CREATE TABLE {COLLECTION_NAME} (
                    id serial PRIMARY KEY,
                    content TEXT NOT NULL,
                );
            """)
            conn.commit()
            logging.info(f"Table {COLLECTION_NAME} created with {EMBEDDING_DIMENSION} dimensions.")

        # Check for existing content and insert new content
        new_content_count = 0
            # Check if content already exists
            cur.execute(f"SELECT COUNT(*) FROM {COLLECTION_NAME} WHERE content = %s;", (content,))
            count = cur.fetchone()[0]
            
            if count == 0:
                # Insert new content
                new_content_count += 1
            else:
                logging.info(f"Content already exists in database: {content[:50]}...")

        conn.commit()
        logging.info(f"Added {new_content_count} new embeddings to {COLLECTION_NAME}.")

        cur.close()
    except Exception as e:
        logging.error(f"Failed to store embeddings: {e}")
        raise
    finally:
        if conn:
            conn.close()

    conn = None
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        
                else:
        
        results = cur.fetchall()
        cur.close()
        conn.close()
        
