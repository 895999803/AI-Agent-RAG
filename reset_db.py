#!/usr/bin/env python3
"""
Database Reset Script
This script resets the database table to match the current embedding dimensions.
Run this script when switching between different embedding models.
"""

import psycopg2
from pgvector.psycopg2 import register_vector
import logging
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Database configuration
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "linxiaomo")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "rag_db")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "document_embeddings")

# Embedding configuration
EMBEDDING_DIMENSION = 384  # Current dimension for sentence-transformers

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

def reset_database_table():
    """Reset the database table to match current embedding dimensions"""
    conn = None
    try:
        # Connect to database
        logging.info("Connecting to database...")
        conn = psycopg2.connect(
            user=DB_USER,
            password=DB_PASSWORD,
            host=DB_HOST,
            port=DB_PORT,
            dbname=DB_NAME,
            options="-c client_encoding=utf8"
        )
        register_vector(conn)
        logging.info("[SUCCESS] Database connection established")

        cur = conn.cursor()

        # Check current table structure
        cur.execute("""
            SELECT column_name, data_type, udt_name
            FROM information_schema.columns
            WHERE table_name = %s AND table_schema = 'public'
            ORDER BY ordinal_position;
        """, (COLLECTION_NAME,))
        columns = cur.fetchall()

        if columns:
            logging.info(f"Current table '{COLLECTION_NAME}' structure:")
            for col in columns:
                logging.info(f"  {col[0]}: {col[1]} ({col[2]})")
        else:
            logging.info(f"Table '{COLLECTION_NAME}' does not exist")

        # Drop existing table if it exists
        logging.info(f"Dropping existing table '{COLLECTION_NAME}' if it exists...")
        cur.execute(f"DROP TABLE IF EXISTS {COLLECTION_NAME};")
        conn.commit()
        logging.info("[SUCCESS] Old table dropped")

        # Create new table with correct dimensions and metadata
        logging.info(f"Creating new table '{COLLECTION_NAME}' with {EMBEDDING_DIMENSION} dimensions and metadata...")
        cur.execute(f"""
            CREATE TABLE {COLLECTION_NAME} (
                id serial PRIMARY KEY,
                content TEXT NOT NULL,
                embedding VECTOR({EMBEDDING_DIMENSION}),
                metadata JSONB
            );
        """)
        conn.commit()
        logging.info("[SUCCESS] New table created")

        # Verify table creation
        cur.execute(f"SELECT COUNT(*) FROM {COLLECTION_NAME};")
        count = cur.fetchone()[0]
        logging.info(f"Table verification: {count} rows (should be 0)")

        # Show new table structure
        cur.execute("""
            SELECT column_name, data_type, udt_name
            FROM information_schema.columns
            WHERE table_name = %s AND table_schema = 'public'
            ORDER BY ordinal_position;
        """, (COLLECTION_NAME,))
        new_columns = cur.fetchall()

        logging.info(f"New table '{COLLECTION_NAME}' structure:")
        for col in new_columns:
            logging.info(f"  {col[0]}: {col[1]} ({col[2]})")

        cur.close()

        logging.info("SUCCESS: Database table reset completed successfully!")
        logging.info(f"Table '{COLLECTION_NAME}' now supports {EMBEDDING_DIMENSION}-dimensional embeddings")

    except Exception as e:
        logging.error(f"[FAILED] Database reset failed: {e}")
        raise
    finally:
        if conn:
            conn.close()
            logging.info("Database connection closed")

def test_table_functionality():
    """Test that the new table works with current embedding dimensions"""
    conn = None
    try:
        logging.info("Testing table functionality with sample embeddings...")

        conn = psycopg2.connect(
            user=DB_USER,
            password=DB_PASSWORD,
            host=DB_HOST,
            port=DB_PORT,
            dbname=DB_NAME,
            options="-c client_encoding=utf8"
        )
        register_vector(conn)
        cur = conn.cursor()

        # Insert test data
        test_embedding = [0.1] * EMBEDDING_DIMENSION  # 384-dimensional test vector
        test_content = "This is a test document for verifying database functionality."

        cur.execute(f"INSERT INTO {COLLECTION_NAME} (content, embedding) VALUES (%s, %s);",
                    (test_content, test_embedding))
        conn.commit()
        logging.info("[SUCCESS] Test embedding inserted")

        # Query test data
        cur.execute(f"SELECT content, embedding FROM {COLLECTION_NAME} LIMIT 1;")
        result = cur.fetchone()
        if result:
            content, embedding = result
            logging.info(f"[SUCCESS] Retrieved content: {content[:50]}...")
            logging.info(f"[SUCCESS] Embedding dimension: {len(embedding)}")

        # Clean up test data
        cur.execute(f"DELETE FROM {COLLECTION_NAME} WHERE content = %s;", (test_content,))
        conn.commit()
        logging.info("[SUCCESS] Test data cleaned up")

        cur.close()
        logging.info("[SUCCESS] Table functionality test passed!")

    except Exception as e:
        logging.error(f"[FAILED] Table functionality test failed: {e}")
        raise
    finally:
        if conn:
            conn.close()

if __name__ == "__main__":
    print("Database Table Reset Script")
    print("=" * 50)

    try:
        # Reset the database table
        reset_database_table()
        print()

        # Test the new table
        test_table_functionality()
        print()

        print("SUCCESS: Database reset completed successfully!")
        print(f"Table '{COLLECTION_NAME}' is now ready for {EMBEDDING_DIMENSION}-dimensional embeddings.")

    except Exception as e:
        print(f"ERROR: Database reset failed: {e}")
        print("Please check your database connection and try again.")
        exit(1)
