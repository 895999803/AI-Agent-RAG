import psycopg2
from pgvector.psycopg2 import register_vector

def test_db_connection():
    """Test basic database connection"""
    try:
        conn = psycopg2.connect(
            user="postgres",
            password="linxiaomo",
            host="localhost",
            port=5432,
            dbname="rag_db",
            options="-c client_encoding=utf8"
        )
        # Don't register vector here - test basic connection first
        print("[SUCCESS] Basic connection successful!")
        conn.close()
        return True
    except Exception as e:
        print(f"[FAILED] Connection failed: {e}")
        return False

def test_pgvector_extension():
    """Test if pgvector extension is available"""
    try:
        conn = psycopg2.connect(
            user="postgres",
            password="linxiaomo",
            host="localhost",
            port=5432,
            dbname="rag_db",
            options="-c client_encoding=utf8"
        )
        register_vector(conn)
        cur = conn.cursor()

        # Check if vector extension exists
        cur.execute("SELECT * FROM pg_extension WHERE extname = 'vector';")
        if cur.fetchone():
            print("[SUCCESS] pgvector extension is installed!")

            # Test vector functionality
            cur.execute("CREATE TEMP TABLE test_vector (id serial PRIMARY KEY, vec vector(3));")
            cur.execute("INSERT INTO test_vector (vec) VALUES ('[1,2,3]');")
            cur.execute("SELECT vec FROM test_vector;")
            result = cur.fetchone()
            print(f"[SUCCESS] Vector operations work: {result[0]}")

            cur.close()
            conn.close()
            return True
        else:
            print("[FAILED] pgvector extension is not installed!")
            print("  Please install pgvector extension by running:")
            print("  CREATE EXTENSION vector;")
            cur.close()
            conn.close()
            return False
    except Exception as e:
        print(f"[FAILED] pgvector test failed: {e}")
        return False

def test_database_setup():
    """Test database setup functionality"""
    try:
        from src.database import setup_database
        setup_database()
        print("[SUCCESS] Database setup successful!")
        return True
    except Exception as e:
        print(f"[FAILED] Database setup failed: {e}")
        return False

if __name__ == "__main__":
    print("Testing database connection and functionality...\n")

    # Test 1: Basic connection
    print("1. Testing basic connection:")
    conn_ok = test_db_connection()
    print()

    # Test 2: pgvector extension (only if connection works)
    if conn_ok:
        print("2. Testing pgvector extension:")
        vector_ok = test_pgvector_extension()
        print()

        # Test 3: Database setup
        if vector_ok:
            print("3. Testing database setup:")
            setup_ok = test_database_setup()
            print()

            if setup_ok:
                print("SUCCESS: All tests passed! Database is ready for the RAG application.")
            else:
                print("WARNING: Database setup failed. Check logs for details.")
        else:
            print("ERROR: pgvector extension test failed. Cannot proceed with database setup.")
    else:
        print("ERROR: Basic connection failed. Cannot proceed with further tests.")
