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
            # Check if source_file column exists
            cur.execute(f"SELECT column_name FROM information_schema.columns WHERE table_name = '{COLLECTION_NAME}' AND column_name = 'source_file';")
            source_file_exists = cur.fetchone()
            
            if not source_file_exists:
                # Add source_file column if it doesn't exist
                logging.info("Adding source_file column to existing table...")
                cur.execute(f"ALTER TABLE {COLLECTION_NAME} ADD COLUMN source_file TEXT;")
                conn.commit()
                logging.info("source_file column added successfully.")
            
            logging.info(f"Table {COLLECTION_NAME} already exists. Skipping table creation.")
        else:
            # Create table if it doesn't exist
            logging.info(f"EMBEDDING_DIMENSION is set to: {EMBEDDING_DIMENSION}")
            cur.execute(f"""
                CREATE TABLE {COLLECTION_NAME} (
                    id serial PRIMARY KEY,
                    content TEXT NOT NULL,
                    embedding VECTOR({EMBEDDING_DIMENSION}),
                    source_file TEXT
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
                    embedding VECTOR({EMBEDDING_DIMENSION}),
                    source_file TEXT
                );
            """)
            conn.commit()
            logging.info(f"Table {COLLECTION_NAME} created with {EMBEDDING_DIMENSION} dimensions.")

        # Check for existing content and insert new content
        new_content_count = 0
        for item in contents_with_embeddings:
            if isinstance(item, tuple) and len(item) == 3:
                # Format: (content, embedding, source_file)
                content, embedding, source_file = item
            elif isinstance(item, tuple) and len(item) == 2:
                # Format: (content, embedding) - backward compatibility
                content, embedding = item
                source_file = None
            else:
                # Handle unexpected format
                logging.warning(f"Unexpected item format: {type(item)}")
                continue

            # Check if content already exists
            cur.execute(f"SELECT COUNT(*) FROM {COLLECTION_NAME} WHERE content = %s;", (content,))
            count = cur.fetchone()[0]
            
            if count == 0:
                # Insert new content
                cur.execute(f"INSERT INTO {COLLECTION_NAME} (content, embedding, source_file) VALUES (%s, %s, %s);",
                            (content, embedding, source_file))
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

def retrieve_similar_documents(query_embedding, top_k=5, source_files=None):
    """
    从数据库中检索与查询向量最相似的文档
    
    Args:
        query_embedding: 查询向量
        top_k: 返回的最相似文档数量
        source_files: 可选的源文件列表，如果提供则只检索这些文件中的文档
    
    Returns:
        tuple: (文档内容列表, 检索状态信息)
            - 文档内容列表: 相似文档的内容列表
            - 检索状态信息: 包含检索结果的详细信息
    """
    conn = None
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        
        # 检查指定的文件是否存在于数据库中
        available_files = []
        if source_files:
            # 获取数据库中所有可用的源文件
            cur.execute(f"SELECT DISTINCT source_file FROM {COLLECTION_NAME} WHERE source_file IS NOT NULL")
            db_files = [row[0] for row in cur.fetchall()]
            
            # 检查用户指定的文件是否都存在
            for file in source_files:
                if file in db_files:
                    available_files.append(file)
                else:
                    logging.warning(f"指定的文件 {file} 在数据库中不存在")
        
        # 构建查询
        if source_files and available_files:
            # 如果指定了源文件且有可用文件，只检索这些文件中的文档
            query = f"""
                SELECT content, embedding <-> %s::vector as distance
                FROM {COLLECTION_NAME} 
                WHERE source_file = ANY(%s)
                ORDER BY distance 
                LIMIT %s
            """
            cur.execute(query, (query_embedding, available_files, top_k))
        elif source_files and not available_files:
            # 如果指定的文件都不存在，返回空结果
            cur.close()
            conn.close()
            return [], {
                'status': 'no_files_found',
                'message': f"指定的文件 {source_files} 在数据库中不存在，请先进行数据注入",
                'available_files': [],
                'requested_files': source_files
            }
        else:
            # 检索所有文档
            query = f"""
                SELECT content, embedding <-> %s::vector as distance
                FROM {COLLECTION_NAME} 
                ORDER BY distance 
                LIMIT %s
            """
            cur.execute(query, (query_embedding, top_k))
        
        results = cur.fetchall()
        cur.close()
        conn.close()
        
        # 获取数据库中所有可用的文件
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute(f"SELECT DISTINCT source_file FROM {COLLECTION_NAME} WHERE source_file IS NOT NULL")
        all_available_files = [row[0] for row in cur.fetchall()]
        cur.close()
        conn.close()
        
        documents = [result[0] for result in results]
        
        # 构建状态信息
        if source_files:
            if available_files:
                # 检查是否所有请求的文件都可用
                all_files_available = len(available_files) == len(source_files)
                if all_files_available:
                    status_info = {
                        'status': 'success',
                        'message': f"成功检索到 {len(documents)} 个相关文档",
                        'available_files': available_files,
                        'requested_files': source_files,
                        'missing_files': []
                    }
                else:
                    status_info = {
                        'status': 'partial_success',
                        'message': f"部分文件可用，成功检索到 {len(documents)} 个相关文档",
                        'available_files': available_files,
                        'requested_files': source_files,
                        'missing_files': [f for f in source_files if f not in available_files]
                    }
            else:
                status_info = {
                    'status': 'no_files_found',
                    'message': f"指定的文件 {source_files} 在数据库中不存在，请先进行数据注入",
                    'available_files': all_available_files,
                    'requested_files': source_files,
                    'missing_files': source_files
                }
        else:
            status_info = {
                'status': 'success',
                'message': f"成功检索到 {len(documents)} 个相关文档",
                'available_files': all_available_files,
                'requested_files': None,
                'missing_files': []
            }
        
        logging.info(f"Retrieved {len(documents)} similar documents.")
        return documents, status_info
        
    except Exception as e:
        logging.error(f"Error retrieving similar documents: {e}")
        return [], {
            'status': 'error',
            'message': f"检索过程中发生错误: {str(e)}",
            'available_files': [],
            'requested_files': source_files,
            'missing_files': []
        }

