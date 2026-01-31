import sys
import os

# Add the src directory to the Python path so we can import modules
sys.path.append(os.path.join(os.path.dirname(__file__)))

from embedding import get_embedding
from database import retrieve_similar_documents
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    filename='rag_project/logs/rag_app.log',
    filemode='a',
    encoding='utf-8'
)

def retrieve_documents(query, top_k=5, source_files=None):
    """
    检索与查询相关的文档
    
    Args:
        query: 查询文本
        top_k: 返回的最相似文档数量
        source_files: 可选的源文件列表，如果提供则只检索这些文件中的文档
    
    Returns:
        tuple: (文档内容列表, 检索状态信息)
    """
    logging.info(f"Retrieving documents for query: {query}")
    query_embedding = get_embedding(query)
    relevant_chunks, status_info = retrieve_similar_documents(query_embedding, top_k=top_k, source_files=source_files)
    logging.info(f"Retrieved {len(relevant_chunks)} relevant chunks.")
    
    # 记录检索状态
    if status_info['status'] == 'no_files_found':
        logging.warning(f"No files found: {status_info['message']}")
    elif status_info['status'] == 'partial_success':
        logging.info(f"Partial success: {status_info['message']}")
    
    return relevant_chunks, status_info
