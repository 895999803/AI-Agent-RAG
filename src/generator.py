import sys
import os

# Add the src directory to the Python path so we can import modules
sys.path.append(os.path.join(os.path.dirname(__file__)))

from openai import OpenAI
from src.config import OPENROUTER_API_KEY, LLM_MODEL, TEMPERATURE
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    filename='rag_project/logs/rag_app.log',
    filemode='a',
    encoding='utf-8'
)

def generate_answer(question, context_chunks, context_metadata=None):
    if not OPENROUTER_API_KEY:
        logging.error("OPENROUTER_API_KEY is not set in config.py or .env")
        raise ValueError("OPENROUTER_API_KEY is not set.")

    # Combine context chunks with metadata if available
    if context_metadata:
        full_context_parts = []
        for i, chunk in enumerate(context_chunks):
            metadata_info = ""
            if i < len(context_metadata) and context_metadata[i]:
                metadata = context_metadata[i]
                source_file = metadata.get("source_file", "Unknown")
                chunk_index = metadata.get("chunk_index", i)
                metadata_info = f" [Source: {source_file}, Chunk: {chunk_index}]"
            
            full_context_parts.append(f"Chunk {i+1}:{metadata_info}\n{chunk}")
        
        full_context = "\n\n".join(full_context_parts)
    else:
        full_context = "\n\n".join(context_chunks)
    
    logging.info(f"Generating answer for question: {question}")
    logging.debug(f"Context used for generation: {full_context[:200]}...") # Log first 200 chars

    messages = [
        {"role": "system", "content": "You are a helpful assistant that answers questions based on the provided context only. If the answer is not in the context, state that you don't know. When referencing specific information, mention which source document it comes from if the metadata is available."},
        {"role": "user", "content": f"Context: {full_context}\n\nQuestion: {question}\nAnswer:"}
    ]

    try:
        client = OpenAI(
            api_key=OPENROUTER_API_KEY,
            base_url="https://openrouter.ai/api/v1"
        )
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=messages,
            temperature=TEMPERATURE
        )
        answer = response.choices[0].message.content
        logging.info("Answer generated successfully.")
        return answer
    except Exception as e:
        logging.error(f"Error generating answer: {e}")
        return "Error generating answer."
