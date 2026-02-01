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

    if not OPENROUTER_API_KEY:
        logging.error("OPENROUTER_API_KEY is not set in config.py or .env")
        raise ValueError("OPENROUTER_API_KEY is not set.")

        full_context = "\n\n".join(context_chunks)
    logging.info(f"Generating answer for question: {question}")
    logging.debug(f"Context used for generation: {full_context[:200]}...") # Log first 200 chars

    messages = [
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
