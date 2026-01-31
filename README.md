# DocQA RAG Project

This project implements a Retrieval-Augmented Generation (RAG) system for Document Question Answering (DocQA) using OpenAI embeddings, pgvector for vector storage and retrieval, and a Streamlit web interface.

## Project Structure

```
rag_project/
├── src/
│   ├── config.py             # Configuration variables
│   ├── database.py           # PostgreSQL and pgvector interaction
│   ├── embedding.py          # OpenAI Embedding generation
│   ├── document_processor.py # PDF parsing and text chunking
│   ├── retriever.py          # Document retrieval logic
│   ├── generator.py          # Answer generation with LLM
│   └── main.py               # Document ingestion and example Q&A
├── documents/                # Storage for PDF documents
│   └── document.pdf
├── app.py                    # Streamlit web application
├── requirements.txt          # Python dependencies
├── .env                      # Environment variables
├── README.md                 # Project README
└── logs/                     # Log files
    └── rag_app.log
```

## Setup and Installation

1.  **Clone the repository (or set up files manually).**
2.  **Create a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: .\venv\Scripts\activate
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **PostgreSQL and pgvector Setup:**
    *   Ensure you have PostgreSQL installed and running.
    *   Create a database (e.g., `rag_db`).
    *   Enable the `pgvector` extension in your database:
        ```sql
        CREATE EXTENSION vector;
        ```
5.  **Configure `.env`:**
    *   Rename `.env.example` to `.env` (if applicable) or directly edit the provided `.env` file.
    *   Fill in your `OPENAI_API_KEY` and PostgreSQL connection details.
    ```
    OPENAI_API_KEY="sk-YOUR_OPENAI_API_KEY"
    DB_USER="your_username"
    DB_PASSWORD="your_password"
    DB_HOST="localhost"
    DB_PORT="5432"
    DB_NAME="rag_db"
    COLLECTION_NAME="document_embeddings"
    ```

## Usage

### 1. Document Ingestion

Run the `main.py` script to process and embed your documents into the vector store:

```bash
python src/main.py --ingest
```

### 2. Ask Questions (CLI)

Run `main.py` without the `--ingest` flag to ask questions via the command line:

```bash
python src/main.py --query "What is this paper about?"
```

### 3. Web Interface (Streamlit)

To use the interactive web interface, run the `app.py` script:

```bash
streamlit run app.py
```

Open your web browser to the URL provided by Streamlit (usually `http://localhost:8501`).

## Logging

Application logs are written to `logs/rag_app.log`.
