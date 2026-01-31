import os
from pypdf import PdfReader
# from langchain.text_splitter import RecursiveCharacterTextSplitter # Commented out due to import issues
from src.config import CHUNK_SIZE, CHUNK_OVERLAP
import logging


class SimpleRecursiveCharacterTextSplitter:
    def __init__(self, chunk_size, chunk_overlap, length_function):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.length_function = length_function

    def split_text(self, text):
        # A very basic recursive split that prioritizes sentences, then words.
        # This is a simplified version of Langchain's RecursiveCharacterTextSplitter.
        # For production, consider using Langchain or a more robust library.

        # Split by paragraphs or double newlines first
        paragraphs = text.split('\n\n')
        chunks = []

        for para in paragraphs:
            current_chunk = ""
            words = para.split(' ')
            for word in words:
                if self.length_function(current_chunk + " " + word) <= self.chunk_size:
                    if current_chunk:
                        current_chunk += " "
                    current_chunk += word
                else:
                    if current_chunk:
                        chunks.append(current_chunk)
                    current_chunk = word # Start new chunk with current word
            if current_chunk:
                chunks.append(current_chunk)

        # Apply overlap (simple: just add previous chunk's end to next chunk's start)
        # This basic implementation doesn't handle overlap perfectly like Langchain's.
        # It's more of a conceptual demonstration.
        final_chunks = []
        for i in range(len(chunks)):
            if i > 0 and self.chunk_overlap > 0:
                overlap_text = chunks[i-1][-self.chunk_overlap:] if len(chunks[i-1]) >= self.chunk_overlap else chunks[i-1]
                final_chunks.append(overlap_text + chunks[i])
            else:
                final_chunks.append(chunks[i])

        return final_chunks

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    filename='rag_project/logs/rag_app.log',
    filemode='a',
    encoding='utf-8'
)


def extract_text_from_pdf(pdf_path):
    text = ""
    try:
        with open(pdf_path, "rb") as file:
            reader = PdfReader(file)
            for page_num in range(len(reader.pages)):
                page_text = reader.pages[page_num].extract_text()
                if page_text:
                    text += page_text
        # Clean the text to remove problematic characters
        # Remove NUL characters and other control characters
        cleaned_text = ''.join(char for char in text if ord(char) >= 32 or char in '\t\n\r')
        logging.info(f"Extracted text from {pdf_path}.")
        return cleaned_text
    except Exception as e:
        logging.error(f"Error extracting text from PDF {pdf_path}: {e}")
        raise


def chunk_text(text):
    text_splitter = SimpleRecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
    )
    chunks = text_splitter.split_text(text)
    logging.info(f"Split text into {len(chunks)} chunks.")
    return chunks


def process_document(pdf_path):
    logging.info(f"Processing document: {pdf_path}")
    raw_text = extract_text_from_pdf(pdf_path)
    chunks = chunk_text(raw_text)
    logging.info(f"Document processing complete. {len(chunks)} chunks generated.")
    
    # Add source file metadata to each chunk
    source_file = os.path.basename(pdf_path)
    chunks_with_metadata = [(chunk, source_file) for chunk in chunks]
    
    return chunks_with_metadata


def process_all_documents(documents_dir):
    """
    Process all PDF files in the documents directory.
    Returns a list of tuples: (chunk_text, source_filename)
    """
    all_chunks_with_metadata = []
    documents_path = os.path.join(os.path.dirname(__file__), '..', documents_dir)
    
    if not os.path.exists(documents_path):
        logging.error(f"Documents directory not found: {documents_path}")
        raise FileNotFoundError(f"Documents directory not found: {documents_path}")
    
    pdf_files = [f for f in os.listdir(documents_path) if f.endswith('.pdf')]
    
    if not pdf_files:
        logging.warning(f"No PDF files found in {documents_path}")
        return []
    
    logging.info(f"Found {len(pdf_files)} PDF files to process: {pdf_files}")
    
    for pdf_file in pdf_files:
        pdf_path = os.path.join(documents_path, pdf_file)
        try:
            chunks_with_metadata = process_document(pdf_path)
            # Add metadata (source filename) to each chunk
            for chunk, source_file in chunks_with_metadata:
                all_chunks_with_metadata.append((chunk, source_file))
        except Exception as e:
            logging.error(f"Failed to process {pdf_file}: {e}")
            continue
    
    logging.info(f"Successfully processed {len(all_chunks_with_metadata)} chunks from {len(pdf_files)} documents.")
    return all_chunks_with_metadata

