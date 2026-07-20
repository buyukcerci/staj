import os

PDF_PATH = os.path.join(os.path.dirname(__file__), "anayasa.pdf")

CHROMA_DB_PATH = os.path.join(os.path.dirname(__file__), "chroma_db")
COLLECTION_NAME = "anayasa"

CHUNK_SIZE = 1024
CHUNK_OVERLAP = 200
SEPARATORS = ["\nMadde ", "\n\n", "\n", " ", ""]

EMBEDDING_MODEL = "BAAI/bge-m3"
LLM_MODEL = "qwen3:14b"
TOP_K = 5

LOG_FILE = "app.log"
