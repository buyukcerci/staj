import os
from dotenv import load_dotenv

load_dotenv()

PDF_PATH = os.path.join(os.path.dirname(__file__), "anayasa.pdf")

CHROMA_DB_PATH = os.path.join(os.path.dirname(__file__), "chroma_db")
COLLECTION_NAME = "anayasa"

CHUNK_SIZE = 1024
CHUNK_OVERLAP = 200
SEPARATORS = ["\nMadde ", "\n\n", "\n", " ", ""]

EMBEDDING_MODEL = "BAAI/bge-m3"
LLM_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:14b")
TOP_K = 5
HYBRID_ALPHA = float(os.getenv("HYBRID_ALPHA", "0.5"))

OLLAMA_RETRY_MAX_ATTEMPTS = int(os.getenv("OLLAMA_RETRY_MAX_ATTEMPTS", "5"))
OLLAMA_RETRY_BACKOFF_FACTOR = float(os.getenv("OLLAMA_RETRY_BACKOFF_FACTOR", "2.0"))

LOG_FILE = "app.log"

SERVER_HOST = os.getenv("SERVER_HOST", "0.0.0.0")
SERVER_PORT = int(os.getenv("SERVER_PORT", "8000"))
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

SYSTEM_PROMPT = (
    "Sen Türkiye Cumhuriyeti Anayasası hakkında soruları cevaplayan bir asistansın.\n"
    "Aşağıdaki metni kullanarak soruyu cevapla. Eğer metin cevabı içermiyorsa bunu belirt.\n\n"
    "Her context bölümü, hangi madde veya maddelere ait olduğunu göstermek için\n"
    "[Madde X] ile başlar. Cevabını mutlaka madde numaralarını da vererek oluştur.\n\n"
    "Örnek format:\n"
    "- Soru: Cumhurbaşkanının görev süresi kaç yıldır?\n"
    "- Cevap: Cumhurbaşkanının görev süresi 5 yıldır (Anayasa Madde 101).\n\n"
    "Context:\n{context}"
)
