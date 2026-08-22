import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file from the project root directory
ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env")

class Config:
    # API Keys
    SARVAM_API_KEY = os.getenv("SARVAM_API_KEY", "")
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
    GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
    
    # Model Configurations
    EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    
    # Retry / backoff settings (used by tenacity in stt.py, pipeline.py)
    RETRY_MAX_ATTEMPTS = int(os.getenv("RETRY_MAX_ATTEMPTS", "3"))
    RETRY_BACKOFF_BASE = float(os.getenv("RETRY_BACKOFF_BASE", "1.0"))
    RETRY_BACKOFF_MAX = float(os.getenv("RETRY_BACKOFF_MAX", "10.0"))
    
    # Directory paths
    INDEX_DIR = ROOT_DIR / "index"
    DATA_DIR = ROOT_DIR / "data"
    
    # Ensure directories exist
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    LIGHTWEIGHT_MODE = os.getenv("LIGHTWEIGHT_MODE", "false").lower() == "true"

# Instantiate config
config = Config()
