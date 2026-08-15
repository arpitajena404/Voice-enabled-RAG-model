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
    
    # Directory paths
    INDEX_DIR = ROOT_DIR / "index"
    DATA_DIR = ROOT_DIR / "data"
    
    # Ensure directories exist
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

# Instantiate config
config = Config()
