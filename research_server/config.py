import os
from pathlib import Path

class Config:
    """Central configuration for the research server."""
    
    # Base paths
    BASE_DIR = Path(__file__).parent
    DATA_DIR = BASE_DIR / "data"
    PDF_DIR = DATA_DIR / "pdfs"
    GROBID_JSON_DIR = DATA_DIR / "grobid_json"
    CACHE_DIR = DATA_DIR / "cache"
    
    # Create directories if they don't exist
    for dir_path in [DATA_DIR, PDF_DIR, GROBID_JSON_DIR, CACHE_DIR]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # Flask settings
    FLASK_HOST = os.getenv("FLASK_HOST", "0.0.0.0")
    FLASK_PORT = int(os.getenv("FLASK_PORT", 5000))
    DEBUG = os.getenv("DEBUG", "False").lower() == "true"
    
    # OpenAlex settings
    OPENALEX_API_URL = "https://api.openalex.org"
    OPENALEX_EMAIL = os.getenv("OPENALEX_EMAIL", "researcher@example.com")
    OPENALEX_MAX_RESULTS = 25
    OPENALEX_RATE_LIMIT_DELAY = 0.1  # seconds between requests
    
    # GROBID settings - using public cloud service (no Docker needed!)
    GROBID_URL = os.getenv("GROBID_URL", "https://kermitt2-grobid.hf.space")
    GROBID_TIMEOUT = 180  # seconds (cloud may be slower)
    
    # Ollama settings
    OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
    OLLAMA_MODEL = "llama3.1:8b"  # Using Llama 3.1 (available model)
    OLLAMA_TIMEOUT = 60
    
    # TF-IDF settings
    TFIDF_MAX_FEATURES = 10
    TFIDF_NGRAM_RANGE = (1, 2)
    
    # Retry settings
    MAX_RETRY_ATTEMPTS = 3
    RETRY_BACKOFF_FACTOR = 2
    
    # Cache settings
    CACHE_EXPIRY_SECONDS = 86400  # 24 hours
    
    # PDF settings
    PDF_MAX_SIZE_MB = 50
    PDF_TIMEOUT = 30
