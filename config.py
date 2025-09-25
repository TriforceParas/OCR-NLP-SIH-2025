import os
from pathlib import Path

# Project root directory
ROOT_DIR = Path(__file__).parent.absolute()

# Directory paths
DATA_DIR = ROOT_DIR / "data"
UPLOADS_DIR = ROOT_DIR / "uploads"
LOGS_DIR = ROOT_DIR / "logs"

# JSON mapping files
ORG_MAP_FILE = DATA_DIR / "json" / "org_map.json"
DOCTYPE_MAP_FILE = DATA_DIR / "json" / "doctype_map.json"

# Create directories if they don't exist
for dir_path in [DATA_DIR, UPLOADS_DIR, LOGS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Flask configuration
FLASK_HOST = "0.0.0.0"
FLASK_PORT = 8000
MAX_CONTENT_LENGTH = 50 * 1024 * 1024  # 50MB

# NLP configuration
SHORT_SUMMARY_WORDS = 20
NORMAL_SUMMARY_SENTENCES = 5
NLP_MODEL = "facebook/bart-large-cnn"  # For summarization
CLASSIFICATION_CONFIDENCE_THRESHOLD = 0.3
SUPPORTED_LANGUAGES = ['en']  # Supported languages for processing
DEFAULT_LANGUAGE = 'en'  # Default language

# OCR configuration
OCR_DPI = 300
OCR_LANGUAGES = ['eng']
TESSERACT_CONFIG = '--oem 3 --psm 6'

# File processing
ALLOWED_EXTENSIONS = {'pdf', 'docx', 'doc', 'txt'}
DOWNLOAD_TIMEOUT = 30  # seconds