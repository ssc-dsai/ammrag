import os
from dotenv import load_dotenv

# Load dotenv file
load_dotenv(".env")

# Embedder configuration
embedder_config = {
    "provider": "google",
    "config": {
        "api_key": os.environ.get("GEMINI_API_KEY"),
        "model": "models/text-embedding-004",
    },
}


# Qdrant configuration
qdrant_location = os.environ.get("QDRANT_LOCATION", "http://localhost:6333")
qdrant_api_key = os.environ.get("QDRANT_API_KEY") or None
qdrant_collection_name = "nss"


ollama_host = os.environ.get("OLLAMA_HOST", "http://localhost:11434/v1")
ollama_model = os.environ.get("OLLAMA_MODEL", "llama3.2:latest")

mmore_results_path = os.environ.get("MMORE_RESULTS_PATH", "./results")