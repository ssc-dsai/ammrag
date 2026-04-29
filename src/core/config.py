"""
Configuration settings for the application
"""

from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Optional
from enum import Enum
import os
import logging
import dotenv
import yaml
from pathlib import Path

dotenv.load_dotenv()


class Settings(BaseSettings):
    """Application settings"""
    
    # Server
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = True
    
    # API
    api_prefix: str = ""
    app_name: str = "AMMRAG"
    app_version: str = "0.1.0"
    app_description: str = "Agentic Multi-Modal Retrieval Augmented Generation (AMMRAG)"
    
    # CORS
    allow_origins: list = ["*"]
    allow_credentials: bool = True
    allow_methods: list = ["*"]
    allow_headers: list = ["*"]
    
    # Database 
    database_url: Optional[str] = None

    # PostgreSQL Configuration
    postgres_host: str = os.getenv("POSTGRES_HOST", "localhost")
    postgres_port: int = int(os.getenv("POSTGRES_PORT", 5432))
    postgres_db: str = os.getenv("POSTGRES_DB", "ammrag")
    postgres_user: str = os.getenv("POSTGRES_USER", "ammrag_user")
    postgres_password: str = os.getenv("POSTGRES_PASSWORD", "password")

    @property
    def postgres_dsn(self) -> str:
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    # Qdrant Configuration
    qdrant_url: str = os.getenv("QDRANT_URL", "http://localhost:6333")
    qdrant_api_key: Optional[str] = os.getenv("QDRANT_API_KEY", None)
    qdrant_collection_name: str = os.getenv("QDRANT_COLLECTION_NAME", "ammrag")
    qdrant_vector_name: Optional[str] = os.getenv("QDRANT_VECTOR_NAME", None)
    qdrant_distance: str = os.getenv("QDRANT_DISTANCE", "cosine")
    qdrant_timeout: int = int(os.getenv("QDRANT_TIMEOUT", "30"))


    # Embedder Configuration
    dense_embedder_model: str = os.getenv("DENSE_EMBEDDER_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    dense_embedder_token_length: int = int(os.getenv("DENSE_EMBEDDER_TOKEN_LENGTH", 384))
    sparse_embedder_model: str = os.getenv("SPARSE_EMBEDDER_MODEL", "Qdrant/Splade_PP_en_v1")

    # LLM configuration
    class LLMHost(str, Enum):
        OLLAMA = "ollama"
        OPENAPI = "openapi"
    llm_host : LLMHost = LLMHost(os.getenv("LLM_HOST", "ollama"))

    # Ollama Configuration
    ollama_host: str = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "gemma4:e4b")
    ollama_timeout: int = int(os.getenv("OLLAMA_TIMEOUT", 120))  # seconds
    ollama_temperature: float = float(os.getenv("OLLAMA_TEMPERATURE", 0.3))
    ollama_retries: int = int(os.getenv("OLLAMA_RETRIES", 3))
    ollama_num_ctx: int = int(os.getenv("OLLAMA_NUM_CTX", 8192))

    # MCP Server Configuration
    mcp_port: int = 8001
    mcp_enabled: bool = True
    mcp_collection_name: Optional[str] = os.getenv("MCP_COLLECTION_NAME", None)

    # Temporary file download path (None = system default via tempfile.mkdtemp)
    temp_file_path: Optional[str] = os.getenv("TEMP_FILE_PATH", "./tmp")


    # Configuration file path
    config_path: Path = Path(os.getenv("CONFIG_PATH", "config/config.yml"))
    projects: list[dict] = []

    def __init__(self, **data):
        super().__init__(**data)
        self.projects = self.load_projects(self.config_path)

    def load_projects(self, config_path: Path) -> list[dict]:
        """Load project dicts from the YAML config file."""
        _log = logging.getLogger(__name__)
        if not config_path.exists():
            return []
        try:
            with config_path.open("r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
        except Exception as exc:
            _log.error("Failed to read config file '%s': %s", config_path, exc)
            return []
        config = data.get("config", {}) if isinstance(data, dict) else {}

        projects = []
        for i, entry in enumerate(config.get("projects", [])):
            name = entry.get("name")
            path = entry.get("path")
            if not name or not path:
                _log.warning("Skipping project entry #%d in config: missing 'name' or 'path' (%s)", i, entry)
                continue
            projects.append({"name": name, "path": path})
            _log.info("Loaded project '%s' from config", name)
        return projects


# Create global settings instance
settings = Settings()


