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
from pathlib import Path

dotenv.load_dotenv()


def load_config(config_path: str) -> dict:
    path = Path(config_path)
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data.get("config", {}) if isinstance(data, dict) else {}


class Settings(BaseSettings):
    """Application settings"""

    # App Info
    # app_name: str = "File Import Service"
    # app_version: str = "1.0.0"
    # app_description: str = "API for importing files via URIs and retrieving them"
    
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
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: Optional[str] = os.getenv("QDRANT_API_KEY", None)
    qdrant_collection_name: str = os.getenv("QDRANT_COLLECTION_NAME", "ammrag")


    # Embedder & Tokenizer Configuration
    embedder_model: str = os.getenv("EMBEDDER_MODEL", "all-MiniLM-L6-v2")

    # LLM configuration
    class LLMHost(str, Enum):
        OLLAMA = "ollama"
        OPENAPI = "openapi"
    llm_host : LLMHost = LLMHost(os.getenv("LLM_HOST", "ollama"))

    # Ollama Configuration
    ollama_host: str = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "gpt-oss:latest")
    ollama_vision_model: str = os.getenv("OLLAMA_VISION_MODEL", "llava")
    ollama_timeout: int = int(os.getenv("OLLAMA_TIMEOUT", 120))  # seconds
    ollama_vision_timeout: int = int(os.getenv("OLLAMA_VISION_TIMEOUT", 300))  # seconds

    # MCP Server Configuration
    mcp_port: int = 8001
    mcp_enabled: bool = True

    source_path: str = "http://localhost:8642/"

    # Temporary file download path (None = system default via tempfile.mkdtemp)
    temp_file_path: Optional[str] = os.getenv("TEMP_FILE_PATH", "./tmp")


    # Configuration file path
    config_path: str = os.getenv("CONFIG_PATH", "config/config.yml")


    class Config:
        env_file = ".env"
        case_sensitive = False
        extra = "ignore"  # Allow extra environment variables




# Create global settings instance
settings = Settings()
