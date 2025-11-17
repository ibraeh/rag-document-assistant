"""
Configuration management for RAG Document Assistant
"""
from pydantic_settings import BaseSettings
from pathlib import Path
from typing import Optional, Set


class Settings(BaseSettings):
    """Application settings loaded from environment variables or defaults."""

    # 🔐 OpenAI Configuration
    openai_api_key: Optional[str] = None

    # 🧠 Embedding Configuration
    use_local_embeddings: bool = True
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dimension: int = 384

    # 🗃️ Vector Store Configuration
    vector_db_path: str = "../data/vector_db"
    collection_name: str = "documents"

    # 📤 File Upload Configuration
    upload_dir: str = "../data/uploads"
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    allowed_extensions: Set[str] = {".pdf", ".docx", ".txt"}

    # 🤖 LLM Configuration
    llm_model: str = "gpt-5-mini"
    llm_temperature: float = 0.7
    max_tokens: int = 500

    # 📄 Document Processing
    chunk_size: int = 1000
    chunk_overlap: int = 200

    # 🔍 Retrieval Configuration
    top_k_results: int = 4
    similarity_threshold: float = 0.7

    # ⚙️ Application Settings
    app_name: str = "RAG Document Assistant"
    debug: bool = False

    class Config:
        env_file = ".env"
        case_sensitive = False


# Create settings instance
settings = Settings()

# Ensure required directories exist
for path in [settings.upload_dir, settings.vector_db_path]:
    Path(path).mkdir(parents=True, exist_ok=True)
