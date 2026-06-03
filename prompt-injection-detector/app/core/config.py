from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # LLM providers
    groq_api_key: str
    groq_model: str = "llama-3.1-8b-instant"

    google_api_key: str
    gemini_model: str = "gemini-2.0-flash"

    # Embeddings (local)
    embedding_model: str = "all-MiniLM-L6-v2"

    # ChromaDB
    chroma_persist_dir: str = "./db"

    # Detection config
    injection_threshold: float = 0.5
    examples_top_k: int = 3
    owasp_top_k: int = 1


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
