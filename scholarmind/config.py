from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    qdrant_path: str = "./data/qdrant"
    qdrant_collection: str = "scholarmind_chunks"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    llm_api_key: str = ""
    llm_model: str = "gpt-4o-mini"
    chunk_size: int = 800
    chunk_overlap: int = 150
    retrieval_candidate_k: int = 20
    retrieval_top_k: int = 5


def get_settings() -> Settings:
    return Settings()
