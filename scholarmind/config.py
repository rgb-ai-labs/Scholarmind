from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    qdrant_path: str = "./data/qdrant"
    qdrant_collection: str = "scholarmind_chunks"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    llm_api_key: str = ""
    llm_model: str = "google/gemma-4-26b-a4b-it:free"
    llm_base_url: str = "https://openrouter.ai/api/v1"
    chunk_size: int = 800
    chunk_overlap: int = 150
    retrieval_candidate_k: int = 20
    retrieval_top_k: int = 5
    retrieval_min_rerank_score: float = -7.0
    llm_max_tokens: int = 512
    s2_api_key: str = ""  # optional Semantic Scholar API key, raises the unauthenticated rate limit
    zotero_api_key: str = ""
    zotero_library_id: str = ""
    zotero_library_type: str = "user"  # "user" or "group"
    # Optional. When set, Figure Q&A sends the figure image (not just its caption) to this
    # vision-capable model via the same LLM_API_KEY/LLM_BASE_URL. Must be a multimodal model
    # (e.g. an OpenRouter Gemini vision model) — an ordinary text-only model will error or
    # ignore the image. Empty (default) means Figure Q&A always falls back to caption-only.
    vision_model: str = ""


def get_settings() -> Settings:
    return Settings()
