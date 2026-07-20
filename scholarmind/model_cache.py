from functools import cache

from scholarmind.ingestion.embedder import Embedder
from scholarmind.retrieval.reranker import Reranker


@cache
def get_embedder(model_name: str) -> Embedder:
    # Loading a sentence-transformers model from disk is expensive; every ingest and every
    # search() call used to construct a fresh Embedder, reloading the same model repeatedly
    # within one process (CLI run, API server, or Streamlit session). Cached per model name —
    # in practice there's only ever one configured EMBEDDING_MODEL per process.
    return Embedder(model_name)


@cache
def get_reranker(model_name: str) -> Reranker:
    return Reranker(model_name)
