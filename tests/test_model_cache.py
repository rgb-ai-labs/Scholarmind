from scholarmind.config import get_settings
from scholarmind.model_cache import get_embedder, get_reranker


def test_get_embedder_returns_the_same_instance_for_the_same_model_name():
    # Regression test: retrieval/search.py and ingestion/pipeline.py used to construct a fresh
    # Embedder on every call, reloading the sentence-transformers model from disk each time
    # (every ingest, every search, every Ask/Summarize/Writing panel render). The cache must
    # actually return the same object, not just an equal one, or the reload cost is unchanged.
    model_name = get_settings().embedding_model

    first = get_embedder(model_name)
    second = get_embedder(model_name)

    assert first is second


def test_get_reranker_returns_the_same_instance_for_the_same_model_name():
    model_name = get_settings().reranker_model

    first = get_reranker(model_name)
    second = get_reranker(model_name)

    assert first is second


def test_get_embedder_still_works_end_to_end():
    # The cache must not change behavior — just avoid reconstruction.
    embedder = get_embedder(get_settings().embedding_model)

    vector = embedder.embed_text("a sentence to embed")

    assert isinstance(vector, list)
    assert len(vector) == embedder.dimension
