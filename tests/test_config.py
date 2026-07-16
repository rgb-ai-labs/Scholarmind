from scholarmind.config import Settings, get_settings


def test_settings_load_with_defaults():
    settings = get_settings()
    assert isinstance(settings, Settings)
    assert settings.qdrant_path == "./data/qdrant"
    assert settings.embedding_model
    assert settings.reranker_model


def test_settings_override_via_env(monkeypatch):
    monkeypatch.setenv("QDRANT_PATH", "./custom/qdrant")
    monkeypatch.setenv("LLM_MODEL", "gpt-4o")
    settings = Settings()
    assert settings.qdrant_path == "./custom/qdrant"
    assert settings.llm_model == "gpt-4o"
