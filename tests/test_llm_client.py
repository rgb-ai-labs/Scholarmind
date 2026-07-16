import pytest

from scholarmind.agents.llm_client import OpenRouterClient
from scholarmind.config import get_settings

_has_llm_key = bool(get_settings().llm_api_key)


@pytest.mark.skipif(not _has_llm_key, reason="LLM_API_KEY not configured")
def test_openrouter_client_real_call_returns_pong():
    settings = get_settings()
    client = OpenRouterClient(
        api_key=settings.llm_api_key,
        base_url=settings.llm_base_url,
        model=settings.llm_model,
        max_tokens=settings.llm_max_tokens,
    )

    result = client.complete("Reply with exactly one word.", "Say the word: pong")

    assert result != ""
    assert "pong" in result.lower()


def test_openrouter_client_normalizes_none_content_to_empty_string():
    settings = get_settings()
    client = OpenRouterClient(
        api_key=settings.llm_api_key,
        base_url=settings.llm_base_url,
        model=settings.llm_model,
        max_tokens=settings.llm_max_tokens,
    )

    class _FakeMessage:
        content = None

    class _FakeChoice:
        message = _FakeMessage()

    class _FakeResponse:
        choices = [_FakeChoice()]

    class _FakeCompletions:
        def create(self, **kwargs):
            return _FakeResponse()

    class _FakeChat:
        completions = _FakeCompletions()

    class _FakeClient:
        chat = _FakeChat()

    client._client = _FakeClient()

    result = client.complete("system", "user")

    assert result == ""
