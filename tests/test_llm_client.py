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
    # A dummy key is enough: the real OpenAI client built here is immediately replaced
    # by the fake below, so this test needs no configured key and makes no network call.
    client = OpenRouterClient(
        api_key="test-key",
        base_url="https://example.invalid/v1",
        model="test-model",
        max_tokens=16,
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


def test_complete_with_image_sends_image_content_block_and_override_model(tmp_path):
    client = OpenRouterClient(
        api_key="test-key",
        base_url="https://example.invalid/v1",
        model="text-only-model",
        max_tokens=16,
    )

    image_path = tmp_path / "figure.png"
    image_path.write_bytes(b"\x89PNG\r\n\x1a\nfake png bytes")

    captured = {}

    class _FakeMessage:
        content = "It shows a bar chart."

    class _FakeChoice:
        message = _FakeMessage()

    class _FakeResponse:
        choices = [_FakeChoice()]

    class _FakeCompletions:
        def create(self, **kwargs):
            captured.update(kwargs)
            return _FakeResponse()

    class _FakeChat:
        completions = _FakeCompletions()

    class _FakeClient:
        chat = _FakeChat()

    client._client = _FakeClient()

    result = client.complete_with_image(
        "system prompt", "what does this show?", str(image_path), "vision-model"
    )

    assert result == "It shows a bar chart."
    assert captured["model"] == "vision-model"  # overrides the client's text-only model

    user_message = captured["messages"][1]
    assert user_message["role"] == "user"
    content_blocks = user_message["content"]
    assert content_blocks[0] == {"type": "text", "text": "what does this show?"}
    assert content_blocks[1]["type"] == "image_url"
    assert content_blocks[1]["image_url"]["url"].startswith("data:image/png;base64,")


def test_complete_with_image_picks_mime_type_from_extension(tmp_path):
    client = OpenRouterClient(
        api_key="test-key", base_url="https://example.invalid/v1", model="m", max_tokens=16
    )

    image_path = tmp_path / "figure.jpg"
    image_path.write_bytes(b"fake jpeg bytes")

    captured = {}

    class _FakeMessage:
        content = "ok"

    class _FakeChoice:
        message = _FakeMessage()

    class _FakeResponse:
        choices = [_FakeChoice()]

    class _FakeCompletions:
        def create(self, **kwargs):
            captured.update(kwargs)
            return _FakeResponse()

    class _FakeChat:
        completions = _FakeCompletions()

    class _FakeClient:
        chat = _FakeChat()

    client._client = _FakeClient()

    client.complete_with_image("s", "u", str(image_path), "vision-model")

    url = captured["messages"][1]["content"][1]["image_url"]["url"]
    assert url.startswith("data:image/jpeg;base64,")
