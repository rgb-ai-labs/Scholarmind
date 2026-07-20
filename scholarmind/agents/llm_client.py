import base64
from collections.abc import Iterator
from pathlib import Path
from typing import Protocol

from openai import OpenAI

_IMAGE_MIME_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
}


class LLMClient(Protocol):
    def complete(self, system_prompt: str, user_prompt: str) -> str: ...


class OpenRouterClient:
    def __init__(self, api_key: str, base_url: str, model: str, max_tokens: int) -> None:
        self._client = OpenAI(api_key=api_key, base_url=base_url)
        self._model = model
        self._max_tokens = max_tokens

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=self._max_tokens,
        )
        return response.choices[0].message.content or ""

    def stream(self, system_prompt: str, user_prompt: str) -> Iterator[str]:
        # Yields text deltas as they arrive. Not part of the LLMClient Protocol — it's an opt-in
        # extension used only by the web app's Ask page (via qa.answer_question_streaming), so
        # plain FakeLLMClient test doubles elsewhere don't need it. The non-streaming complete()
        # above is unchanged and remains what the CLI, API, orchestrator, and every agent use.
        stream = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=self._max_tokens,
            stream=True,
        )
        for chunk in stream:
            choices = getattr(chunk, "choices", None)
            if not choices:
                continue
            content = getattr(choices[0].delta, "content", None)
            if content:
                yield content

    def complete_with_image(
        self, system_prompt: str, user_prompt: str, image_path: str, model: str
    ) -> str:
        # Not part of the LLMClient Protocol — this is an opt-in extension only used by the
        # Figure Q&A path, and only when settings.vision_model is set, so plain FakeLLMClient
        # test doubles elsewhere don't need to implement it. `model` is required (not
        # self._model) since the configured vision model is typically a different, multimodal
        # model from the one used for ordinary text generation.
        image_bytes = Path(image_path).read_bytes()
        mime_type = _IMAGE_MIME_TYPES.get(Path(image_path).suffix.lower(), "image/png")
        data_url = f"data:{mime_type};base64,{base64.b64encode(image_bytes).decode('ascii')}"

        response = self._client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_prompt},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                },
            ],
            max_tokens=self._max_tokens,
        )
        return response.choices[0].message.content or ""
