from typing import Protocol

from openai import OpenAI


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
