"""Non-blocking adapter for the existing OpenAI-compatible llama-server."""

from __future__ import annotations

import os
import requests


class LocalLLM:
    def __init__(self, base_url: str | None = None):
        self.base_url = (base_url or os.environ.get("TUTOR_SERVER_URL", "http://127.0.0.1:8080")).rstrip("/")

    def model(self) -> str:
        response = requests.get(f"{self.base_url}/v1/models", timeout=3)
        response.raise_for_status()
        models = response.json().get("data", [])
        if not models:
            raise RuntimeError("llama-server is reachable but has no loaded model.")
        return models[0]["id"]

    def reply(self, messages: list[dict[str, str]], memories: list[dict] | None = None) -> str:
        model = self.model()
        context = ""
        if memories:
            context = "\nRelevant prior conversation details (use only when relevant):\n" + "\n".join(
                f"- Learner: {item['user_text'][:500]} Tutor: {item['assistant_text'][:500]}" for item in memories
            )
        prompt = [dict(messages[0], content=messages[0]["content"] + context), *messages[1:]]
        response = requests.post(
            f"{self.base_url}/v1/chat/completions",
            json={"model": model, "messages": prompt, "temperature": 0.4, "max_tokens": 180},
            timeout=(3, 180),
        )
        response.raise_for_status()
        text = response.json()["choices"][0]["message"]["content"].strip()
        if not text:
            raise RuntimeError("llama-server returned an empty response.")
        return text

