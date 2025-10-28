"""Teacher model client that wraps API interactions with retries and logging."""

from __future__ import annotations

import os
from typing import Any, Dict, Iterable, Optional

import httpx
from tenacity import Retrying, retry_if_exception_type, stop_after_attempt, wait_exponential

from .config import ModelEndpointConfig
from .utils import deep_merge_dict


class TeacherClientError(RuntimeError):
    """Raised when the teacher API call fails."""


def _truncate(value: str, limit: int = 1000) -> str:
    if len(value) <= limit:
        return value
    return value[:limit] + "...[truncated]"


class TeacherClient:
    """Thin wrapper around a chat-completions style API."""

    def __init__(self, config: ModelEndpointConfig):
        self.config = config
        api_key = os.getenv(config.api_key_env)
        if not api_key:
            raise EnvironmentError(
                f"Missing teacher API key in environment variable '{config.api_key_env}'."
            )

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        headers.update(config.extra_headers or {})

        self._client = httpx.Client(
            base_url=config.base_url or self._default_base_url(),
            headers=headers,
            timeout=config.request_timeout,
        )

    @staticmethod
    def _default_base_url() -> str:
        return "https://api.openai.com/v1"

    def close(self) -> None:
        self._client.close()

    def generate(
        self,
        *,
        messages: Iterable[Dict[str, Any]],
        tools: Optional[Iterable[Dict[str, Any]]] = None,
        tool_choice: Optional[str] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        max_output_tokens: Optional[int] = None,
        response_format: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Call the teacher model and return the JSON response."""

        attempt_limit = max(1, self.config.retry_attempts)
        retryer = Retrying(
            retry=retry_if_exception_type(TeacherClientError),
            wait=wait_exponential(multiplier=1.5, min=1, max=15),
            stop=stop_after_attempt(attempt_limit),
            reraise=True,
        )

        def _do_call() -> Dict[str, Any]:
            payload: Dict[str, Any] = {
                "model": self.config.model,
                "messages": list(messages),
                "temperature": temperature
                if temperature is not None
                else self.config.temperature,
                "top_p": top_p if top_p is not None else self.config.top_p,
                "max_tokens": max_output_tokens
                if max_output_tokens is not None
                else self.config.max_output_tokens,
            }
            if tools:
                payload["tools"] = list(tools)
            if tool_choice:
                payload["tool_choice"] = tool_choice
            if response_format:
                payload["response_format"] = response_format
            if self.config.request_overrides:
                payload = deep_merge_dict(payload, self.config.request_overrides)

            path = self.config.completion_path
            if not path.startswith("/"):
                path = f"/{path}"

            try:
                response = self._client.post(path, json=payload)
            except httpx.HTTPError as exc:
                raise TeacherClientError(
                    f"[{self.config.provider}:{self.config.model}] transport error: {exc}"
                ) from exc

            if response.status_code >= 400:
                request_id = (
                    response.headers.get("x-request-id")
                    or response.headers.get("x-requestid")
                    or response.headers.get("request-id")
                    or "unknown"
                )
                raise TeacherClientError(
                    f"[{self.config.provider}:{self.config.model}] HTTP {response.status_code} "
                    f"(request_id={request_id}, payload_keys={sorted(payload.keys())}) "
                    f"body={_truncate(response.text)}"
                )

            try:
                data = response.json()
            except ValueError as exc:
                raise TeacherClientError(
                    f"[{self.config.provider}:{self.config.model}] Invalid JSON response: "
                    f"{_truncate(response.text)}"
                ) from exc

            if "choices" not in data or not data["choices"]:
                raise TeacherClientError(
                    f"[{self.config.provider}:{self.config.model}] API returned no choices."
                )
            return data

        return retryer.call(_do_call)

    def __enter__(self) -> "TeacherClient":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
