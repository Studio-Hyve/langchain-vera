"""Shared helpers: a VeraClient wired to an in-memory transport, so no test touches the network."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import httpx
import pytest

from langchain_vera import VeraClient

Handler = Callable[[httpx.Request], httpx.Response]


def success(
    data: dict[str, Any] | None = None,
    *,
    charged: int = 1,
    headers: dict[str, str] | None = None,
    **extra: Any,
) -> httpx.Response:
    """A 200 answer shaped like Vera's, with the credits header the real API sets."""
    body = {
        "request_id": "req_test",
        "data": data if data is not None else {},
        "credits_used": charged,
        "elapsed_ms": 123,
        **extra,
    }
    return httpx.Response(
        200,
        json=body,
        headers={
            "x-waggle-credits-used": str(charged),
            "x-waggle-request-id": "req_test",
            **(headers or {}),
        },
    )


def failure(
    status: int,
    message: str = "",
    *,
    code: str | None = None,
    headers: dict[str, str] | None = None,
) -> httpx.Response:
    """A non-2xx answer in Vera's `{"error": {...}}` shape."""
    error: dict[str, Any] = {"message": message}
    if code:
        error["code"] = code
    return httpx.Response(status, json={"error": error}, headers=headers or {})


@dataclass
class Recorder:
    """Captures what the client sent and answers with `respond`."""

    respond: Handler
    requests: list[httpx.Request] = field(default_factory=list)

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return self.respond(request)

    @property
    def last(self) -> httpx.Request:
        return self.requests[-1]

    @property
    def body(self) -> dict[str, Any]:
        return json.loads(self.last.content)  # type: ignore[no-any-return]


def make_client(
    respond: Handler | httpx.Response | None = None, **kwargs: Any
) -> tuple[VeraClient, Recorder]:
    """A VeraClient whose sync and async HTTP both go to `respond`."""
    if isinstance(respond, httpx.Response):
        fixed = respond
        respond = lambda _request: fixed  # noqa: E731
    recorder = Recorder(respond or (lambda _request: success({"markdown": "# Hi"})))
    transport = httpx.MockTransport(recorder)
    client = VeraClient(
        "vera_live_test_key",
        http_client=httpx.Client(transport=transport),
        async_http_client=httpx.AsyncClient(transport=transport),
        **kwargs,
    )
    return client, recorder


@pytest.fixture(autouse=True)
def _no_ambient_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tests must not depend on, or use, a real key from the developer's environment."""
    monkeypatch.delenv("VERA_API_KEY", raising=False)
