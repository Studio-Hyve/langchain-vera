"""VeraLoader, with HTTP mocked."""

from __future__ import annotations

import logging
from typing import Any

import httpx
import pytest
from langchain_core.documents import Document

from langchain_vera import (
    VeraAuthenticationError,
    VeraConfigurationError,
    VeraLoader,
    VeraRequestError,
)
from tests.unit_tests.conftest import failure, make_client, success

PAGE: dict[str, Any] = {
    "url": "https://example.com/",
    "title": "Example Domain",
    "markdown": "# Example Domain\n\nFor illustrative examples.",
    "canonical_url": "https://example.com/",
    "metadata": {"description": "An example page.", "language": "en"},
    "status_code": 200,
}


def test_loads_a_document_with_source_metadata() -> None:
    client, recorder = make_client(success(PAGE))
    docs = VeraLoader("https://example.com", client=client).load()
    assert docs == [
        Document(
            page_content=PAGE["markdown"],
            metadata={
                "source": "https://example.com",
                "title": "Example Domain",
                "description": "An example page.",
                "language": "en",
                "canonical_url": "https://example.com/",
                "final_url": "https://example.com/",
                "status_code": 200,
                "request_id": "req_test",
                "credits_used": 1,
            },
        )
    ]
    assert recorder.body == {"url": "https://example.com", "response_format": "markdown"}


def test_source_is_the_requested_url_and_final_url_is_kept_when_it_differs() -> None:
    client, _ = make_client(success({**PAGE, "url": "https://www.example.com/landing"}))
    (doc,) = VeraLoader("https://example.com/old", client=client).load()
    assert doc.metadata["source"] == "https://example.com/old"
    assert doc.metadata["final_url"] == "https://www.example.com/landing"


def test_only_present_metadata_keys_are_set() -> None:
    client, _ = make_client(success({"markdown": "hello"}, charged=1))
    (doc,) = VeraLoader("https://example.com", client=client).load()
    assert doc.metadata == {
        "source": "https://example.com",
        "request_id": "req_test",
        "credits_used": 1,
    }


def test_falls_back_to_text() -> None:
    client, _ = make_client(success({"text": "plain"}))
    (doc,) = VeraLoader("https://example.com", client=client).load()
    assert doc.page_content == "plain"


def test_several_urls_in_order_and_lazy() -> None:
    client, recorder = make_client(lambda request: success({"markdown": request.url.path}))
    urls = ["https://a.example.com", "https://b.example.com"]
    iterator = VeraLoader(urls, client=client).lazy_load()
    assert recorder.requests == []  # nothing is fetched until iterated
    docs = list(iterator)
    assert [d.metadata["source"] for d in docs] == urls
    assert len(recorder.requests) == 2


def test_render_is_passed_and_costs_show_in_metadata() -> None:
    client, recorder = make_client(success(PAGE, charged=5))
    (doc,) = VeraLoader("https://example.com", render=True, client=client).load()
    assert recorder.body["render"] is True
    assert doc.metadata["credits_used"] == 5


def test_needs_urls_and_a_key() -> None:
    with pytest.raises(ValueError, match="at least one URL"):
        VeraLoader([])
    with pytest.raises(VeraConfigurationError, match="VERA_API_KEY"):
        VeraLoader("https://example.com")


def test_builds_its_own_client_from_the_key() -> None:
    loader = VeraLoader("https://example.com", api_key="vera_live_test", timeout=5)
    assert loader._client.base_url == "https://vera.waggle.ae/api/vera/v1"


def test_a_refused_page_raises_by_default() -> None:
    client, _ = make_client(
        failure(422, "This page is disallowed by robots.txt.", code="robots_disallowed")
    )
    with pytest.raises(VeraRequestError):
        VeraLoader("https://example.com", client=client).load()


def test_continue_on_failure_skips_a_page_vera_cannot_read(
    caplog: pytest.LogCaptureFixture,
) -> None:
    def respond(request: httpx.Request) -> httpx.Response:
        return (
            failure(422, "Refused.")
            if "bad" in str(request.url) or b"bad" in request.content
            else success(PAGE)
        )

    client, _ = make_client(respond)
    loader = VeraLoader(
        ["https://good.example.com", "https://bad.example.com", "https://good2.example.com"],
        client=client,
        continue_on_failure=True,
    )
    with caplog.at_level(logging.WARNING, logger="langchain_vera.document_loaders"):
        docs = loader.load()
    assert [d.metadata["source"] for d in docs] == [
        "https://good.example.com",
        "https://good2.example.com",
    ]
    assert "Skipping https://bad.example.com" in caplog.text


def test_continue_on_failure_never_hides_a_bad_key() -> None:
    client, _ = make_client(failure(401, "API key is invalid, expired or revoked."))
    loader = VeraLoader(
        ["https://a.example.com", "https://b.example.com"], client=client, continue_on_failure=True
    )
    with pytest.raises(VeraAuthenticationError):
        loader.load()


async def test_async_load() -> None:
    client, recorder = make_client(success(PAGE))
    docs = await VeraLoader(
        ["https://a.example.com", "https://b.example.com"], client=client
    ).aload()
    assert len(docs) == 2
    assert len(recorder.requests) == 2


async def test_async_continue_on_failure() -> None:
    client, _ = make_client(failure(422, "Refused."))
    docs = await VeraLoader("https://example.com", client=client, continue_on_failure=True).aload()
    assert docs == []
    with pytest.raises(VeraRequestError):
        await VeraLoader("https://example.com", client=client).aload()
