"""A LangChain document loader that reads pages through Vera."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Iterator, Sequence
from typing import Any

import httpx
from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document
from pydantic import SecretStr

from langchain_vera._client import DEFAULT_BASE_URL, VeraClient, VeraResponse
from langchain_vera.errors import (
    VeraConflictError,
    VeraConnectionError,
    VeraRequestError,
    VeraServerError,
)

logger = logging.getLogger(__name__)

# Failures specific to one page. The rest (bad key, no credits, rate limit) would hit every page,
# so they always stop the load.
_SKIPPABLE = (VeraRequestError, VeraServerError, VeraConnectionError, VeraConflictError)


def to_document(url: str, response: VeraResponse) -> Document:
    """Build a `Document` from a scrape response.

    The page content is the Markdown. `metadata["source"]` is the URL you asked for, following the
    LangChain convention; the other keys are present only when Vera returned them.
    """
    data = response.data
    raw_meta = data.get("metadata")
    meta: dict[str, Any] = raw_meta if isinstance(raw_meta, dict) else {}
    text = data.get("markdown") or data.get("text") or ""
    candidates: dict[str, Any] = {
        "source": url,
        "title": data.get("title"),
        "description": meta.get("description"),
        "language": meta.get("language"),
        "canonical_url": data.get("canonical_url") or meta.get("canonical_url"),
        "final_url": data.get("url") if data.get("url") != url else None,
        "status_code": data.get("status_code"),
        "request_id": response.request_id,
        "credits_used": response.credits_used,
    }
    metadata = {key: value for key, value in candidates.items() if value not in (None, "")}
    return Document(page_content=str(text), metadata=metadata)


class VeraLoader(BaseLoader):
    """Load web pages as Markdown `Document`s through Vera.

    Each URL costs 1 credit (5 with `render=True`). A page that cannot be read is not charged.

    Example:
        ```python
        from langchain_vera import VeraLoader

        loader = VeraLoader(["https://example.com", "https://example.org"])
        docs = loader.load()
        docs[0].metadata["source"]  # "https://example.com"
        ```
    """

    def __init__(
        self,
        urls: str | Sequence[str],
        *,
        api_key: str | SecretStr | None = None,
        render: bool = False,
        continue_on_failure: bool = False,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = 60.0,
        client: VeraClient | None = None,
    ) -> None:
        """Create the loader.

        Args:
            urls: One URL or a list of URLs.
            api_key: A Vera API key. Defaults to the `VERA_API_KEY` environment variable.
            render: Load each page in a real browser first (5 credits a page instead of 1).
            continue_on_failure: Skip a page Vera cannot read (refused, unreachable, or a
                server error) and log a warning instead of raising. A bad key, no credits or a
                rate limit always raise, since every page would fail the same way.
            base_url: The Vera API base URL.
            timeout: Seconds to wait for each page.
            client: A ready `VeraClient`, used instead of building one.

        Raises:
            ValueError: If `urls` is empty.
            VeraConfigurationError: If no API key can be found.
        """
        self.urls: list[str] = [urls] if isinstance(urls, str) else list(urls)
        if not self.urls:
            msg = "urls must contain at least one URL."
            raise ValueError(msg)
        self.render = render
        self.continue_on_failure = continue_on_failure
        self._client = client or VeraClient(
            api_key,
            base_url=base_url,
            timeout=httpx.Timeout(timeout, connect=min(timeout, 10.0)),
        )

    def lazy_load(self) -> Iterator[Document]:
        """Yield one `Document` per URL, fetching as it goes."""
        for url in self.urls:
            try:
                response = self._client.scrape(url, render=self.render)
            except _SKIPPABLE as exc:
                if not self.continue_on_failure:
                    raise
                logger.warning("Skipping %s: %s", url, exc)
                continue
            yield to_document(url, response)

    async def alazy_load(self) -> AsyncIterator[Document]:
        """Async version of `lazy_load`."""
        for url in self.urls:
            try:
                response = await self._client.ascrape(url, render=self.render)
            except _SKIPPABLE as exc:
                if not self.continue_on_failure:
                    raise
                logger.warning("Skipping %s: %s", url, exc)
                continue
            yield to_document(url, response)
