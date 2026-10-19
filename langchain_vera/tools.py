"""LangChain tools backed by Vera: read a page as Markdown, and search the web."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal

import httpx
from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.tools import BaseTool, ToolException
from langchain_core.utils.utils import secret_from_env
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, SecretStr, model_validator
from typing_extensions import Self

from langchain_vera._client import (
    API_KEY_ENV,
    DEFAULT_BASE_URL,
    VeraClient,
    VeraResponse,
)
from langchain_vera.errors import VeraRequestError


class VeraScrapeInput(BaseModel):
    """Input to `VeraScrapeTool`."""

    url: str = Field(
        description=(
            "The full http:// or https:// address of a public web page to read, "
            "for example https://example.com/article."
        )
    )


class VeraSearchInput(BaseModel):
    """Input to `VeraSearchTool`."""

    query: str = Field(
        description=(
            "What to search for, in plain words or with search operators "
            '(for example site:example.com or a "quoted phrase").'
        )
    )


class _VeraTool(BaseTool):
    """Shared set-up for the Vera tools: key, client and error policy."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    api_key: SecretStr | None = Field(default_factory=secret_from_env(API_KEY_ENV, default=None))
    """Vera API key. Defaults to the `VERA_API_KEY` environment variable."""

    base_url: str = DEFAULT_BASE_URL
    """The Vera API base URL."""

    timeout: float = 60.0
    """Seconds to wait for Vera to answer."""

    client: VeraClient | None = Field(default=None, exclude=True)
    """A ready `VeraClient` to use instead of building one from `api_key`."""

    handle_tool_error: bool | str | Callable[[ToolException], str] | None = True
    """A refused request (bad URL, blocked by robots.txt...) goes back to the model as text."""

    response_format: Literal["content", "content_and_artifact"] = "content_and_artifact"

    _vera: VeraClient = PrivateAttr()

    @model_validator(mode="after")
    def _build_client(self) -> Self:
        if self.client is not None:
            self._vera = self.client
        else:
            key = self.api_key.get_secret_value() if self.api_key else None
            self._vera = VeraClient(
                key,
                base_url=self.base_url,
                timeout=httpx.Timeout(self.timeout, connect=min(self.timeout, 10.0)),
            )
        return self


def _refused(error: VeraRequestError) -> ToolException:
    """A refused request is something the model can act on, so it is a `ToolException`."""
    return ToolException(error.message)


def _truncate(text: str, limit: int | None) -> str:
    if limit is None or len(text) <= limit:
        return text
    return f"{text[:limit]}\n\n[Truncated: showing the first {limit} of {len(text)} characters.]"


class VeraScrapeTool(_VeraTool):
    """Read a public web page and return its main content as Markdown.

    Costs 1 credit per page, or 5 with `render=True`. A refused or failed read is not charged.
    Sign up free at https://vera.waggle.ae/signup for 3,000 credits.

    Setup:
        ```bash
        pip install -U langchain-vera
        export VERA_API_KEY="vera_live_..."
        ```

    Instantiate:
        ```python
        from langchain_vera import VeraScrapeTool

        tool = VeraScrapeTool()  # or VeraScrapeTool(render=True, max_characters=20_000)
        ```

    Invoke:
        ```python
        markdown = tool.invoke({"url": "https://example.com"})
        ```

    Invoke with a `ToolCall` to also get the artifact (url, title, credits used, request id):
        ```python
        message = tool.invoke(
            {
                "type": "tool_call",
                "name": "vera_scrape",
                "id": "1",
                "args": {"url": "https://example.com"},
            }
        )
        message.content  # the Markdown
        message.artifact  # {"url": ..., "title": ..., "credits_used": 1, ...}
        ```
    """

    name: str = "vera_scrape"
    description: str = (
        "Read a public web page and return its main content as Markdown. "
        "Input is the page's full http(s) URL. Use it to read articles, documentation, "
        "product pages and other public pages. It does not search: to find pages, "
        "use a search tool first."
    )
    args_schema: type[BaseModel] = VeraScrapeInput

    render: bool = False
    """Load the page in a real browser first, for content built by JavaScript. 5 credits, not 1."""

    max_characters: int | None = Field(default=None, ge=1)
    """Cut the Markdown to this many characters, to protect the model's context window."""

    def _result(self, url: str, response: VeraResponse) -> tuple[str, dict[str, Any]]:
        data = response.data
        text = data.get("markdown") or data.get("text") or ""
        content = (
            _truncate(str(text), self.max_characters)
            if str(text).strip()
            else f"No readable content was found at {url}."
        )
        artifact: dict[str, Any] = {
            "url": data.get("url") or url,
            "title": data.get("title"),
            "status_code": data.get("status_code"),
            "credits_used": response.credits_used,
            "request_id": response.request_id,
            "elapsed_ms": response.elapsed_ms,
        }
        return content, artifact

    def _run(
        self,
        url: str,
        run_manager: CallbackManagerForToolRun | None = None,
    ) -> tuple[str, dict[str, Any]]:
        try:
            response = self._vera.scrape(url, render=self.render)
        except VeraRequestError as exc:
            raise _refused(exc) from exc
        return self._result(url, response)

    async def _arun(
        self,
        url: str,
        run_manager: AsyncCallbackManagerForToolRun | None = None,
    ) -> tuple[str, dict[str, Any]]:
        try:
            response = await self._vera.ascrape(url, render=self.render)
        except VeraRequestError as exc:
            raise _refused(exc) from exc
        return self._result(url, response)


class VeraSearchTool(_VeraTool):
    """Search the web, news or videos with Vera.

    Costs 3 credits per search, whatever `max_results` is (1 to 100). A failed search is not
    charged. Sign up free at https://vera.waggle.ae/signup for 3,000 credits.

    Setup:
        ```bash
        pip install -U langchain-vera
        export VERA_API_KEY="vera_live_..."
        ```

    Instantiate:
        ```python
        from langchain_vera import VeraSearchTool

        tool = VeraSearchTool(max_results=5)  # search_type="news" or "videos" also work
        ```

    Invoke:
        ```python
        tool.invoke({"query": "open source web scrapers"})
        ```

    Invoke with a `ToolCall` to also get the full result data as the artifact:
        ```python
        message = tool.invoke(
            {"type": "tool_call", "name": "vera_search", "id": "1", "args": {"query": "vera"}}
        )
        message.artifact["results"]  # list of result dicts
        ```
    """

    name: str = "vera_search"
    description: str = (
        "Search the web and return a numbered list of results, each with a title, URL and "
        "snippet. Input is a search query. Use it to find pages, then read them with a page "
        "reader tool."
    )
    args_schema: type[BaseModel] = VeraSearchInput

    search_type: Literal["web", "news", "videos"] = "web"
    """Which index to search."""

    max_results: int = Field(default=10, ge=1, le=100)
    """How many results to return (1 to 100). The cost is 3 credits whatever the number."""

    country: str | None = Field(default=None, pattern=r"^[a-z]{2}$")
    """Two-letter lowercase country code to localise results, such as `us` or `ae`."""

    language: str | None = Field(default=None, pattern=r"^[a-z]{2}$")
    """Two-letter lowercase language code for results, such as `en` or `ar`."""

    def _result(self, query: str, response: VeraResponse) -> tuple[str, dict[str, Any]]:
        data = response.data
        results = data.get("results")
        rows = results if isinstance(results, list) else []
        content = (
            format_results(rows, self.search_type) if rows else f"No results found for {query!r}."
        )
        artifact = {
            **data,
            "credits_used": response.credits_used,
            "request_id": response.request_id,
        }
        return content, artifact

    def _run(
        self,
        query: str,
        run_manager: CallbackManagerForToolRun | None = None,
    ) -> tuple[str, dict[str, Any]]:
        try:
            response = self._vera.search(
                query,
                kind=self.search_type,
                limit=self.max_results,
                country=self.country,
                language=self.language,
            )
        except VeraRequestError as exc:
            raise _refused(exc) from exc
        return self._result(query, response)

    async def _arun(
        self,
        query: str,
        run_manager: AsyncCallbackManagerForToolRun | None = None,
    ) -> tuple[str, dict[str, Any]]:
        try:
            response = await self._vera.asearch(
                query,
                kind=self.search_type,
                limit=self.max_results,
                country=self.country,
                language=self.language,
            )
        except VeraRequestError as exc:
            raise _refused(exc) from exc
        return self._result(query, response)


def format_results(rows: list[Any], search_type: str = "web") -> str:
    """Render search rows as a numbered, model-friendly list."""
    lines: list[str] = []
    for index, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            continue
        title = str(row.get("title") or row.get("url") or "Untitled")
        lines.append(f"{index}. {title}")
        if row.get("url"):
            lines.append(f"   {row['url']}")
        details = [
            str(row[key]) for key in ("source", "channel", "duration", "date") if row.get(key)
        ]
        if details:
            lines.append(f"   ({' | '.join(details)})")
        if row.get("snippet"):
            lines.append(f"   {row['snippet']}")
        elif search_type == "videos" and row.get("description"):
            lines.append(f"   {str(row['description'])[:300]}")
    return "\n".join(lines)


__all__ = [
    "VeraScrapeInput",
    "VeraScrapeTool",
    "VeraSearchInput",
    "VeraSearchTool",
    "format_results",
]
