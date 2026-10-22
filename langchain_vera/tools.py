"""LangChain tools backed by Vera: read a page, search the web, and search intelligence."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any, Literal

import httpx
from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.tools import BaseTool, ToolException
from langchain_core.utils.utils import secret_from_env
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    SecretStr,
    field_validator,
    model_validator,
)
from typing_extensions import Self

from langchain_vera._client import (
    AI_ANSWERS_TIMEOUT,
    AI_ENGINES,
    API_KEY_ENV,
    COMPANY_FOOTPRINT_TIMEOUT,
    DEFAULT_BASE_URL,
    GEO_RADAR_TIMEOUT,
    KEYWORD_IDEAS_TIMEOUT,
    SERP_DEFAULT_FEATURES,
    SERP_TIMEOUT,
    AIEngine,
    KeywordExpansion,
    SerpEngine,
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
    Sign up at https://vera.waggle.ae/signup for free credits.

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
    charged. Sign up at https://vera.waggle.ae/signup for free credits.

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


_ENGINE_NAMES = {
    "chatgpt": "ChatGPT",
    "claude": "Claude",
    "gemini": "Gemini",
    "perplexity": "Perplexity",
}
_INTENT_ORDER = ("question", "comparison", "commercial", "local", "informational")
_DEFAULT_ENGINES: tuple[AIEngine, ...] = ("chatgpt",)
_COUNTRY = r"^[a-z]{2}$"
_LANGUAGE = r"^[a-z]{2,3}(-[a-z0-9]{2,8})*$"


def _json_text(data: dict[str, Any]) -> str:
    return json.dumps(data, ensure_ascii=False, indent=1, default=str)


def _artifact(response: VeraResponse) -> dict[str, Any]:
    """The whole answer plus what it cost: what a ToolMessage carries beside the text."""
    return {
        **response.data,
        "credits_used": response.credits_used,
        "request_id": response.request_id,
    }


class _VeraTextTool(_VeraTool):
    """Shared set-up for tools whose answer is a brief of text for the model."""

    max_characters: int | None = Field(default=None, ge=1)
    """Cut the text to this many characters, to protect the model's context window."""

    def _brief(self, data: dict[str, Any], empty: str) -> str:
        """The service's own `llm_markdown` brief, cut to `max_characters`; JSON if it sent none."""
        brief = data.get("llm_markdown")
        text = brief.strip() if isinstance(brief, str) else ""
        return _truncate(text or _json_text(data), self.max_characters) if data else empty


class VeraSerpInput(BaseModel):
    """Input to `VeraSerpTool`."""

    query: str = Field(
        description=(
            "What to search for, in plain words or with search operators "
            '(for example site:example.com or a "quoted phrase").'
        )
    )


class VeraSerpTool(_VeraTextTool):
    """Read a search results page on Google, Bing, DuckDuckGo, Brave or Yahoo.

    Returns a Markdown brief: organic results (rank, title, URL, snippet, tagged forum, video or
    shop) and, on Google, the AI Overview with its citations and People Also Ask, plus the map
    listings (`local_pack=True`) and shopping results (`shopping=True`) when asked for. Priced by
    what comes back: the most it can cost is held first and you are charged for what is served. A
    live results page of up to ten results is typically 3 credits, 1 from the cache; every further
    ten results is another page. A failed search is not charged.
    Sign up at https://vera.waggle.ae/signup for free credits.

    Setup:
        ```bash
        pip install -U langchain-vera
        export VERA_API_KEY="vera_live_..."
        ```

    Instantiate:
        ```python
        from langchain_vera import VeraSerpTool

        tool = VeraSerpTool(gl="ae", hl="en", depth=10)
        # or VeraSerpTool(location="Dubai", local_pack=True) for the map listings
        ```

    Invoke:
        ```python
        tool.invoke({"query": "best coffee in dubai"})
        ```

    Invoke with a `ToolCall` to also get the full results page as the artifact:
        ```python
        message = tool.invoke(
            {"type": "tool_call", "name": "vera_serp", "id": "1", "args": {"query": "coffee"}}
        )
        message.artifact["organic"]  # ranked results
        message.artifact["credits_used"]  # what this call was charged
        ```
    """

    name: str = "vera_serp"
    description: str = (
        "Run one search and return the results page as a Markdown brief: organic results with "
        "rank, title, URL and snippet and, on Google, the AI Overview with its citations and "
        "People Also Ask. Input is the search query. Use it for rank checks, competitor and SERP "
        "research, and local or shopping lookups. For a quick list of links, use a plain web "
        "search tool instead."
    )
    args_schema: type[BaseModel] = VeraSerpInput
    timeout: float = SERP_TIMEOUT

    engine: SerpEngine = "google"
    """The search engine: `google`, `bing`, `duckduckgo`, `brave` or `yahoo`."""

    gl: str | None = Field(default=None, pattern=_COUNTRY)
    """Two-letter lowercase country code to localise results, such as `us` or `ae`."""

    hl: str | None = Field(default=None, pattern=_LANGUAGE)
    """Language code for the results page, such as `en`, `ar` or `pt-br`."""

    location: str | None = None
    """A city or region to search from, such as `Dubai`. Needed for a local query."""

    depth: int = Field(default=10, ge=1, le=100)
    """How many results to read (1 to 100). Each ten is one results page, so more costs more."""

    local_pack: bool = False
    """Add the Google Maps listings (name, rating, address, phone, website).

    Google only. Priced on its own, once per request, and only when it returns something.
    """

    shopping: bool = False
    """Add the Google Shopping carousel. Google only; priced on its own."""

    fresh: bool = False
    """Search live instead of reusing a result from the last 24 hours. Costs the full price."""

    @model_validator(mode="after")
    def _add_ons_are_google_only(self) -> Self:
        if self.engine != "google" and (self.local_pack or self.shopping):
            msg = "local_pack and shopping are Google add-ons: use engine='google'."
            raise ValueError(msg)
        return self

    def _features(self) -> list[str] | None:
        if not (self.local_pack or self.shopping):
            return None
        add_ons = [
            name
            for name, on in (("local_pack", self.local_pack), ("shopping", self.shopping))
            if on
        ]
        return [*SERP_DEFAULT_FEATURES, *add_ons]

    def _result(self, query: str, response: VeraResponse) -> tuple[str, dict[str, Any]]:
        data = response.data
        brief, organic = data.get("llm_markdown"), data.get("organic")
        if isinstance(brief, str) and brief.strip():
            content = _truncate(brief.strip(), self.max_characters)
        elif isinstance(organic, list) and organic:
            content = _truncate(format_results(organic), self.max_characters)
        else:
            content = f"No results found for {query!r}."
        return content, _artifact(response)

    def _run(
        self,
        query: str,
        run_manager: CallbackManagerForToolRun | None = None,
    ) -> tuple[str, dict[str, Any]]:
        try:
            response = self._vera.serp(
                query,
                engine=self.engine,
                gl=self.gl,
                hl=self.hl,
                location=self.location,
                depth=self.depth,
                features=self._features(),
                fresh=self.fresh,
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
            response = await self._vera.aserp(
                query,
                engine=self.engine,
                gl=self.gl,
                hl=self.hl,
                location=self.location,
                depth=self.depth,
                features=self._features(),
                fresh=self.fresh,
            )
        except VeraRequestError as exc:
            raise _refused(exc) from exc
        return self._result(query, response)


class VeraKeywordIdeasInput(BaseModel):
    """Input to `VeraKeywordIdeasTool`."""

    seed: str = Field(
        description=(
            "The keyword to find ideas for, for example 'specialty coffee'. Arabic works too."
        )
    )


class VeraKeywordIdeasTool(_VeraTextTool):
    """Find keyword ideas for a seed keyword.

    Ideas come from Google autocomplete, People Also Ask and related searches, cleaned and
    de-duplicated, each labelled by intent (question, comparison, commercial, local or
    informational). Ideas only: no search volumes. Priced by what comes back (`expand` sets how many
    autocomplete calls are made; more calls cost more). A failed call is not charged.
    Sign up at https://vera.waggle.ae/signup for free credits.

    Setup:
        ```bash
        pip install -U langchain-vera
        export VERA_API_KEY="vera_live_..."
        ```

    Instantiate:
        ```python
        from langchain_vera import VeraKeywordIdeasTool

        tool = VeraKeywordIdeasTool(gl="ae", hl="en", limit=50, expand="questions")
        ```

    Invoke:
        ```python
        tool.invoke({"seed": "specialty coffee"})
        ```

    Invoke with a `ToolCall` to also get every idea as the artifact:
        ```python
        message = tool.invoke(
            {
                "type": "tool_call",
                "name": "vera_keyword_ideas",
                "id": "1",
                "args": {"seed": "specialty coffee"},
            }
        )
        message.artifact["ideas"]  # [{"keyword": ..., "sources": [...], "intent": ...}]
        ```
    """

    name: str = "vera_keyword_ideas"
    description: str = (
        "Find keyword ideas for a seed keyword from Google autocomplete, People Also Ask and "
        "related searches, grouped by intent (question, comparison, commercial, local, "
        "informational). Input is the seed keyword. Use it for keyword research and content "
        "planning. It gives ideas only, not search volumes."
    )
    args_schema: type[BaseModel] = VeraKeywordIdeasInput
    timeout: float = KEYWORD_IDEAS_TIMEOUT

    gl: str | None = Field(default=None, pattern=_COUNTRY)
    """Two-letter lowercase country code. Taken from the location when left out."""

    hl: str | None = Field(default=None, pattern=_LANGUAGE)
    """Language code. Arabic for an Arabic seed, otherwise English, when left out."""

    location: str | None = None
    """A city or region name to find local ideas for."""

    limit: int = Field(default=50, ge=1, le=200)
    """Most ideas to return (1 to 200)."""

    expand: KeywordExpansion = "questions"
    """Autocomplete calls to make: `none` (fewest), `questions`, `alphabet` or `all` (most)."""

    fresh: bool = False
    """Search the results page live (autocomplete is always live)."""

    def _result(self, seed: str, response: VeraResponse) -> tuple[str, dict[str, Any]]:
        content = _truncate(format_keyword_ideas(response.data, seed), self.max_characters)
        return content, _artifact(response)

    def _run(
        self,
        seed: str,
        run_manager: CallbackManagerForToolRun | None = None,
    ) -> tuple[str, dict[str, Any]]:
        try:
            response = self._vera.keyword_ideas(
                seed,
                gl=self.gl,
                hl=self.hl,
                location=self.location,
                limit=self.limit,
                expand=self.expand,
                fresh=self.fresh,
            )
        except VeraRequestError as exc:
            raise _refused(exc) from exc
        return self._result(seed, response)

    async def _arun(
        self,
        seed: str,
        run_manager: AsyncCallbackManagerForToolRun | None = None,
    ) -> tuple[str, dict[str, Any]]:
        try:
            response = await self._vera.akeyword_ideas(
                seed,
                gl=self.gl,
                hl=self.hl,
                location=self.location,
                limit=self.limit,
                expand=self.expand,
                fresh=self.fresh,
            )
        except VeraRequestError as exc:
            raise _refused(exc) from exc
        return self._result(seed, response)


class VeraAiAnswersInput(BaseModel):
    """Input to `VeraAiAnswersTool`."""

    prompt: str = Field(
        description=(
            "The question to put to the AI assistant, worded the way a buyer would ask it, for "
            "example 'What are the best specialty coffee roasters in Dubai?'."
        )
    )
    brand: str | None = Field(
        default=None,
        description=(
            "The brand name to look for in the answer. Leave out to use the brand the tool was "
            "set up with."
        ),
    )
    brand_domain: str | None = Field(
        default=None,
        description="The brand's website, for example roasters.ae. Leave out to use the default.",
    )
    competitors: list[str] | None = Field(
        default=None,
        description="Competitor names or domains to look for as well, up to 20.",
    )


class VeraAiAnswersTool(_VeraTool):
    """Ask AI assistants a question and see whether a brand is named in the answer.

    Puts a buyer's question to ChatGPT, Claude or Gemini and reports whether the brand and its
    competitors are named, how often, in what order, which pages the assistant cited and the share
    of voice. **This is the most expensive tool here**: each call to an assistant is priced on its
    own (Claude's highest) and the most it can cost is held first, so it asks one assistant once by
    default (`engines=["chatgpt"]`, `samples=1`). A failed or not-set-up assistant is not charged.
    Sign up at https://vera.waggle.ae/signup for free credits.

    Setup:
        ```bash
        pip install -U langchain-vera
        export VERA_API_KEY="vera_live_..."
        ```

    Instantiate:
        ```python
        from langchain_vera import VeraAiAnswersTool

        tool = VeraAiAnswersTool(
            brand="Roasters",
            brand_domain="roasters.ae",
            competitors=["Other Roasters"],
            engines=["chatgpt"],
            country="ae",
        )
        ```

    Invoke:
        ```python
        tool.invoke({"prompt": "What are the best specialty coffee roasters in Dubai?"})
        ```

    Invoke with a `ToolCall` to also get the full metrics as the artifact:
        ```python
        message = tool.invoke(
            {
                "type": "tool_call",
                "name": "vera_ai_answers",
                "id": "1",
                "args": {"prompt": "Best coffee roasters in Dubai?"},
            }
        )
        message.artifact["engines"][0]["aggregate"]["share_of_voice"]
        ```
    """

    name: str = "vera_ai_answers"
    description: str = (
        "Ask an AI assistant (ChatGPT, Claude or Gemini) a buyer's question and report whether a "
        "brand and its competitors are named in the answer: how often, in what order, with the "
        "pages the assistant cited and the share of voice. Input is the question, and the brand to "
        "look for unless the tool was set up with one. Expensive: each assistant call costs far "
        "more than a search, so use it sparingly."
    )
    args_schema: type[BaseModel] = VeraAiAnswersInput
    timeout: float = AI_ANSWERS_TIMEOUT

    brand: str | None = None
    """The brand to look for when the model gives none."""

    brand_domain: str | None = None
    """The brand's website, such as `roasters.ae`, used when the model gives none."""

    competitors: list[str] = Field(default_factory=list)
    """Competitor names or domains (up to 20), used when the model gives none."""

    engines: list[AIEngine] = Field(default_factory=lambda: list(_DEFAULT_ENGINES))
    """Which assistants to ask: `chatgpt`, `claude`, `gemini` (and `perplexity` when set up).
    Each is a separate paid call. Never empty."""

    samples: int = Field(default=1, ge=1, le=5)
    """How many times to ask each assistant (1 to 5), for a mention rate. Each is a paid call."""

    country: str | None = Field(default=None, pattern=_COUNTRY)
    """Two-letter lowercase country code of the market (`ae` when left out)."""

    language: str | None = Field(default=None, pattern=_LANGUAGE)
    """Language code of the market (`en` when left out)."""

    city: str | None = None
    """A city to localise the question to."""

    @field_validator("engines")
    @classmethod
    def _engines_not_empty(cls, value: list[AIEngine]) -> list[AIEngine]:
        if not value:
            msg = (
                "engines must not be empty: the API would ask every assistant, which is the most "
                f"expensive way to run this. Choose from: {', '.join(AI_ENGINES)}."
            )
            raise ValueError(msg)
        return value

    def _arguments(
        self,
        prompt: str,
        brand: str | None,
        brand_domain: str | None,
        competitors: list[str] | None,
    ) -> dict[str, Any]:
        name = (brand or self.brand or "").strip()
        if not name:
            msg = "No brand was given: pass the brand name to look for."
            raise ToolException(msg)
        return {
            "prompt": prompt,
            "brand": name,
            "brand_domain": brand_domain or self.brand_domain,
            "competitors": competitors if competitors else self.competitors,
            "engines": self.engines,
            "samples": self.samples,
            "country": self.country,
            "language": self.language,
            "city": self.city,
        }

    @staticmethod
    def _result(response: VeraResponse) -> tuple[str, dict[str, Any]]:
        return format_ai_answers(response.data), _artifact(response)

    def _run(
        self,
        prompt: str,
        brand: str | None = None,
        brand_domain: str | None = None,
        competitors: list[str] | None = None,
        run_manager: CallbackManagerForToolRun | None = None,
    ) -> tuple[str, dict[str, Any]]:
        arguments = self._arguments(prompt, brand, brand_domain, competitors)
        try:
            response = self._vera.ai_answers(
                arguments.pop("prompt"), arguments.pop("brand"), **arguments
            )
        except VeraRequestError as exc:
            raise _refused(exc) from exc
        return self._result(response)

    async def _arun(
        self,
        prompt: str,
        brand: str | None = None,
        brand_domain: str | None = None,
        competitors: list[str] | None = None,
        run_manager: AsyncCallbackManagerForToolRun | None = None,
    ) -> tuple[str, dict[str, Any]]:
        arguments = self._arguments(prompt, brand, brand_domain, competitors)
        try:
            response = await self._vera.aai_answers(
                arguments.pop("prompt"), arguments.pop("brand"), **arguments
            )
        except VeraRequestError as exc:
            raise _refused(exc) from exc
        return self._result(response)


class VeraGeoRadarInput(BaseModel):
    """Input to `VeraGeoRadarTool`."""

    query: str = Field(
        description=(
            "The topic query to look at, for example 'best specialty coffee roasters in dubai'."
        )
    )
    brand: str | None = Field(
        default=None,
        description=(
            "The brand name to evaluate. Leave out to use the brand the tool was set up with."
        ),
    )
    domain: str | None = Field(
        default=None,
        description="The brand's website, for example roasters.ae. Leave out to use the default.",
    )


class VeraGeoRadarTool(_VeraTextTool):
    """GEO citation radar: how likely AI answers are to cite a brand for one topic query.

    Returns a Markdown brief with a citation probability score (0 to 100), the forum threads worth
    joining, the list articles that name competitors, the questions the site does not answer yet and
    a brief for AI writers. Priced by what comes back. A failed call is not charged. Leave
    `location` and `gl` out and the API currently uses Dubai and `ae`: set them for any other
    market. Sign up at https://vera.waggle.ae/signup for free credits.

    Setup:
        ```bash
        pip install -U langchain-vera
        export VERA_API_KEY="vera_live_..."
        ```

    Instantiate:
        ```python
        from langchain_vera import VeraGeoRadarTool

        tool = VeraGeoRadarTool(brand="Roasters", domain="roasters.ae", location="Dubai", gl="ae")
        ```

    Invoke:
        ```python
        tool.invoke({"query": "best specialty coffee roasters in dubai"})
        ```
    """

    name: str = "vera_geo_radar"
    description: str = (
        "Check how likely AI answers are to cite a brand for one topic query. Returns a citation "
        "probability score (0 to 100), forum threads worth joining, list articles that name "
        "competitors, unanswered questions and a brief for AI writers. Input is the topic query, "
        "plus the brand and its website unless the tool was set up with them."
    )
    args_schema: type[BaseModel] = VeraGeoRadarInput
    timeout: float = GEO_RADAR_TIMEOUT

    brand: str | None = None
    """The brand to evaluate when the model gives none."""

    domain: str | None = None
    """The brand's website, such as `roasters.ae`, used when the model gives none."""

    location: str | None = None
    """A city or region to search from. The API uses Dubai when left out."""

    gl: str | None = Field(default=None, pattern=_COUNTRY)
    """Two-letter lowercase country code. The API uses `ae` when left out."""

    hl: str | None = Field(default=None, pattern=_LANGUAGE)
    """Language code, such as `en` or `ar`."""

    depth: int = Field(default=30, ge=1, le=100)
    """How many organic results to read (1 to 100)."""

    fresh: bool = False
    """Search live instead of reusing a result from the last 24 hours. Costs the full price."""

    def _result(self, query: str, response: VeraResponse) -> tuple[str, dict[str, Any]]:
        return self._brief(response.data, f"No radar was returned for {query!r}."), _artifact(
            response
        )

    def _run(
        self,
        query: str,
        brand: str | None = None,
        domain: str | None = None,
        run_manager: CallbackManagerForToolRun | None = None,
    ) -> tuple[str, dict[str, Any]]:
        try:
            response = self._vera.geo_radar_query(
                query,
                brand=brand or self.brand,
                domain=domain or self.domain,
                location=self.location,
                gl=self.gl,
                hl=self.hl,
                depth=self.depth,
                fresh=self.fresh,
            )
        except VeraRequestError as exc:
            raise _refused(exc) from exc
        return self._result(query, response)

    async def _arun(
        self,
        query: str,
        brand: str | None = None,
        domain: str | None = None,
        run_manager: AsyncCallbackManagerForToolRun | None = None,
    ) -> tuple[str, dict[str, Any]]:
        try:
            response = await self._vera.ageo_radar_query(
                query,
                brand=brand or self.brand,
                domain=domain or self.domain,
                location=self.location,
                gl=self.gl,
                hl=self.hl,
                depth=self.depth,
                fresh=self.fresh,
            )
        except VeraRequestError as exc:
            raise _refused(exc) from exc
        return self._result(query, response)


class VeraCompanyFootprintInput(BaseModel):
    """Input to `VeraCompanyFootprintTool`."""

    company: str = Field(description="The company or brand name, for example 'Roasters'.")
    domain: str | None = Field(
        default=None,
        description="The company's official website, for example roasters.ae, if known.",
    )


class VeraCompanyFootprintTool(_VeraTextTool):
    """Audit how a company shows up in search.

    Returns a Markdown brief that separates the company's own web properties, its social and
    reference profiles (Wikipedia, LinkedIn, Instagram, Crunchbase), its map listings, what third
    parties say about it and the questions people ask about the brand. Priced by what comes back. A
    failed call is not charged. Leave `location` and `gl` out and the API currently uses Dubai and
    `ae`: set them for any other market. Sign up at https://vera.waggle.ae/signup for free credits.

    Setup:
        ```bash
        pip install -U langchain-vera
        export VERA_API_KEY="vera_live_..."
        ```

    Instantiate:
        ```python
        from langchain_vera import VeraCompanyFootprintTool

        tool = VeraCompanyFootprintTool(location="Dubai", gl="ae")
        ```

    Invoke:
        ```python
        tool.invoke({"company": "Roasters", "domain": "roasters.ae"})
        ```
    """

    name: str = "vera_company_footprint"
    description: str = (
        "Audit how a company shows up in search: its own web properties, social and reference "
        "profiles (Wikipedia, LinkedIn, Instagram, Crunchbase), map listings, what third parties "
        "say about it and the questions people ask about the brand. Input is the company name, "
        "and its website if known. Use it for company research and reputation checks."
    )
    args_schema: type[BaseModel] = VeraCompanyFootprintInput
    timeout: float = COMPANY_FOOTPRINT_TIMEOUT

    location: str | None = None
    """A city or region to search from. The API uses Dubai when left out."""

    gl: str | None = Field(default=None, pattern=_COUNTRY)
    """Two-letter lowercase country code. The API uses `ae` when left out."""

    hl: str | None = Field(default=None, pattern=_LANGUAGE)
    """Language code, such as `en` or `ar`."""

    depth: int = Field(default=20, ge=1, le=100)
    """How many organic results to read (1 to 100)."""

    fresh: bool = False
    """Search live instead of reusing a result from the last 24 hours. Costs the full price."""

    def _result(self, company: str, response: VeraResponse) -> tuple[str, dict[str, Any]]:
        return self._brief(response.data, f"No footprint was returned for {company!r}."), _artifact(
            response
        )

    def _run(
        self,
        company: str,
        domain: str | None = None,
        run_manager: CallbackManagerForToolRun | None = None,
    ) -> tuple[str, dict[str, Any]]:
        try:
            response = self._vera.company_footprint(
                company,
                domain=domain,
                location=self.location,
                gl=self.gl,
                hl=self.hl,
                depth=self.depth,
                fresh=self.fresh,
            )
        except VeraRequestError as exc:
            raise _refused(exc) from exc
        return self._result(company, response)

    async def _arun(
        self,
        company: str,
        domain: str | None = None,
        run_manager: AsyncCallbackManagerForToolRun | None = None,
    ) -> tuple[str, dict[str, Any]]:
        try:
            response = await self._vera.acompany_footprint(
                company,
                domain=domain,
                location=self.location,
                gl=self.gl,
                hl=self.hl,
                depth=self.depth,
                fresh=self.fresh,
            )
        except VeraRequestError as exc:
            raise _refused(exc) from exc
        return self._result(company, response)


def format_keyword_ideas(data: dict[str, Any], seed: str = "") -> str:
    """Render keyword ideas as a model-friendly list grouped by intent."""
    ideas = data.get("ideas")
    rows = [row for row in ideas if isinstance(row, dict)] if isinstance(ideas, list) else []
    label = str(data.get("seed") or seed)
    if not rows:
        return f"No keyword ideas found for {label!r}."
    groups: dict[str, list[str]] = {}
    for row in rows:
        keyword = str(row.get("keyword") or "").strip()
        if keyword:
            groups.setdefault(str(row.get("intent") or "informational"), []).append(keyword)
    ordered = [*_INTENT_ORDER, *(name for name in groups if name not in _INTENT_ORDER)]
    total = sum(len(group) for group in groups.values())
    lines = [f"Keyword ideas for {label!r} ({total} ideas)"]
    for intent in ordered:
        if intent in groups:
            lines.append(f"\n{intent}:")
            lines.extend(f"- {keyword}" for keyword in groups[intent])
    notes = data.get("notes")
    if isinstance(notes, list) and notes:
        lines.append("\nNotes: " + " ".join(str(note) for note in notes))
    return "\n".join(lines)


def _percent(value: Any) -> str | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool) and 0 <= value <= 1:
        return f"{round(value * 100)}%"
    return None


def _engine_line(entry: dict[str, Any], brand: str) -> list[str]:
    engine = str(entry.get("engine") or "assistant")
    name = _ENGINE_NAMES.get(engine, engine.title())
    if entry.get("status") == "not_configured":
        return [f"- {name}: not set up, so it did not answer and was not charged."]
    aggregate = entry.get("aggregate")
    answered = aggregate.get("samples_ok") if isinstance(aggregate, dict) else 0
    if not isinstance(aggregate, dict) or not answered:
        samples = entry.get("samples")
        errors = [
            str(sample["error"].get("message") or sample["error"].get("code") or "")
            for sample in (samples if isinstance(samples, list) else [])
            if isinstance(sample, dict) and isinstance(sample.get("error"), dict)
        ]
        reason = f" ({errors[0]})" if errors and errors[0] else ""
        return [f"- {name}: did not answer{reason}."]

    mentions = aggregate.get("mentions") or 0
    verdict = f"{brand} named in {mentions} of {answered} answer{'s' if answered != 1 else ''}"
    facts = []
    if (rate := _percent(aggregate.get("mention_rate"))) and answered > 1:
        facts.append(f"mention rate {rate}")
    if share := _percent(aggregate.get("share_of_voice")):
        facts.append(f"share of voice {share}")
    rank = aggregate.get("first_mention_rank_avg")
    if isinstance(rank, (int, float)) and not isinstance(rank, bool):
        facts.append(f"average first rank {rank:g}")
    if cited := _percent(aggregate.get("brand_domain_cited_rate")):
        facts.append(f"brand site cited in {cited} of answers")
    if total := aggregate.get("total_citations"):
        facts.append(f"{total} citations")
    lines = [f"- {name}: {verdict}" + (f" ({', '.join(facts)})" if facts else "")]

    rivals = aggregate.get("competitors")
    parts = []
    for rival in rivals if isinstance(rivals, list) else []:
        if not isinstance(rival, dict):
            continue
        detail = f"named in {rival.get('mentions') or 0} of {answered}"
        if isinstance(rival.get("first_mention_rank_avg"), (int, float)):
            detail += f", average rank {rival['first_mention_rank_avg']:g}"
        parts.append(f"{rival.get('name')}: {detail}")
    if parts:
        lines.append("    Competitors: " + "; ".join(parts))

    share_by_domain = aggregate.get("citation_share_by_domain")
    if isinstance(share_by_domain, dict):
        top = sorted(
            ((domain, value) for domain, value in share_by_domain.items() if _percent(value)),
            key=lambda pair: (-pair[1], pair[0]),
        )[:5]
        if top:
            lines.append(
                "    Most cited domains: " + ", ".join(f"{d} {_percent(v)}" for d, v in top)
            )
    return lines


def format_ai_answers(data: dict[str, Any]) -> str:
    """Render an AI answers report as a short, model-friendly summary per assistant."""
    engines = data.get("engines")
    entries = (
        [entry for entry in engines if isinstance(entry, dict)] if isinstance(engines, list) else []
    )
    if not entries:
        return "No assistant answered."
    brand = str(data.get("brand") or "The brand")
    site = f" ({data['brand_domain']})" if data.get("brand_domain") else ""
    lines = [f"AI answers for {str(data.get('prompt') or '')!r}: is {brand}{site} named?"]
    for entry in entries:
        lines.extend(_engine_line(entry, brand))
    notes = data.get("notes")
    if isinstance(notes, list) and notes:
        lines.append("Notes: " + " ".join(str(note) for note in notes))
    return "\n".join(lines)


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
    "VeraAiAnswersInput",
    "VeraAiAnswersTool",
    "VeraCompanyFootprintInput",
    "VeraCompanyFootprintTool",
    "VeraGeoRadarInput",
    "VeraGeoRadarTool",
    "VeraKeywordIdeasInput",
    "VeraKeywordIdeasTool",
    "VeraScrapeInput",
    "VeraScrapeTool",
    "VeraSearchInput",
    "VeraSearchTool",
    "VeraSerpInput",
    "VeraSerpTool",
    "format_ai_answers",
    "format_keyword_ideas",
    "format_results",
]
