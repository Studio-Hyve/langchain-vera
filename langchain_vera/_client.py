"""A small, typed HTTP client for the Vera API, built on httpx with explicit timeouts."""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from importlib import metadata
from typing import Any, Literal

import httpx
from pydantic import SecretStr

from langchain_vera.errors import (
    VeraAPIError,
    VeraAuthenticationError,
    VeraConfigurationError,
    VeraConflictError,
    VeraConnectionError,
    VeraInsufficientCreditsError,
    VeraPermissionError,
    VeraRateLimitError,
    VeraRequestError,
    VeraResponseError,
    VeraServerError,
    VeraTimeoutError,
)

DEFAULT_BASE_URL = "https://vera.waggle.ae/api/vera/v1"
API_KEY_ENV = "VERA_API_KEY"
DEFAULT_TIMEOUT = httpx.Timeout(60.0, connect=10.0)

CREDITS_HEADER = "x-waggle-credits-used"
REQUEST_ID_HEADER = "x-waggle-request-id"
REPLAYED_HEADER = "x-waggle-idempotency-replayed"

SEARCH_MAX_RESULTS = 100
MAX_FIELDS = 10

SearchKind = Literal["web", "news", "videos"]
_SEARCH_PATHS: dict[str, str] = {
    "web": "/search",
    "news": "/search/news",
    "videos": "/search/videos",
}

_RULE_KEYS = frozenset({"selector", "attribute", "multiple"})
_IDEMPOTENCY_KEY = re.compile(r"[A-Za-z0-9_.:-]{16,128}")
_TWO_LETTERS = re.compile(r"[a-z]{2}")


def _version() -> str:
    try:
        return metadata.version("langchain-vera")
    except metadata.PackageNotFoundError:  # pragma: no cover - running from a source tree
        return "0.0.0"


@dataclass(frozen=True)
class VeraResponse:
    """A successful Vera answer.

    Attributes:
        data: The `data` object of the response (page content, search results, contacts...).
        credits_used: What this call was charged. The `X-Waggle-Credits-Used` header is the
            authority; it is `0` for an idempotent replay.
        request_id: Vera's request id.
        elapsed_ms: Server-side time, when reported.
        idempotency_replayed: True when the answer was replayed from an earlier call with the
            same idempotency key.
        raw: The whole response body as Vera sent it.
    """

    data: dict[str, Any]
    credits_used: int
    request_id: str | None = None
    elapsed_ms: int | None = None
    idempotency_replayed: bool = False
    raw: dict[str, Any] = field(default_factory=dict, repr=False)


def _header_int(response: httpx.Response, name: str) -> int | None:
    value = response.headers.get(name)
    try:
        return int(value) if value is not None else None
    except ValueError:
        return None


def _header_float(response: httpx.Response, name: str) -> float | None:
    value = response.headers.get(name)
    try:
        return float(value) if value is not None else None
    except ValueError:
        return None


def error_from_response(response: httpx.Response) -> VeraAPIError:
    """Map a non-2xx answer to the matching typed error."""
    status = response.status_code
    message, code = "", None
    try:
        body = response.json()
    except ValueError:
        body = None
    if isinstance(body, dict):
        error = body.get("error")
        if isinstance(error, str):
            message = error
        elif isinstance(error, dict):
            raw_message, raw_code = error.get("message"), error.get("code")
            message = raw_message if isinstance(raw_message, str) else ""
            code = raw_code if isinstance(raw_code, str) else None

    retry_after = _header_float(response, "retry-after")
    charged = _header_int(response, CREDITS_HEADER)
    cls: type[VeraAPIError]
    if status == httpx.codes.UNAUTHORIZED:
        cls = VeraAuthenticationError
        summary = "Vera rejected the API key. It is missing, invalid, expired or revoked"
    elif status == httpx.codes.PAYMENT_REQUIRED:
        cls = VeraInsufficientCreditsError
        summary = (
            "Not enough Vera credits. Add credits or choose a plan at "
            "https://vera.waggle.ae/pricing"
        )
    elif status == httpx.codes.FORBIDDEN:
        cls, summary = VeraPermissionError, "Vera refused the request"
    elif status in {httpx.codes.CONFLICT, httpx.codes.GONE}:
        cls, summary = VeraConflictError, "Vera could not run the request (idempotency key)"
    elif status == httpx.codes.TOO_MANY_REQUESTS:
        cls, summary = VeraRateLimitError, "Vera rate limit reached"
    elif status in {httpx.codes.BAD_GATEWAY, httpx.codes.SERVICE_UNAVAILABLE}:
        cls, summary = VeraServerError, "Vera is temporarily unavailable"
    elif status >= httpx.codes.INTERNAL_SERVER_ERROR:
        cls, summary = VeraServerError, "Vera had an internal error"
    else:
        cls, summary = VeraRequestError, "Vera could not complete the request"
    # Vera labels most failures "fetch_failed", a 401 included, so its code is only trusted for
    # failed fetches and searches.
    if status not in {httpx.codes.UNPROCESSABLE_ENTITY, httpx.codes.BAD_GATEWAY}:
        code = None
    parts = [f"{summary} (HTTP {status})."]
    if message:
        parts.append(f"Vera said: {message}")
    if retry_after is not None:
        parts.append(f"Retry after {retry_after:g} seconds.")
    if charged == 0:
        parts.append("No credits were charged.")
    return cls(
        " ".join(parts),
        status_code=status,
        code=code,
        request_id=response.headers.get(REQUEST_ID_HEADER),
        credits_used=charged,
        retry_after=retry_after,
    )


def parse_response(response: httpx.Response) -> VeraResponse:
    """Turn an httpx response into a `VeraResponse`, or raise the matching typed error."""
    if not response.is_success:
        raise error_from_response(response)
    try:
        body = response.json()
    except ValueError as exc:
        msg = "Vera returned a response that is not JSON."
        raise VeraResponseError(msg, status_code=response.status_code) from exc
    if not isinstance(body, dict) or not isinstance(body.get("data"), dict):
        msg = "Vera returned an unexpected response: no data object."
        raise VeraResponseError(msg, status_code=response.status_code)
    charged = _header_int(response, CREDITS_HEADER)
    if charged is None:
        body_charged = body.get("credits_used")
        charged = body_charged if isinstance(body_charged, int) else 0
    elapsed = body.get("elapsed_ms")
    return VeraResponse(
        data=body["data"],
        credits_used=charged,
        request_id=response.headers.get(REQUEST_ID_HEADER) or body.get("request_id"),
        elapsed_ms=elapsed if isinstance(elapsed, int) else None,
        idempotency_replayed=response.headers.get(REPLAYED_HEADER) == "true",
        raw=body,
    )


def _check_url(url: str) -> str:
    cleaned = url.strip() if isinstance(url, str) else ""
    if not re.match(r"https?://", cleaned, re.IGNORECASE) or not 8 <= len(cleaned) <= 2048:
        msg = "url must be a full http:// or https:// address of at most 2048 characters."
        raise ValueError(msg)
    return cleaned


def _check_key(key: str | None) -> str | None:
    if key is None:
        return None
    if not _IDEMPOTENCY_KEY.fullmatch(key):
        msg = (
            "idempotency_key must be 16 to 128 letters, digits, underscores, dots, colons "
            "or hyphens."
        )
        raise ValueError(msg)
    return key


def _field_rules(fields: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    if len(fields) > MAX_FIELDS:
        msg = f"At most {MAX_FIELDS} fields can be extracted per scrape, got {len(fields)}."
        raise ValueError(msg)
    rules: dict[str, dict[str, Any]] = {}
    for name, rule in fields.items():
        if isinstance(rule, str):
            rules[name] = {"selector": rule}
        elif isinstance(rule, Mapping) and isinstance(rule.get("selector"), str):
            rules[name] = {k: v for k, v in rule.items() if k in _RULE_KEYS}
        else:
            msg = f"Field {name!r} must be a CSS selector string or a mapping with a 'selector'."
            raise ValueError(msg)
    return rules


def build_scrape_request(
    url: str,
    *,
    format: Literal["markdown", "html"] = "markdown",  # noqa: A002
    render: bool = False,
    fields: Mapping[str, Any] | None = None,
    min_quality_score: int | None = None,
    wait_ms: int | None = None,
    wait_for: str | None = None,
) -> dict[str, Any]:
    """Build the JSON body of `POST /scrape`.

    `format="markdown"` maps to the API's `response_format: "markdown"`, `format="html"` to
    `return_html: true`, and `fields` to the API's `extract` rules.
    """
    if format not in {"markdown", "html"}:
        msg = "format must be 'markdown' or 'html'."
        raise ValueError(msg)
    body: dict[str, Any] = {"url": _check_url(url)}
    if render:
        body["render"] = True
    if format == "html":
        body["return_html"] = True
    else:
        body["response_format"] = "markdown"
    if fields:
        if render:
            msg = "fields cannot be combined with render=True: extraction reads the static page."
            raise ValueError(msg)
        if format == "html":
            msg = "fields cannot be combined with format='html'."
            raise ValueError(msg)
        body["extract"] = _field_rules(fields)
    if (wait_ms is not None or wait_for) and not render:
        msg = "wait_ms and wait_for need render=True."
        raise ValueError(msg)
    if wait_ms is not None:
        if not isinstance(wait_ms, int) or not 0 <= wait_ms <= 3000:
            msg = "wait_ms must be a whole number from 0 to 3000."
            raise ValueError(msg)
        body["wait_ms"] = wait_ms
    if wait_for:
        body["wait_for"] = wait_for
    if min_quality_score is not None:
        if not isinstance(min_quality_score, int) or not 0 <= min_quality_score <= 100:
            msg = "min_quality_score must be a whole number from 0 to 100."
            raise ValueError(msg)
        body["min_quality_score"] = min_quality_score
    return body


def build_search_request(
    query: str,
    *,
    limit: int = 10,
    country: str | None = None,
    language: str | None = None,
) -> dict[str, Any]:
    """Build the JSON body of the search endpoints."""
    cleaned = query.strip() if isinstance(query, str) else ""
    if not cleaned or len(cleaned) > 500 or any(ord(c) < 32 or ord(c) == 127 for c in cleaned):
        msg = "query must be 1 to 500 characters with no control characters."
        raise ValueError(msg)
    valid_limit = isinstance(limit, int) and not isinstance(limit, bool)
    if not valid_limit or not 1 <= limit <= SEARCH_MAX_RESULTS:
        msg = f"limit must be a whole number from 1 to {SEARCH_MAX_RESULTS}."
        raise ValueError(msg)
    body: dict[str, Any] = {"query": cleaned, "limit": limit}
    for name, value in (("country", country), ("language", language)):
        if value is not None:
            if not _TWO_LETTERS.fullmatch(value):
                msg = f"{name} must be two lowercase letters, for example 'us'."
                raise ValueError(msg)
            body[name] = value
    return body


class VeraClient:
    """Client for the Vera API.

    Calls are plain request/response with timeouts and no automatic retries: a retried scrape
    could be charged twice, so retry yourself, ideally with an `idempotency_key`.

    Example:
        ```python
        client = VeraClient()  # reads VERA_API_KEY
        page = client.scrape("https://example.com")
        print(page.data["markdown"], page.credits_used)
        ```
    """

    def __init__(
        self,
        api_key: str | SecretStr | None = None,
        *,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float | httpx.Timeout = DEFAULT_TIMEOUT,
        http_client: httpx.Client | None = None,
        async_http_client: httpx.AsyncClient | None = None,
    ) -> None:
        """Create a client.

        Args:
            api_key: A Vera API key (`vera_live_...`). Defaults to the `VERA_API_KEY`
                environment variable.
            base_url: The API base URL.
            timeout: Seconds, or an `httpx.Timeout`. The default is 60 seconds with a 10 second
                connect timeout.
            http_client: An `httpx.Client` to use instead of a temporary one per call.
            async_http_client: An `httpx.AsyncClient` to use instead of a temporary one per call.

        Raises:
            VeraConfigurationError: If no API key is given and `VERA_API_KEY` is not set.
        """
        key = api_key.get_secret_value() if isinstance(api_key, SecretStr) else api_key
        key = (key or os.environ.get(API_KEY_ENV, "")).strip()
        if not key:
            msg = f"No Vera API key. Pass api_key or set the {API_KEY_ENV} environment variable."
            raise VeraConfigurationError(msg)
        self._api_key = SecretStr(key)
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout if isinstance(timeout, httpx.Timeout) else httpx.Timeout(timeout)
        self._http_client = http_client
        self._async_http_client = async_http_client

    def __repr__(self) -> str:
        """Show the base URL only: the API key is never printed."""
        return f"VeraClient(base_url={self.base_url!r})"

    def _headers(self, idempotency_key: str | None) -> dict[str, str]:
        headers = {
            "Authorization": f"Bearer {self._api_key.get_secret_value()}",
            "Accept": "application/json",
            "User-Agent": f"langchain-vera/{_version()}",
        }
        if idempotency_key:
            headers["Idempotency-Key"] = idempotency_key
        return headers

    def _post(
        self, path: str, body: dict[str, Any], idempotency_key: str | None = None
    ) -> VeraResponse:
        headers = self._headers(_check_key(idempotency_key))
        url = self.base_url + path
        try:
            if self._http_client is not None:
                response = self._http_client.post(url, json=body, headers=headers)
            else:
                with httpx.Client(timeout=self.timeout) as client:
                    response = client.post(url, json=body, headers=headers)
        except httpx.TimeoutException as exc:
            msg = f"Vera did not answer within the timeout ({exc.__class__.__name__})."
            raise VeraTimeoutError(msg) from exc
        except httpx.HTTPError as exc:
            msg = f"Could not reach Vera: {exc}"
            raise VeraConnectionError(msg) from exc
        return parse_response(response)

    async def _apost(
        self, path: str, body: dict[str, Any], idempotency_key: str | None = None
    ) -> VeraResponse:
        headers = self._headers(_check_key(idempotency_key))
        url = self.base_url + path
        try:
            if self._async_http_client is not None:
                response = await self._async_http_client.post(url, json=body, headers=headers)
            else:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    response = await client.post(url, json=body, headers=headers)
        except httpx.TimeoutException as exc:
            msg = f"Vera did not answer within the timeout ({exc.__class__.__name__})."
            raise VeraTimeoutError(msg) from exc
        except httpx.HTTPError as exc:
            msg = f"Could not reach Vera: {exc}"
            raise VeraConnectionError(msg) from exc
        return parse_response(response)

    # Scrape: 1 credit, 5 with render=True ---------------------------------------------------

    def scrape(
        self,
        url: str,
        *,
        format: Literal["markdown", "html"] = "markdown",  # noqa: A002
        render: bool = False,
        fields: Mapping[str, Any] | None = None,
        min_quality_score: int | None = None,
        wait_ms: int | None = None,
        wait_for: str | None = None,
        idempotency_key: str | None = None,
    ) -> VeraResponse:
        """Read a public page (`POST /scrape`). 1 credit, or 5 with `render=True`.

        Args:
            url: The page to read.
            format: `"markdown"` (the main content, in `data["markdown"]`) or `"html"` (the raw
                page, in `data["html"]`). Text, links and metadata come back either way.
            render: Load the page in a real browser first, for content built by JavaScript.
            fields: Named CSS-selector rules, up to 10, read from the static page into
                `data["extracted"]`. A value is a selector string or a mapping with `selector` and
                optional `attribute` and `multiple`. Not with `render` or `format="html"`.
            min_quality_score: Refuse (uncharged) a result below this 0 to 100 quality score.
            wait_ms: With `render`, milliseconds to wait after load (0 to 3000).
            wait_for: With `render`, a CSS selector to wait for.
            idempotency_key: Protects a retried call from being charged twice.

        Raises:
            ValueError: If an argument is invalid. Nothing is sent and nothing is charged.
            VeraAPIError: If Vera answers with an error (see `langchain_vera.errors`).
        """
        body = build_scrape_request(
            url,
            format=format,
            render=render,
            fields=fields,
            min_quality_score=min_quality_score,
            wait_ms=wait_ms,
            wait_for=wait_for,
        )
        return self._post("/scrape", body, idempotency_key)

    async def ascrape(
        self,
        url: str,
        *,
        format: Literal["markdown", "html"] = "markdown",  # noqa: A002
        render: bool = False,
        fields: Mapping[str, Any] | None = None,
        min_quality_score: int | None = None,
        wait_ms: int | None = None,
        wait_for: str | None = None,
        idempotency_key: str | None = None,
    ) -> VeraResponse:
        """Async version of `scrape`."""
        body = build_scrape_request(
            url,
            format=format,
            render=render,
            fields=fields,
            min_quality_score=min_quality_score,
            wait_ms=wait_ms,
            wait_for=wait_for,
        )
        return await self._apost("/scrape", body, idempotency_key)

    # Search: 3 credits whatever the limit ---------------------------------------------------

    def search(
        self,
        query: str,
        *,
        kind: SearchKind = "web",
        limit: int = 10,
        country: str | None = None,
        language: str | None = None,
        idempotency_key: str | None = None,
    ) -> VeraResponse:
        """Search the web, news or videos. 3 credits, whatever the limit (1 to 100)."""
        path = self._search_path(kind)
        body = build_search_request(query, limit=limit, country=country, language=language)
        return self._post(path, body, idempotency_key)

    async def asearch(
        self,
        query: str,
        *,
        kind: SearchKind = "web",
        limit: int = 10,
        country: str | None = None,
        language: str | None = None,
        idempotency_key: str | None = None,
    ) -> VeraResponse:
        """Async version of `search`."""
        path = self._search_path(kind)
        body = build_search_request(query, limit=limit, country=country, language=language)
        return await self._apost(path, body, idempotency_key)

    @staticmethod
    def _search_path(kind: str) -> str:
        try:
            return _SEARCH_PATHS[kind]
        except KeyError:
            msg = "kind must be 'web', 'news' or 'videos'."
            raise ValueError(msg) from None

    # Contacts and products: 1 credit each --------------------------------------------------

    def contacts(self, url: str, *, idempotency_key: str | None = None) -> VeraResponse:
        """Find the emails, phones and social links a page publishes. 1 credit."""
        return self._post("/contacts", {"url": _check_url(url)}, idempotency_key)

    async def acontacts(self, url: str, *, idempotency_key: str | None = None) -> VeraResponse:
        """Async version of `contacts`."""
        return await self._apost("/contacts", {"url": _check_url(url)}, idempotency_key)

    def products(
        self,
        url: str,
        *,
        marketplace: Literal["amazon", "walmart", "shopee"] | None = None,
        idempotency_key: str | None = None,
    ) -> VeraResponse:
        """Read the product data a page publishes. 1 credit."""
        return self._post("/products", self._product_body(url, marketplace), idempotency_key)

    async def aproducts(
        self,
        url: str,
        *,
        marketplace: Literal["amazon", "walmart", "shopee"] | None = None,
        idempotency_key: str | None = None,
    ) -> VeraResponse:
        """Async version of `products`."""
        return await self._apost("/products", self._product_body(url, marketplace), idempotency_key)

    @staticmethod
    def _product_body(url: str, marketplace: str | None) -> dict[str, Any]:
        body: dict[str, Any] = {"url": _check_url(url)}
        if marketplace is not None:
            if marketplace not in {"amazon", "walmart", "shopee"}:
                msg = "marketplace must be 'amazon', 'walmart' or 'shopee'."
                raise ValueError(msg)
            body["marketplace"] = marketplace
        return body
