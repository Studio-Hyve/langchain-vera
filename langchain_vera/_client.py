"""A small, typed HTTP client for the Vera API, built on httpx with explicit timeouts."""

from __future__ import annotations

import os
import re
from collections.abc import Mapping, Sequence
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

DEFAULT_BASE_URL = "https://vera.waggle.ae/v1"
"""The direct Vera API. The website's `/api/vera` proxy lacks the newer routes and cuts a request
at 40 seconds, so it is not the default."""
API_KEY_ENV = "VERA_API_KEY"
PRICING_URL = "https://vera.waggle.ae/pricing"
CONNECT_TIMEOUT = 10.0
DEFAULT_TIMEOUT = httpx.Timeout(60.0, connect=CONNECT_TIMEOUT)

# Seconds the long-running search tools get, unless you set a `timeout` yourself. The API's own
# limits are 30 s for a results page, 60 s for keyword ideas and the query radar, 45 s for a company
# footprint and 150 s for AI answers, each after a few seconds to price the call; a margin goes on
# top so that Vera answers (a 504, not charged) before the client gives up.
SERP_TIMEOUT = 50.0
KEYWORD_IDEAS_TIMEOUT = 80.0
GEO_RADAR_TIMEOUT = 80.0
COMPANY_FOOTPRINT_TIMEOUT = 65.0
AI_ANSWERS_TIMEOUT = 175.0

CREDITS_HEADER = "x-waggle-credits-used"
REQUEST_ID_HEADER = "x-waggle-request-id"
REPLAYED_HEADER = "x-waggle-idempotency-replayed"

SEARCH_MAX_RESULTS = 100
MAX_FIELDS = 10

SERP_ENGINES = ("google", "bing", "duckduckgo", "brave", "yahoo")
SERP_MAX_DEPTH = 100
SERP_DEFAULT_FEATURES = ("organic", "paa", "ai_overview")
SERP_ADD_ONS = ("local_pack", "shopping")
KEYWORD_EXPANSIONS = ("none", "questions", "alphabet", "all")
KEYWORD_MAX_LIMIT = 200
AI_ENGINES = ("chatgpt", "claude", "gemini", "perplexity")
AI_MAX_SAMPLES = 5
AI_MAX_PROMPT = 1000
AI_MAX_BRAND = 200
AI_MAX_COMPETITORS = 20
AI_MAX_COMPETITOR = 200
AI_MAX_CITY = 100
DEPTH_MAX = 100
MAX_QUERY = 500
MAX_LOCATION = 200
MAX_COMPANY = 200
MAX_DOMAIN = 253

SearchKind = Literal["web", "news", "videos"]
SerpEngine = Literal["google", "bing", "duckduckgo", "brave", "yahoo"]
KeywordExpansion = Literal["none", "questions", "alphabet", "all"]
AIEngine = Literal["chatgpt", "claude", "gemini", "perplexity"]
_SEARCH_PATHS: dict[str, str] = {
    "web": "/search",
    "news": "/search/news",
    "videos": "/search/videos",
}

_RULE_KEYS = frozenset({"selector", "attribute", "multiple"})
_IDEMPOTENCY_KEY = re.compile(r"[A-Za-z0-9_.:-]{16,128}")
_TWO_LETTERS = re.compile(r"[a-z]{2}")
_LANGUAGE = re.compile(r"[a-z]{2,3}(?:-[a-z0-9]{2,8})*")


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
        summary = f"Not enough Vera credits. Add credits or choose a plan at {PRICING_URL}"
    elif status == httpx.codes.FORBIDDEN:
        cls, summary = VeraPermissionError, "Vera refused the request"
    elif status in {httpx.codes.CONFLICT, httpx.codes.GONE}:
        cls, summary = VeraConflictError, "Vera could not run the request (idempotency key)"
    elif status == httpx.codes.TOO_MANY_REQUESTS:
        cls, summary = VeraRateLimitError, "Vera rate limit reached"
    elif status in {httpx.codes.BAD_GATEWAY, httpx.codes.SERVICE_UNAVAILABLE}:
        cls, summary = VeraServerError, "Vera is temporarily unavailable"
    elif status == httpx.codes.GATEWAY_TIMEOUT:
        cls = VeraServerError
        summary = (
            "Vera timed out waiting for the service behind this tool. Try again, or ask for "
            "less: a smaller depth, fewer samples or one assistant"
        )
    elif status >= httpx.codes.INTERNAL_SERVER_ERROR:
        cls, summary = VeraServerError, "Vera had an internal error"
    elif status in {httpx.codes.BAD_REQUEST, httpx.codes.UNPROCESSABLE_ENTITY}:
        cls, summary = VeraRequestError, "Invalid input"
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
    # The header is the proof. A 503 or 504 without it is still one Vera documents as not charged.
    unavailable = status in {httpx.codes.SERVICE_UNAVAILABLE, httpx.codes.GATEWAY_TIMEOUT}
    if charged == 0 or (unavailable and charged is None):
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


# --- The search tools: results page, keyword ideas, AI answers, query radar, company footprint ---
#
# These routes take any JSON object and the search service checks every field, so the checks below
# only catch what is certain (a missing required field, a number out of range, an unknown choice)
# and save a round trip. A refusal from the service is never charged either way.


def _text(value: object, name: str, *, max_length: int, required: bool = True) -> str | None:
    """A trimmed string of at most `max_length` characters with no control characters."""
    cleaned = value.strip() if isinstance(value, str) else ""
    if not cleaned:
        if required:
            msg = f"{name} is required."
            raise ValueError(msg)
        return None
    if len(cleaned) > max_length or any(ord(c) < 32 or ord(c) == 127 for c in cleaned):
        msg = f"{name} must be at most {max_length} characters with no control characters."
        raise ValueError(msg)
    return cleaned


def _whole(value: object, name: str, low: int, high: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or not low <= value <= high:
        msg = f"{name} must be a whole number from {low} to {high}."
        raise ValueError(msg)
    return value


def _market(
    body: dict[str, Any],
    country: str | None,
    language: str | None,
    *,
    country_key: str = "gl",
    language_key: str = "hl",
) -> None:
    """Add country and language, lower-cased (the service lower-cases them too)."""
    if country is not None:
        code = country.strip().lower()
        if not _TWO_LETTERS.fullmatch(code):
            msg = f"{country_key} must be a two-letter country code, for example 'ae'."
            raise ValueError(msg)
        body[country_key] = code
    if language is not None:
        tag = language.strip().lower()
        if not _LANGUAGE.fullmatch(tag):
            msg = f"{language_key} must be a language code, for example 'en', 'ar' or 'pt-br'."
            raise ValueError(msg)
        body[language_key] = tag


def _serp_path(engine: str) -> str:
    if engine not in SERP_ENGINES:
        msg = f"engine must be one of: {', '.join(SERP_ENGINES)}."
        raise ValueError(msg)
    return f"/serp/{engine}"


def build_serp_request(
    q: str,
    *,
    engine: SerpEngine = "google",
    gl: str | None = None,
    hl: str | None = None,
    location: str | None = None,
    depth: int = 10,
    features: Sequence[str] | None = None,
    fresh: bool = False,
) -> dict[str, Any]:
    """Build the JSON body of `POST /serp/{engine}`.

    `features` is `None` for the default results page (`organic`, `paa`, `ai_overview`). To add the
    Google map listings or shopping, pass the defaults plus `"local_pack"` and/or `"shopping"`:
    each add-on is priced on its own and exists on Google only.
    """
    _serp_path(engine)
    body: dict[str, Any] = {
        "q": _text(q, "q", max_length=MAX_QUERY),
        "depth": _whole(depth, "depth", 1, SERP_MAX_DEPTH),
    }
    _market(body, gl, hl)
    place = _text(location, "location", max_length=MAX_LOCATION, required=False)
    if place:
        body["location"] = place
    if features is not None:
        allowed = {*SERP_DEFAULT_FEATURES, *SERP_ADD_ONS, "places"}
        chosen: list[str] = []
        for feature in features:
            name = str(feature).strip().lower()
            if name not in allowed:
                msg = f"Unknown feature {feature!r}. Use: {', '.join(sorted(allowed))}."
                raise ValueError(msg)
            name = "local_pack" if name == "places" else name
            if name not in chosen:
                chosen.append(name)
        if not chosen:
            msg = "features must not be empty; leave it out for the default results page."
            raise ValueError(msg)
        if engine != "google" and set(chosen) & set(SERP_ADD_ONS):
            msg = "local_pack and shopping are Google add-ons; use engine='google'."
            raise ValueError(msg)
        body["features"] = chosen
    if fresh:
        body["fresh"] = True
    return body


def build_keyword_ideas_request(
    seed: str,
    *,
    gl: str | None = None,
    hl: str | None = None,
    location: str | None = None,
    limit: int = 50,
    expand: KeywordExpansion = "questions",
    fresh: bool = False,
) -> dict[str, Any]:
    """Build the JSON body of `POST /keywords/ideas`.

    `limit` and `expand` are always sent, so what a call costs does not depend on a server default.
    """
    if expand not in KEYWORD_EXPANSIONS:
        msg = f"expand must be one of: {', '.join(KEYWORD_EXPANSIONS)}."
        raise ValueError(msg)
    body: dict[str, Any] = {
        "seed": _text(seed, "seed", max_length=200),
        "limit": _whole(limit, "limit", 1, KEYWORD_MAX_LIMIT),
        "expand": expand,
    }
    _market(body, gl, hl)
    place = _text(location, "location", max_length=MAX_LOCATION, required=False)
    if place:
        body["location"] = place
    if fresh:
        body["fresh"] = True
    return body


def _competitors(values: Sequence[str] | None) -> list[str]:
    names: list[str] = []
    seen: set[str] = set()
    for value in values or ():
        name = str(value).strip()
        if not name or name.casefold() in seen:
            continue
        if len(name) > AI_MAX_COMPETITOR:
            msg = f"A competitor name is longer than {AI_MAX_COMPETITOR} characters."
            raise ValueError(msg)
        seen.add(name.casefold())
        names.append(name)
    if len(names) > AI_MAX_COMPETITORS:
        msg = f"At most {AI_MAX_COMPETITORS} competitors are accepted, got {len(names)}."
        raise ValueError(msg)
    return names


def build_ai_answers_request(
    prompt: str,
    brand: str,
    *,
    brand_domain: str | None = None,
    competitors: Sequence[str] | None = None,
    engines: Sequence[str] = ("chatgpt",),
    samples: int = 1,
    country: str | None = None,
    language: str | None = None,
    city: str | None = None,
) -> dict[str, Any]:
    """Build the JSON body of `POST /ai/answers`.

    `engines` is always sent and never empty: the API asks every assistant when the field is left
    out, which is the most expensive way to run this. Each assistant call is priced on its own.
    """
    if isinstance(engines, str) or not engines:
        msg = "engines must be a non-empty list, for example ['chatgpt']."
        raise ValueError(msg)
    chosen: list[str] = []
    for engine in engines:
        name = str(engine).strip().lower()
        if name not in AI_ENGINES:
            msg = f"Unknown engine {engine!r}. Use: {', '.join(AI_ENGINES)}."
            raise ValueError(msg)
        if name not in chosen:
            chosen.append(name)
    body: dict[str, Any] = {
        "prompt": _text(prompt, "prompt", max_length=AI_MAX_PROMPT),
        "brand": _text(brand, "brand", max_length=AI_MAX_BRAND),
        "engines": chosen,
        "samples": _whole(samples, "samples", 1, AI_MAX_SAMPLES),
    }
    domain = _text(brand_domain, "brand_domain", max_length=MAX_DOMAIN, required=False)
    if domain:
        body["brand_domain"] = domain
    names = _competitors(competitors)
    if names:
        body["competitors"] = names
    _market(body, country, language, country_key="country", language_key="language")
    place = _text(city, "city", max_length=AI_MAX_CITY, required=False)
    if place:
        body["city"] = place
    return body


def build_geo_radar_request(
    query: str,
    *,
    brand: str | None = None,
    domain: str | None = None,
    location: str | None = None,
    gl: str | None = None,
    hl: str | None = None,
    depth: int = 30,
    fresh: bool = False,
) -> dict[str, Any]:
    """Build the JSON body of `POST /geo/radar/query`."""
    body: dict[str, Any] = {
        "query": _text(query, "query", max_length=MAX_QUERY),
        "depth": _whole(depth, "depth", 1, DEPTH_MAX),
    }
    name = _text(brand, "brand", max_length=AI_MAX_BRAND, required=False)
    if name:
        body["brand"] = name
    site = _text(domain, "domain", max_length=MAX_DOMAIN, required=False)
    if site:
        body["domain"] = site
    _market(body, gl, hl)
    place = _text(location, "location", max_length=MAX_LOCATION, required=False)
    if place:
        body["location"] = place
    if fresh:
        body["fresh"] = True
    return body


def build_company_footprint_request(
    company: str,
    *,
    domain: str | None = None,
    location: str | None = None,
    gl: str | None = None,
    hl: str | None = None,
    depth: int = 20,
    fresh: bool = False,
) -> dict[str, Any]:
    """Build the JSON body of `POST /company/footprint`."""
    body: dict[str, Any] = {
        "company": _text(company, "company", max_length=MAX_COMPANY),
        "depth": _whole(depth, "depth", 1, DEPTH_MAX),
    }
    site = _text(domain, "domain", max_length=MAX_DOMAIN, required=False)
    if site:
        body["domain"] = site
    _market(body, gl, hl)
    place = _text(location, "location", max_length=MAX_LOCATION, required=False)
    if place:
        body["location"] = place
    if fresh:
        body["fresh"] = True
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
        timeout: float | httpx.Timeout | None = None,
        http_client: httpx.Client | None = None,
        async_http_client: httpx.AsyncClient | None = None,
    ) -> None:
        """Create a client.

        Args:
            api_key: A Vera API key (`vera_live_...`). Defaults to the `VERA_API_KEY`
                environment variable.
            base_url: The API base URL. The default is the direct API,
                `https://vera.waggle.ae/v1`.
            timeout: Seconds, or an `httpx.Timeout`, applied to every call. Left out, page and
                search calls get 60 seconds (10 to connect), and the long-running search tools
                get what they need: 50 seconds for a results page, 80 for keyword ideas and the
                query radar, 65 for a company footprint and 175 for AI answers (the API takes up
                to about 150). If you set one, make it long enough for the calls you make.
            http_client: An `httpx.Client` to use instead of a temporary one per call. It is used
                as it is, except that the long-running search tools ask it for their own timeout
                unless you passed `timeout`.
            async_http_client: An `httpx.AsyncClient` to use instead of a temporary one per call,
                with the same rule.

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
        self._timeout_explicit = timeout is not None
        if timeout is None:
            self.timeout = DEFAULT_TIMEOUT
        elif isinstance(timeout, httpx.Timeout):
            self.timeout = timeout
        else:
            self.timeout = httpx.Timeout(timeout)
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

    def _timeout_for(self, long_call: float | None) -> httpx.Timeout:
        """What a call waits: your `timeout`, else the long-running tool's own, else the default."""
        if long_call is None or self._timeout_explicit:
            return self.timeout
        return httpx.Timeout(long_call, connect=CONNECT_TIMEOUT)

    def _post(
        self,
        path: str,
        body: dict[str, Any],
        idempotency_key: str | None = None,
        *,
        long_call: float | None = None,
    ) -> VeraResponse:
        headers = self._headers(_check_key(idempotency_key))
        url = self.base_url + path
        try:
            if self._http_client is not None:
                extra: dict[str, Any] = {}
                if long_call is not None and not self._timeout_explicit:
                    extra["timeout"] = self._timeout_for(long_call)
                response = self._http_client.post(url, json=body, headers=headers, **extra)
            else:
                with httpx.Client(timeout=self._timeout_for(long_call)) as client:
                    response = client.post(url, json=body, headers=headers)
        except httpx.TimeoutException as exc:
            msg = f"Vera did not answer within the timeout ({exc.__class__.__name__})."
            raise VeraTimeoutError(msg) from exc
        except httpx.HTTPError as exc:
            msg = f"Could not reach Vera: {exc}"
            raise VeraConnectionError(msg) from exc
        return parse_response(response)

    async def _apost(
        self,
        path: str,
        body: dict[str, Any],
        idempotency_key: str | None = None,
        *,
        long_call: float | None = None,
    ) -> VeraResponse:
        headers = self._headers(_check_key(idempotency_key))
        url = self.base_url + path
        try:
            if self._async_http_client is not None:
                extra: dict[str, Any] = {}
                if long_call is not None and not self._timeout_explicit:
                    extra["timeout"] = self._timeout_for(long_call)
                response = await self._async_http_client.post(
                    url, json=body, headers=headers, **extra
                )
            else:
                async with httpx.AsyncClient(timeout=self._timeout_for(long_call)) as client:
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

    # Search tools: priced by what comes back --------------------------------------------------
    #
    # The most a call can cost is held first, you are charged for what was served and the rest is
    # given back. `credits_used` on the response is the charge. A failed call is not charged, and
    # an `idempotency_key` makes a retried call free (a failure is replayed too, so retry a failed
    # call with a new key).

    def serp(
        self,
        q: str,
        *,
        engine: SerpEngine = "google",
        gl: str | None = None,
        hl: str | None = None,
        location: str | None = None,
        depth: int = 10,
        features: Sequence[str] | None = None,
        fresh: bool = False,
        idempotency_key: str | None = None,
    ) -> VeraResponse:
        """Read a search results page (`POST /serp/{engine}`), priced by what comes back.

        Returns the organic results tagged by kind (forum, video, shop) and, on Google, the page's
        AI Overview, People Also Ask and Knowledge Graph in `data`, plus `data["llm_markdown"]`, a
        compact brief for a model. A live results page of up to ten results is typically 3 credits
        and 1 when served from the cache; every further ten results is another page.

        Args:
            q: What to search for, up to 500 characters.
            engine: `"google"`, `"bing"`, `"duckduckgo"`, `"brave"` or `"yahoo"`.
            gl: Two-letter country code to localise results, such as `"ae"`.
            hl: Language code, such as `"en"` or `"ar"`.
            location: A city or region to search from, such as `"Dubai"`.
            depth: How many results to read, 1 to 100.
            features: `None` for the default results page. To add map listings or shopping on
                Google, pass `["organic", "paa", "ai_overview", "local_pack", "shopping"]` or the
                subset you want. Each add-on is priced on its own.
            fresh: Search live instead of reusing a result from the last 24 hours (full price).
            idempotency_key: Protects a retried call from being charged twice.

        Raises:
            ValueError: If an argument is invalid. Nothing is sent and nothing is charged.
            VeraAPIError: If Vera answers with an error (see `langchain_vera.errors`).
        """
        body = build_serp_request(
            q,
            engine=engine,
            gl=gl,
            hl=hl,
            location=location,
            depth=depth,
            features=features,
            fresh=fresh,
        )
        return self._post(_serp_path(engine), body, idempotency_key, long_call=SERP_TIMEOUT)

    async def aserp(
        self,
        q: str,
        *,
        engine: SerpEngine = "google",
        gl: str | None = None,
        hl: str | None = None,
        location: str | None = None,
        depth: int = 10,
        features: Sequence[str] | None = None,
        fresh: bool = False,
        idempotency_key: str | None = None,
    ) -> VeraResponse:
        """Async version of `serp`."""
        body = build_serp_request(
            q,
            engine=engine,
            gl=gl,
            hl=hl,
            location=location,
            depth=depth,
            features=features,
            fresh=fresh,
        )
        return await self._apost(_serp_path(engine), body, idempotency_key, long_call=SERP_TIMEOUT)

    def keyword_ideas(
        self,
        seed: str,
        *,
        gl: str | None = None,
        hl: str | None = None,
        location: str | None = None,
        limit: int = 50,
        expand: KeywordExpansion = "questions",
        fresh: bool = False,
        idempotency_key: str | None = None,
    ) -> VeraResponse:
        """Keyword ideas for a seed (`POST /keywords/ideas`), priced by what comes back.

        Ideas come from Google autocomplete, People Also Ask and related searches, cleaned and
        de-duplicated, each with its `sources` and an `intent` label (question, comparison,
        commercial, local or informational). They are in `data["ideas"]`. Works in Arabic. Ideas
        only: no search volumes.

        Args:
            seed: The keyword to find ideas for, up to 200 characters.
            gl: Two-letter country code. Taken from the location when left out.
            hl: Language code. Arabic for an Arabic seed, otherwise English, when left out.
            location: A city or region name.
            limit: Most ideas to return, 1 to 200.
            expand: How many autocomplete calls to make: `"none"` (the seed alone, cheapest),
                `"questions"`, `"alphabet"` or `"all"` (the most calls). More calls cost more.
            fresh: Search the results page live (autocomplete is always live).
            idempotency_key: Protects a retried call from being charged twice.
        """
        body = build_keyword_ideas_request(
            seed, gl=gl, hl=hl, location=location, limit=limit, expand=expand, fresh=fresh
        )
        return self._post("/keywords/ideas", body, idempotency_key, long_call=KEYWORD_IDEAS_TIMEOUT)

    async def akeyword_ideas(
        self,
        seed: str,
        *,
        gl: str | None = None,
        hl: str | None = None,
        location: str | None = None,
        limit: int = 50,
        expand: KeywordExpansion = "questions",
        fresh: bool = False,
        idempotency_key: str | None = None,
    ) -> VeraResponse:
        """Async version of `keyword_ideas`."""
        body = build_keyword_ideas_request(
            seed, gl=gl, hl=hl, location=location, limit=limit, expand=expand, fresh=fresh
        )
        return await self._apost(
            "/keywords/ideas", body, idempotency_key, long_call=KEYWORD_IDEAS_TIMEOUT
        )

    def ai_answers(
        self,
        prompt: str,
        brand: str,
        *,
        brand_domain: str | None = None,
        competitors: Sequence[str] | None = None,
        engines: Sequence[AIEngine] = ("chatgpt",),
        samples: int = 1,
        country: str | None = None,
        language: str | None = None,
        city: str | None = None,
        idempotency_key: str | None = None,
    ) -> VeraResponse:
        """What AI assistants answer, and whether a brand is in it (`POST /ai/answers`).

        Puts a buyer's question to each assistant and reports, in `data["engines"]`, whether the
        brand and its competitors are named in the answer, in what order, with the citations and
        cited domains behind it, a mention rate over the samples and the share of voice. This is
        the most expensive call: each call to an assistant is priced on its own, Claude's the
        highest, and the most it can cost is held first. Start with one assistant and one sample.
        An assistant that is not set up answers `not_configured` and costs nothing.

        Args:
            prompt: The question to ask, as a buyer would, up to 1,000 characters.
            brand: The brand name to look for, up to 200 characters.
            brand_domain: The brand's website, such as `"roasters.ae"`, to tell whether your site
                was cited.
            competitors: Up to 20 competitor names or domains, for the share of voice.
            engines: Which assistants to ask: `"chatgpt"`, `"claude"`, `"gemini"` (and
                `"perplexity"` when Vera has it set up). Never empty: the API would ask every
                assistant if the field were left out.
            samples: How many times to ask each assistant, 1 to 5, for a mention rate. Every sample
                is a separate paid call.
            country: Two-letter country code of the market (`"ae"` when left out).
            language: Language code of the market (`"en"` when left out).
            city: A city to localise the question to.
            idempotency_key: Protects a retried call from being charged twice.
        """
        body = build_ai_answers_request(
            prompt,
            brand,
            brand_domain=brand_domain,
            competitors=competitors,
            engines=engines,
            samples=samples,
            country=country,
            language=language,
            city=city,
        )
        return self._post("/ai/answers", body, idempotency_key, long_call=AI_ANSWERS_TIMEOUT)

    async def aai_answers(
        self,
        prompt: str,
        brand: str,
        *,
        brand_domain: str | None = None,
        competitors: Sequence[str] | None = None,
        engines: Sequence[AIEngine] = ("chatgpt",),
        samples: int = 1,
        country: str | None = None,
        language: str | None = None,
        city: str | None = None,
        idempotency_key: str | None = None,
    ) -> VeraResponse:
        """Async version of `ai_answers`."""
        body = build_ai_answers_request(
            prompt,
            brand,
            brand_domain=brand_domain,
            competitors=competitors,
            engines=engines,
            samples=samples,
            country=country,
            language=language,
            city=city,
        )
        return await self._apost("/ai/answers", body, idempotency_key, long_call=AI_ANSWERS_TIMEOUT)

    def geo_radar_query(
        self,
        query: str,
        *,
        brand: str | None = None,
        domain: str | None = None,
        location: str | None = None,
        gl: str | None = None,
        hl: str | None = None,
        depth: int = 30,
        fresh: bool = False,
        idempotency_key: str | None = None,
    ) -> VeraResponse:
        """GEO citation radar for one query, in depth (`POST /geo/radar/query`).

        Reports how likely AI answers are to cite the brand (`citation_probability_score`, 0 to
        100), the forum threads worth joining, the list articles that name competitors, the
        questions the site does not answer yet and a brief for AI writers (`llm_markdown`). Priced
        by what comes back. Leave `location` and `gl` out and the API currently uses Dubai and
        `"ae"`: set them for any other market.

        Args:
            query: The topic query to look at, up to 500 characters.
            brand: The brand name to evaluate.
            domain: The brand's website, such as `"roasters.ae"`.
            location: A city or region to search from.
            gl: Two-letter country code.
            hl: Language code.
            depth: How many organic results to read, 1 to 100.
            fresh: Search live instead of reusing a result from the last 24 hours (full price).
            idempotency_key: Protects a retried call from being charged twice.
        """
        body = build_geo_radar_request(
            query,
            brand=brand,
            domain=domain,
            location=location,
            gl=gl,
            hl=hl,
            depth=depth,
            fresh=fresh,
        )
        return self._post("/geo/radar/query", body, idempotency_key, long_call=GEO_RADAR_TIMEOUT)

    async def ageo_radar_query(
        self,
        query: str,
        *,
        brand: str | None = None,
        domain: str | None = None,
        location: str | None = None,
        gl: str | None = None,
        hl: str | None = None,
        depth: int = 30,
        fresh: bool = False,
        idempotency_key: str | None = None,
    ) -> VeraResponse:
        """Async version of `geo_radar_query`."""
        body = build_geo_radar_request(
            query,
            brand=brand,
            domain=domain,
            location=location,
            gl=gl,
            hl=hl,
            depth=depth,
            fresh=fresh,
        )
        return await self._apost(
            "/geo/radar/query", body, idempotency_key, long_call=GEO_RADAR_TIMEOUT
        )

    def company_footprint(
        self,
        company: str,
        *,
        domain: str | None = None,
        location: str | None = None,
        gl: str | None = None,
        hl: str | None = None,
        depth: int = 20,
        fresh: bool = False,
        idempotency_key: str | None = None,
    ) -> VeraResponse:
        """A company's footprint in search (`POST /company/footprint`), priced by what comes back.

        Separates the company's own web properties, its social and reference profiles (Wikipedia,
        LinkedIn, Instagram, Crunchbase), its map listings, what third parties say about it and the
        questions people ask about the brand, plus a brief (`llm_markdown`). Leave `location` and
        `gl` out and the API currently uses Dubai and `"ae"`: set them for any other market.

        Args:
            company: The company or brand name.
            domain: The company's official website, such as `"roasters.ae"`.
            location: A city or region to search from.
            gl: Two-letter country code.
            hl: Language code.
            depth: How many organic results to read, 1 to 100.
            fresh: Search live instead of reusing a result from the last 24 hours (full price).
            idempotency_key: Protects a retried call from being charged twice.
        """
        body = build_company_footprint_request(
            company, domain=domain, location=location, gl=gl, hl=hl, depth=depth, fresh=fresh
        )
        return self._post(
            "/company/footprint", body, idempotency_key, long_call=COMPANY_FOOTPRINT_TIMEOUT
        )

    async def acompany_footprint(
        self,
        company: str,
        *,
        domain: str | None = None,
        location: str | None = None,
        gl: str | None = None,
        hl: str | None = None,
        depth: int = 20,
        fresh: bool = False,
        idempotency_key: str | None = None,
    ) -> VeraResponse:
        """Async version of `company_footprint`."""
        body = build_company_footprint_request(
            company, domain=domain, location=location, gl=gl, hl=hl, depth=depth, fresh=fresh
        )
        return await self._apost(
            "/company/footprint", body, idempotency_key, long_call=COMPANY_FOOTPRINT_TIMEOUT
        )
