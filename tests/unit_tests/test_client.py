"""VeraClient: request building, response parsing, typed errors and timeouts (HTTP mocked)."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
import pytest
from pydantic import SecretStr

from langchain_vera import (
    VeraAPIError,
    VeraAuthenticationError,
    VeraClient,
    VeraConfigurationError,
    VeraConflictError,
    VeraConnectionError,
    VeraError,
    VeraInsufficientCreditsError,
    VeraPermissionError,
    VeraRateLimitError,
    VeraRequestError,
    VeraResponseError,
    VeraServerError,
    VeraTimeoutError,
)
from tests.unit_tests.conftest import Recorder, failure, make_client, success

SNAPSHOT = json.loads(
    (Path(__file__).parent.parent / "fixtures" / "openapi-request-schemas.json").read_text()
)


class TestSetup:
    def test_needs_a_key(self) -> None:
        with pytest.raises(VeraConfigurationError, match="VERA_API_KEY") as caught:
            VeraClient()
        assert isinstance(caught.value, ValueError)
        assert isinstance(caught.value, VeraError)

    def test_reads_the_environment(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("VERA_API_KEY", "vera_live_from_env")
        recorder = Recorder(lambda _request: success())
        client = VeraClient(http_client=httpx.Client(transport=httpx.MockTransport(recorder)))
        client.scrape("https://example.com")
        assert recorder.last.headers["authorization"] == "Bearer vera_live_from_env"

    def test_an_explicit_key_beats_the_environment(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("VERA_API_KEY", "vera_live_from_env")
        recorder = Recorder(lambda _request: success())
        client = VeraClient(
            "vera_live_explicit", http_client=httpx.Client(transport=httpx.MockTransport(recorder))
        )
        client.scrape("https://example.com")
        assert recorder.last.headers["authorization"] == "Bearer vera_live_explicit"

    def test_sends_the_key_as_a_bearer_token(self) -> None:
        client, recorder = make_client()
        client.scrape("https://example.com")
        assert recorder.last.headers["authorization"] == "Bearer vera_live_test_key"
        assert recorder.last.headers["accept"] == "application/json"
        assert recorder.last.headers["user-agent"].startswith("langchain-vera/")

    def test_accepts_a_secret_str_and_never_prints_the_key(self) -> None:
        client = VeraClient(SecretStr("vera_live_secret_value"))
        assert "vera_live_secret_value" not in repr(client)
        assert "vera_live_secret_value" not in str(client.__dict__)

    def test_trims_the_base_url(self) -> None:
        client, recorder = make_client(base_url="https://vera.example/api/v1/")
        client.scrape("https://example.com")
        assert str(recorder.last.url) == "https://vera.example/api/v1/scrape"

    def test_default_base_url_is_production(self) -> None:
        client, recorder = make_client()
        client.scrape("https://example.com")
        assert str(recorder.last.url) == "https://vera.waggle.ae/api/vera/v1/scrape"


class TestScrapeRequest:
    def test_default_is_markdown(self) -> None:
        client, recorder = make_client()
        client.scrape("https://example.com")
        assert recorder.last.method == "POST"
        assert recorder.body == {"url": "https://example.com", "response_format": "markdown"}

    def test_html_maps_to_return_html(self) -> None:
        client, recorder = make_client()
        client.scrape("https://example.com", format="html")
        assert recorder.body == {"url": "https://example.com", "return_html": True}

    def test_render_and_waits(self) -> None:
        client, recorder = make_client()
        client.scrape("https://example.com", render=True, wait_ms=1500, wait_for=".results")
        assert recorder.body == {
            "url": "https://example.com",
            "render": True,
            "response_format": "markdown",
            "wait_ms": 1500,
            "wait_for": ".results",
        }

    def test_fields_map_to_extract_rules(self) -> None:
        client, recorder = make_client()
        client.scrape(
            "https://example.com",
            fields={
                "title": "h1",
                "links": {"selector": "a.nav", "attribute": "href", "multiple": True},
                "ignored_keys_dropped": {"selector": "p", "bogus": 1},
            },
        )
        assert recorder.body["extract"] == {
            "title": {"selector": "h1"},
            "links": {"selector": "a.nav", "attribute": "href", "multiple": True},
            "ignored_keys_dropped": {"selector": "p"},
        }

    def test_min_quality_score(self) -> None:
        client, recorder = make_client()
        client.scrape("https://example.com", min_quality_score=60)
        assert recorder.body["min_quality_score"] == 60

    def test_idempotency_key_is_a_header_not_a_body_field(self) -> None:
        client, recorder = make_client()
        client.scrape("https://example.com", idempotency_key="report-2026-01-01-page-0001")
        assert recorder.last.headers["idempotency-key"] == "report-2026-01-01-page-0001"
        assert "idempotency_key" not in recorder.body

    @pytest.mark.parametrize(
        "url", ["", "example.com", "ftp://example.com/x", "https://e.co/" + "a" * 2050]
    )
    def test_rejects_bad_urls_before_sending(self, url: str) -> None:
        client, recorder = make_client()
        with pytest.raises(ValueError, match="url must be"):
            client.scrape(url)
        assert recorder.requests == []

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"format": "text"}, "format must be"),
            ({"fields": {"a": "h1"}, "render": True}, "cannot be combined with render"),
            ({"fields": {"a": "h1"}, "format": "html"}, "cannot be combined with format"),
            ({"fields": {f"f{i}": "p" for i in range(11)}}, "At most 10 fields"),
            ({"fields": {"a": 5}}, "must be a CSS selector"),
            ({"wait_ms": 100}, "need render=True"),
            ({"wait_for": ".x"}, "need render=True"),
            ({"render": True, "wait_ms": 3001}, "wait_ms must be"),
            ({"render": True, "wait_ms": -1}, "wait_ms must be"),
            ({"min_quality_score": 101}, "min_quality_score must be"),
            ({"idempotency_key": "short"}, "idempotency_key must be"),
        ],
    )
    def test_rejects_invalid_options(self, kwargs: dict[str, Any], message: str) -> None:
        client, recorder = make_client()
        with pytest.raises(ValueError, match=message):
            client.scrape("https://example.com", **kwargs)
        assert recorder.requests == []

    def test_ten_fields_are_allowed(self) -> None:
        client, recorder = make_client()
        client.scrape("https://example.com", fields={f"f{i}": "p" for i in range(10)})
        assert len(recorder.body["extract"]) == 10


class TestSearchRequest:
    @pytest.mark.parametrize(
        ("kind", "path"),
        [("web", "/search"), ("news", "/search/news"), ("videos", "/search/videos")],
    )
    def test_endpoints(self, kind: str, path: str) -> None:
        client, recorder = make_client(success({"results": []}, charged=3))
        client.search("vera", kind=kind)  # type: ignore[arg-type]
        assert recorder.last.url.path == "/api/vera/v1" + path
        assert recorder.body == {"query": "vera", "limit": 10}

    def test_options(self) -> None:
        client, recorder = make_client(success({"results": []}, charged=3))
        client.search("  q  ", limit=100, country="ae", language="ar")
        assert recorder.body == {"query": "q", "limit": 100, "country": "ae", "language": "ar"}

    @pytest.mark.parametrize("limit", [0, 101, -1])
    def test_limit_bounds(self, limit: int) -> None:
        client, _ = make_client()
        with pytest.raises(ValueError, match="limit must be"):
            client.search("q", limit=limit)

    @pytest.mark.parametrize("limit", [True, 2.5])
    def test_limit_must_be_an_integer(self, limit: Any) -> None:
        client, _ = make_client()
        with pytest.raises(ValueError, match="limit must be"):
            client.search("q", limit=limit)

    @pytest.mark.parametrize("query", ["", "   ", "a" * 501, "bad\x07query"])
    def test_query_validation(self, query: str) -> None:
        client, _ = make_client()
        with pytest.raises(ValueError, match="query must be"):
            client.search(query)

    @pytest.mark.parametrize(
        ("name", "value"), [("country", "US"), ("country", "usa"), ("language", "EN")]
    )
    def test_country_and_language_are_two_lowercase_letters(self, name: str, value: str) -> None:
        client, _ = make_client()
        options: dict[str, Any] = {name: value}
        with pytest.raises(ValueError, match=name):
            client.search("q", **options)

    def test_unknown_kind(self) -> None:
        client, _ = make_client()
        with pytest.raises(ValueError, match="kind must be"):
            client.search("q", kind="images")  # type: ignore[arg-type]


class TestContactsAndProducts:
    def test_contacts(self) -> None:
        client, recorder = make_client()
        client.contacts("https://example.com/contact")
        assert recorder.last.url.path.endswith("/contacts")
        assert recorder.body == {"url": "https://example.com/contact"}

    def test_products(self) -> None:
        client, recorder = make_client()
        client.products("https://shop.example.com/p/1")
        assert recorder.body == {"url": "https://shop.example.com/p/1"}
        client.products("https://www.amazon.com/dp/B000000000", marketplace="amazon")
        assert recorder.body["marketplace"] == "amazon"

    def test_unknown_marketplace(self) -> None:
        client, _ = make_client()
        with pytest.raises(ValueError, match="marketplace"):
            client.products("https://shop.example.com/p/1", marketplace="ebay")  # type: ignore[arg-type]


class TestResponses:
    def test_success_fields(self) -> None:
        client, _ = make_client(success({"markdown": "# Hi"}, charged=5))
        response = client.scrape("https://example.com", render=True)
        assert response.data == {"markdown": "# Hi"}
        assert response.credits_used == 5
        assert response.request_id == "req_test"
        assert response.elapsed_ms == 123
        assert response.idempotency_replayed is False
        assert response.raw["credits_used"] == 5

    def test_the_credits_header_is_authoritative(self) -> None:
        # A replay shows the first charge in the stored body but 0 in the header.
        client, _ = make_client(
            success(
                {"markdown": "x"},
                charged=1,
                headers={"x-waggle-credits-used": "0", "x-waggle-idempotency-replayed": "true"},
            )
        )
        response = client.scrape(
            "https://example.com", idempotency_key="report-2026-01-01-page-0001"
        )
        assert response.credits_used == 0
        assert response.idempotency_replayed is True

    def test_falls_back_to_the_body_credits(self) -> None:
        reply = httpx.Response(200, json={"data": {}, "credits_used": 3})
        client, _ = make_client(reply)
        assert client.scrape("https://example.com").credits_used == 3

    def test_no_credit_information_means_zero(self) -> None:
        client, _ = make_client(httpx.Response(200, json={"data": {}}))
        response = client.scrape("https://example.com")
        assert response.credits_used == 0
        assert response.request_id is None

    @pytest.mark.parametrize(
        "reply",
        [
            httpx.Response(200, text="<html>not json</html>"),
            httpx.Response(200, json=["not", "an", "object"]),
            httpx.Response(200, json={"no": "data"}),
            httpx.Response(200, json={"data": "not an object"}),
        ],
    )
    def test_malformed_success_is_a_response_error(self, reply: httpx.Response) -> None:
        client, _ = make_client(reply)
        with pytest.raises(VeraResponseError):
            client.scrape("https://example.com")


class TestErrors:
    @pytest.mark.parametrize(
        ("status", "error_class"),
        [
            (401, VeraAuthenticationError),
            (402, VeraInsufficientCreditsError),
            (403, VeraPermissionError),
            (409, VeraConflictError),
            (410, VeraConflictError),
            (422, VeraRequestError),
            (404, VeraRequestError),
            (429, VeraRateLimitError),
            (500, VeraServerError),
            (502, VeraServerError),
            (503, VeraServerError),
        ],
    )
    def test_status_maps_to_a_typed_error(
        self, status: int, error_class: type[VeraAPIError]
    ) -> None:
        client, _ = make_client(failure(status, "Something."))
        with pytest.raises(error_class) as caught:
            client.scrape("https://example.com")
        assert type(caught.value) is error_class
        assert isinstance(caught.value, VeraAPIError)
        assert caught.value.status_code == status
        assert "Something." in str(caught.value)

    def test_401_hides_the_misleading_fetch_failed_code(self) -> None:
        client, _ = make_client(
            failure(401, "API key is invalid, expired or revoked.", code="fetch_failed")
        )
        with pytest.raises(VeraAuthenticationError) as caught:
            client.scrape("https://example.com")
        assert caught.value.code is None
        assert "missing, invalid, expired or revoked" in str(caught.value)

    def test_422_keeps_the_machine_code_and_request_details(self) -> None:
        client, _ = make_client(
            failure(
                422,
                "This page is disallowed by robots.txt.",
                code="robots_disallowed",
                headers={"x-waggle-credits-used": "0", "x-waggle-request-id": "req_42"},
            )
        )
        with pytest.raises(VeraRequestError) as caught:
            client.scrape("https://example.com")
        error = caught.value
        assert error.code == "robots_disallowed"
        assert error.request_id == "req_42"
        assert error.credits_used == 0
        assert "No credits were charged." in str(error)

    def test_429_carries_retry_after(self) -> None:
        client, _ = make_client(failure(429, "Too many requests.", headers={"retry-after": "30"}))
        with pytest.raises(VeraRateLimitError) as caught:
            client.search("q")
        assert caught.value.retry_after == 30.0
        assert "Retry after 30 seconds." in str(caught.value)

    def test_the_trial_daily_limit_arrives_as_a_429_with_a_message(self) -> None:
        reply = httpx.Response(
            429,
            json={"error": {"code": "trial_daily_limit", "message": "Daily limit reached."}},
            headers={"retry-after": "3600"},
        )
        client, _ = make_client(reply)
        with pytest.raises(VeraRateLimitError, match=r"Daily limit reached\.") as caught:
            client.scrape("https://example.com")
        assert caught.value.code is None  # codes are only trusted for 422 and 502
        assert caught.value.retry_after == 3600.0

    def test_502_keeps_the_code(self) -> None:
        client, _ = make_client(
            failure(502, "Search is temporarily unavailable.", code="fetch_failed")
        )
        with pytest.raises(VeraServerError) as caught:
            client.search("q")
        assert caught.value.code == "fetch_failed"

    def test_non_json_error_body(self) -> None:
        client, _ = make_client(httpx.Response(502, text="<html>Bad gateway</html>"))
        with pytest.raises(VeraServerError) as caught:
            client.scrape("https://example.com")
        assert "Vera said" not in str(caught.value)

    def test_string_error_field(self) -> None:
        client, _ = make_client(httpx.Response(402, json={"error": "Out of credits."}))
        with pytest.raises(VeraInsufficientCreditsError, match=r"Out of credits\."):
            client.scrape("https://example.com")

    def test_does_not_claim_no_charge_without_the_header(self) -> None:
        client, _ = make_client(failure(422, "Nope."))
        with pytest.raises(VeraRequestError) as caught:
            client.scrape("https://example.com")
        assert "No credits were charged." not in str(caught.value)
        assert caught.value.credits_used is None

    def test_transport_failures(self) -> None:
        def refuse(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("connection refused", request=request)

        def stall(request: httpx.Request) -> httpx.Response:
            raise httpx.ReadTimeout("too slow", request=request)

        client, _ = make_client(refuse)
        with pytest.raises(VeraConnectionError, match="Could not reach Vera") as refused:
            client.scrape("https://example.com")
        assert not isinstance(refused.value, VeraTimeoutError)
        client, _ = make_client(stall)
        with pytest.raises(VeraTimeoutError, match="timeout") as stalled:
            client.scrape("https://example.com")
        assert isinstance(stalled.value, VeraConnectionError)


class TestTimeouts:
    def test_defaults(self) -> None:
        client = VeraClient("vera_live_x")
        assert client.timeout == httpx.Timeout(60.0, connect=10.0)

    def test_a_number_becomes_a_timeout(self) -> None:
        assert VeraClient("vera_live_x", timeout=5).timeout == httpx.Timeout(5)

    def test_the_timeout_reaches_the_temporary_httpx_client(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seen: dict[str, Any] = {}
        real = httpx.Client

        def spy(**kwargs: Any) -> httpx.Client:
            seen.update(kwargs)
            return real(transport=httpx.MockTransport(lambda r: success()))

        monkeypatch.setattr("langchain_vera._client.httpx.Client", spy)
        VeraClient("vera_live_x", timeout=7).scrape("https://example.com")
        assert seen["timeout"] == httpx.Timeout(7)

    async def test_the_timeout_reaches_the_temporary_async_client(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seen: dict[str, Any] = {}
        real = httpx.AsyncClient

        def spy(**kwargs: Any) -> httpx.AsyncClient:
            seen.update(kwargs)
            return real(transport=httpx.MockTransport(lambda r: success()))

        monkeypatch.setattr("langchain_vera._client.httpx.AsyncClient", spy)
        await VeraClient("vera_live_x", timeout=9).ascrape("https://example.com")
        assert seen["timeout"] == httpx.Timeout(9)


class TestAsync:
    async def test_scrape_and_search(self) -> None:
        client, recorder = make_client(success({"markdown": "# Async", "results": []}, charged=1))
        page = await client.ascrape("https://example.com", render=True)
        assert page.data["markdown"] == "# Async"
        assert recorder.body["render"] is True
        found = await client.asearch("q", kind="news", limit=3)
        assert recorder.last.url.path.endswith("/search/news")
        assert found.credits_used == 1

    async def test_contacts_and_products(self) -> None:
        client, recorder = make_client()
        await client.acontacts("https://example.com")
        assert recorder.last.url.path.endswith("/contacts")
        await client.aproducts("https://shop.example.com/p", marketplace="walmart")
        assert recorder.body["marketplace"] == "walmart"

    async def test_errors_are_typed_too(self) -> None:
        client, _ = make_client(failure(402, "This request requires 3 credits."))
        with pytest.raises(VeraInsufficientCreditsError):
            await client.asearch("q")

    async def test_transport_failures(self) -> None:
        def stall(request: httpx.Request) -> httpx.Response:
            raise httpx.ReadTimeout("too slow", request=request)

        client, _ = make_client(stall)
        with pytest.raises(VeraTimeoutError):
            await client.ascrape("https://example.com")

    async def test_async_idempotency_key(self) -> None:
        client, recorder = make_client()
        await client.ascrape("https://example.com", idempotency_key="report-2026-01-01-page-0002")
        assert recorder.last.headers["idempotency-key"] == "report-2026-01-01-page-0002"


class TestContractWithPublishedOpenApi:
    """Every body this client builds must fit Vera's published request schemas."""

    @staticmethod
    def _check(path: str, body: dict[str, Any]) -> None:
        schema = SNAPSHOT["schemas"][SNAPSHOT["paths"][f"/v1{path}"]["schema"]]
        assert schema["additionalProperties"] is False
        for required in schema.get("required", []):
            assert required in body
        for key, value in body.items():
            assert key in schema["properties"], f"{key} is not a field of {path}"
            options = schema["properties"][key].get("anyOf", [schema["properties"][key]])
            assert any(
                TestContractWithPublishedOpenApi._fits(option, value) for option in options
            ), (
                key,
                value,
            )

    @staticmethod
    def _fits(option: dict[str, Any], value: Any) -> bool:
        if "enum" in option:
            return value in option["enum"]
        kind = option.get("type")
        if kind == "integer":
            return (
                isinstance(value, int)
                and not isinstance(value, bool)
                and option.get("minimum", value) <= value <= option.get("maximum", value)
            )
        if kind == "boolean":
            return isinstance(value, bool)
        if kind == "string":
            return isinstance(value, str)
        if kind == "object":
            return isinstance(value, dict)
        return False

    def test_every_request_fits_the_schema(self) -> None:
        client, recorder = make_client(success({"results": []}, charged=3))
        calls: list[Callable[[], Any]] = [
            lambda: client.scrape("https://example.com"),
            lambda: client.scrape("https://example.com", format="html", render=True),
            lambda: client.scrape(
                "https://example.com",
                render=True,
                wait_ms=3000,
                wait_for=".x",
                min_quality_score=100,
            ),
            lambda: client.scrape(
                "https://example.com",
                fields={"a": "h1", "b": {"selector": "a", "attribute": "href", "multiple": True}},
            ),
            lambda: client.contacts("https://example.com"),
            lambda: client.products("https://www.amazon.com/dp/B000000000", marketplace="amazon"),
            lambda: client.search("q", limit=100, country="ae", language="ar"),
            lambda: client.search("q", kind="news", limit=1),
            lambda: client.search("q", kind="videos"),
        ]
        for call in calls:
            call()
            path = recorder.last.url.path.removeprefix("/api/vera/v1")
            self._check(path, recorder.body)
