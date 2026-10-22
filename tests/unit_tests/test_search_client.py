"""VeraClient's search tools: results page, keyword ideas, AI answers, query radar, footprint."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from langchain_vera import (
    VeraClient,
    VeraConflictError,
    VeraInsufficientCreditsError,
    VeraRateLimitError,
    VeraRequestError,
    VeraServerError,
    VeraTimeoutError,
)
from langchain_vera._client import (
    build_ai_answers_request,
    build_company_footprint_request,
    build_geo_radar_request,
    build_keyword_ideas_request,
    build_serp_request,
)
from tests.unit_tests.conftest import Recorder, failure, make_client, success

SNAPSHOT = json.loads(
    (Path(__file__).parent.parent / "fixtures" / "openapi-request-schemas.json").read_text()
)
TOOLS = SNAPSHOT["tools"]


class TestSerpRequest:
    def test_default_request(self) -> None:
        client, recorder = make_client(success({"organic": []}, charged=3))
        client.serp(" best coffee in dubai ")
        assert recorder.last.method == "POST"
        assert str(recorder.last.url) == "https://vera.waggle.ae/v1/serp/google"
        assert recorder.body == {"q": "best coffee in dubai", "depth": 10}

    @pytest.mark.parametrize("engine", ["google", "bing", "duckduckgo", "brave", "yahoo"])
    def test_every_engine_has_its_own_path(self, engine: str) -> None:
        client, recorder = make_client()
        client.serp("q", engine=engine)  # type: ignore[arg-type]
        assert recorder.last.url.path == f"/v1/serp/{engine}"

    @pytest.mark.parametrize("engine", ["ask", "Google", "../usage", "google/../../x", ""])
    def test_an_unknown_engine_never_reaches_the_url(self, engine: str) -> None:
        client, recorder = make_client()
        with pytest.raises(ValueError, match="engine must be one of"):
            client.serp("q", engine=engine)  # type: ignore[arg-type]
        assert recorder.requests == []

    def test_market_depth_and_cache_options(self) -> None:
        client, recorder = make_client()
        client.serp("coffee shop", gl="AE", hl="EN", location=" Dubai ", depth=20, fresh=True)
        assert recorder.body == {
            "q": "coffee shop",
            "depth": 20,
            "gl": "ae",
            "hl": "en",
            "location": "Dubai",
            "fresh": True,
        }

    def test_a_regional_language_is_accepted(self) -> None:
        assert build_serp_request("q", hl="pt-BR")["hl"] == "pt-br"
        assert build_ai_answers_request("p", "b", language="zh-hans-cn")["language"] == "zh-hans-cn"

    def test_features_default_to_none_so_the_api_default_applies(self) -> None:
        assert "features" not in build_serp_request("q")

    def test_features_add_the_local_pack_and_shopping(self) -> None:
        body = build_serp_request(
            "q", features=["organic", "paa", "ai_overview", "places", "shopping", "shopping"]
        )
        assert body["features"] == ["organic", "paa", "ai_overview", "local_pack", "shopping"]

    @pytest.mark.parametrize("features", [["maps"], [], ["organic", "video"]])
    def test_bad_features_are_refused(self, features: list[str]) -> None:
        with pytest.raises(ValueError, match="feature"):
            build_serp_request("q", features=features)

    def test_the_add_ons_are_google_only(self) -> None:
        with pytest.raises(ValueError, match="Google add-ons"):
            build_serp_request("q", engine="bing", features=["organic", "local_pack"])
        assert build_serp_request("q", engine="bing", features=["organic"])["features"] == [
            "organic"
        ]

    @pytest.mark.parametrize("depth", [0, 101, -5, 2.5, True, "10"])
    def test_depth_bounds(self, depth: Any) -> None:
        with pytest.raises(ValueError, match="depth must be a whole number from 1 to 100"):
            build_serp_request("q", depth=depth)

    def test_the_extremes_are_accepted(self) -> None:
        assert build_serp_request("q", depth=1)["depth"] == 1
        assert build_serp_request("q", depth=100)["depth"] == 100

    @pytest.mark.parametrize("query", ["", "   ", "a" * 501, "a\x07b"])
    def test_query_validation(self, query: str) -> None:
        with pytest.raises(ValueError, match="q "):
            build_serp_request(query)

    @pytest.mark.parametrize(
        ("gl", "hl"), [("usa", None), ("1a", None), (None, "e"), (None, "e_n")]
    )
    def test_bad_market_codes(self, gl: str | None, hl: str | None) -> None:
        with pytest.raises(ValueError, match=r"gl|hl"):
            build_serp_request("q", gl=gl, hl=hl)

    def test_the_idempotency_key_is_a_header(self) -> None:
        client, recorder = make_client()
        client.serp("q", idempotency_key="report-2026-01-01-serp-0001")
        assert recorder.last.headers["idempotency-key"] == "report-2026-01-01-serp-0001"
        assert "idempotency_key" not in recorder.body

    async def test_async(self) -> None:
        client, recorder = make_client(success({"organic": []}, charged=1))
        response = await client.aserp("q", engine="duckduckgo", depth=30)
        assert recorder.last.url.path == "/v1/serp/duckduckgo"
        assert recorder.body["depth"] == 30
        assert response.credits_used == 1


class TestKeywordIdeasRequest:
    def test_limit_and_expand_are_always_sent(self) -> None:
        client, recorder = make_client(success({"ideas": []}, charged=30))
        client.keyword_ideas(" specialty coffee ")
        assert recorder.last.url.path == "/v1/keywords/ideas"
        assert recorder.body == {"seed": "specialty coffee", "limit": 50, "expand": "questions"}

    def test_options(self) -> None:
        body = build_keyword_ideas_request(
            "قهوة مختصة", gl="AE", hl="ar", location="Dubai", limit=200, expand="all", fresh=True
        )
        assert body == {
            "seed": "قهوة مختصة",
            "limit": 200,
            "expand": "all",
            "gl": "ae",
            "hl": "ar",
            "location": "Dubai",
            "fresh": True,
        }

    @pytest.mark.parametrize("expand", ["none", "questions", "alphabet", "all"])
    def test_expansions(self, expand: str) -> None:
        assert build_keyword_ideas_request("s", expand=expand)["expand"] == expand  # type: ignore[arg-type]

    def test_rejects_bad_input(self) -> None:
        with pytest.raises(ValueError, match="expand must be one of"):
            build_keyword_ideas_request("s", expand="everything")  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="seed is required"):
            build_keyword_ideas_request(" ")
        for limit in (0, 201, 1.5):
            with pytest.raises(ValueError, match="limit must be a whole number from 1 to 200"):
                build_keyword_ideas_request("s", limit=limit)  # type: ignore[arg-type]

    async def test_async(self) -> None:
        client, recorder = make_client()
        await client.akeyword_ideas("s", expand="none", limit=5)
        assert recorder.body == {"seed": "s", "limit": 5, "expand": "none"}


class TestAiAnswersRequest:
    def test_asks_one_assistant_once_by_default_and_never_leaves_engines_out(self) -> None:
        client, recorder = make_client(success({"engines": []}, charged=200))
        client.ai_answers("Best roasters in Dubai?", "Roasters")
        assert recorder.last.url.path == "/v1/ai/answers"
        assert recorder.body == {
            "prompt": "Best roasters in Dubai?",
            "brand": "Roasters",
            "engines": ["chatgpt"],
            "samples": 1,
        }

    def test_every_field(self) -> None:
        body = build_ai_answers_request(
            "p",
            "b",
            brand_domain=" b.com ",
            competitors=["Other", "rivals.ae", "other", " "],
            engines=["chatgpt", "Claude", "gemini", "claude"],
            samples=3,
            country="AE",
            language="AR",
            city="Dubai",
        )
        assert body == {
            "prompt": "p",
            "brand": "b",
            "engines": ["chatgpt", "claude", "gemini"],
            "samples": 3,
            "brand_domain": "b.com",
            "competitors": ["Other", "rivals.ae"],
            "country": "ae",
            "language": "ar",
            "city": "Dubai",
        }

    @pytest.mark.parametrize("engines", [[], (), "chatgpt"])
    def test_an_empty_engine_list_is_refused_because_the_api_would_ask_them_all(
        self, engines: Any
    ) -> None:
        with pytest.raises(ValueError, match="engines must be a non-empty list"):
            build_ai_answers_request("p", "b", engines=engines)

    def test_unknown_engines_are_refused_and_perplexity_is_allowed(self) -> None:
        with pytest.raises(ValueError, match="Unknown engine"):
            build_ai_answers_request("p", "b", engines=["bard"])
        assert build_ai_answers_request("p", "b", engines=["perplexity"])["engines"] == [
            "perplexity"
        ]

    @pytest.mark.parametrize("samples", [0, 6, 1.5, True])
    def test_samples_bounds(self, samples: Any) -> None:
        with pytest.raises(ValueError, match="samples must be a whole number from 1 to 5"):
            build_ai_answers_request("p", "b", samples=samples)

    def test_required_fields_and_limits(self) -> None:
        with pytest.raises(ValueError, match="prompt is required"):
            build_ai_answers_request(" ", "b")
        with pytest.raises(ValueError, match="brand is required"):
            build_ai_answers_request("p", "")
        with pytest.raises(ValueError, match="prompt must be at most 1000"):
            build_ai_answers_request("p" * 1001, "b")
        with pytest.raises(ValueError, match="brand must be at most 200"):
            build_ai_answers_request("p", "b" * 201)

    def test_competitor_limits(self) -> None:
        twenty = [f"rival {n}" for n in range(20)]
        assert len(build_ai_answers_request("p", "b", competitors=twenty)["competitors"]) == 20
        with pytest.raises(ValueError, match="At most 20 competitors"):
            build_ai_answers_request("p", "b", competitors=[*twenty, "one more"])
        with pytest.raises(ValueError, match="longer than 200"):
            build_ai_answers_request("p", "b", competitors=["c" * 201])

    def test_optional_fields_are_left_out_when_empty(self) -> None:
        body = build_ai_answers_request("p", "b", brand_domain="", competitors=[], city=" ")
        assert set(body) == {"prompt", "brand", "engines", "samples"}

    async def test_async(self) -> None:
        client, recorder = make_client()
        await client.aai_answers("p", "b", engines=["claude"], samples=2)
        assert recorder.body["engines"] == ["claude"]
        assert recorder.body["samples"] == 2


class TestRadarAndFootprintRequests:
    def test_geo_radar(self) -> None:
        client, recorder = make_client()
        client.geo_radar_query(
            "best roasters in dubai",
            brand="Roasters",
            domain="roasters.ae",
            location="Dubai",
            gl="ae",
            hl="en",
            depth=40,
            fresh=True,
        )
        assert recorder.last.url.path == "/v1/geo/radar/query"
        assert recorder.body == {
            "query": "best roasters in dubai",
            "depth": 40,
            "brand": "Roasters",
            "domain": "roasters.ae",
            "location": "Dubai",
            "gl": "ae",
            "hl": "en",
            "fresh": True,
        }

    def test_geo_radar_defaults_to_depth_30(self) -> None:
        assert build_geo_radar_request("q") == {"query": "q", "depth": 30}

    def test_company_footprint(self) -> None:
        client, recorder = make_client()
        client.company_footprint("Roasters", domain="roasters.ae", location="Dubai", gl="ae")
        assert recorder.last.url.path == "/v1/company/footprint"
        assert recorder.body == {
            "company": "Roasters",
            "depth": 20,
            "domain": "roasters.ae",
            "location": "Dubai",
            "gl": "ae",
        }

    @pytest.mark.parametrize("depth", [0, 101])
    def test_depth_bounds(self, depth: int) -> None:
        with pytest.raises(ValueError, match="depth must be a whole number from 1 to 100"):
            build_geo_radar_request("q", depth=depth)
        with pytest.raises(ValueError, match="depth must be a whole number from 1 to 100"):
            build_company_footprint_request("c", depth=depth)

    def test_required_text(self) -> None:
        with pytest.raises(ValueError, match="query is required"):
            build_geo_radar_request("")
        with pytest.raises(ValueError, match="company is required"):
            build_company_footprint_request("  ")

    async def test_async(self) -> None:
        client, recorder = make_client()
        await client.ageo_radar_query("q", depth=10)
        assert recorder.last.url.path == "/v1/geo/radar/query"
        await client.acompany_footprint("c", depth=10)
        assert recorder.last.url.path == "/v1/company/footprint"


class TestSearchToolResponses:
    def test_the_whole_envelope_is_kept(self) -> None:
        client, _ = make_client(
            success(
                {
                    "status": "success",
                    "ideas": [{"keyword": "k", "sources": [], "intent": "local"}],
                },
                charged=30,
                headers={"x-waggle-credits-used": "6"},
            )
        )
        response = client.keyword_ideas("s")
        assert response.data["ideas"][0]["keyword"] == "k"
        assert response.credits_used == 6  # the header is the charge: it was cheaper than the hold
        assert response.request_id == "req_test"
        assert response.elapsed_ms == 123

    def test_a_replay_is_free(self) -> None:
        client, _ = make_client(
            success(
                {"engines": []},
                charged=200,
                headers={
                    "x-waggle-credits-used": "0",
                    "x-waggle-idempotency-replayed": "true",
                },
            )
        )
        response = client.ai_answers("p", "b", idempotency_key="report-2026-01-01-answer-0001")
        assert response.credits_used == 0
        assert response.idempotency_replayed is True


class TestSearchToolErrors:
    def test_402_names_the_pricing_page(self) -> None:
        client, _ = make_client(
            failure(
                402, "This request requires 200 credits.", headers={"x-waggle-credits-used": "0"}
            )
        )
        with pytest.raises(VeraInsufficientCreditsError) as caught:
            client.ai_answers("p", "b")
        assert "https://vera.waggle.ae/pricing" in str(caught.value)
        assert "This request requires 200 credits." in str(caught.value)
        assert caught.value.credits_used == 0

    @pytest.mark.parametrize("status", [400, 422])
    def test_invalid_input_carries_the_message(self, status: int) -> None:
        client, _ = make_client(
            failure(
                status,
                "depth: Input should be less than or equal to 100",
                code="invalid_request",
                headers={"x-waggle-credits-used": "0"},
            )
        )
        with pytest.raises(VeraRequestError) as caught:
            client.serp("q")
        text = str(caught.value)
        assert text.startswith("Invalid input")
        assert "depth: Input should be less than or equal to 100" in text
        assert "No credits were charged." in text

    def test_503_is_temporary_and_not_charged(self) -> None:
        client, _ = make_client(
            failure(
                503,
                "This tool is temporarily unavailable. You were not charged.",
                headers={"retry-after": "30", "x-waggle-credits-used": "0"},
            )
        )
        with pytest.raises(VeraServerError) as caught:
            client.keyword_ideas("s")
        text = str(caught.value)
        assert "Vera is temporarily unavailable" in text
        assert "Retry after 30 seconds." in text
        assert "No credits were charged." in text
        assert caught.value.retry_after == 30.0
        assert caught.value.status_code == 503

    def test_504_is_a_timeout_and_not_charged(self) -> None:
        client, _ = make_client(
            failure(
                504,
                "The service behind this tool took too long to answer. You were not charged.",
                headers={"x-waggle-credits-used": "0"},
            )
        )
        with pytest.raises(VeraServerError) as caught:
            client.ai_answers("p", "b")
        text = str(caught.value)
        assert caught.value.status_code == 504
        assert "timed out" in text
        assert "fewer samples" in text
        assert "No credits were charged." in text

    @pytest.mark.parametrize("status", [503, 504])
    def test_503_and_504_are_documented_as_not_charged_even_without_the_header(
        self, status: int
    ) -> None:
        client, _ = make_client(failure(status, "x"))
        with pytest.raises(VeraServerError, match=r"No credits were charged\."):
            client.company_footprint("c")

    @pytest.mark.parametrize("status", [503, 504])
    def test_but_not_when_the_header_shows_a_charge(self, status: int) -> None:
        client, _ = make_client(failure(status, "x", headers={"x-waggle-credits-used": "3"}))
        with pytest.raises(VeraServerError) as caught:
            client.company_footprint("c")
        assert "No credits were charged." not in str(caught.value)

    def test_429_and_409_keep_their_types(self) -> None:
        client, _ = make_client(
            failure(429, "Daily free limit reached.", headers={"retry-after": "3600"})
        )
        with pytest.raises(VeraRateLimitError) as limited:
            client.ai_answers("p", "b", engines=["claude"])
        assert limited.value.retry_after == 3600.0
        client, _ = make_client(failure(409, "Idempotency-Key was already used."))
        with pytest.raises(VeraConflictError):
            client.serp("q", idempotency_key="report-2026-01-01-serp-0002")

    async def test_async_errors_are_typed_too(self) -> None:
        client, _ = make_client(failure(504, "slow"))
        with pytest.raises(VeraServerError, match="timed out"):
            await client.aai_answers("p", "b")


def seen_timeout(recorder: Recorder) -> dict[str, float | None]:
    """The timeout httpx applied to the last request."""
    return dict(recorder.last.extensions["timeout"])


class TestTimeouts:
    """The long calls get the time the API takes; the rest keep the default."""

    CASES = (
        ("serp", ("q",), 50.0),
        ("keyword_ideas", ("s",), 80.0),
        ("geo_radar_query", ("q",), 80.0),
        ("company_footprint", ("c",), 65.0),
        ("ai_answers", ("p", "b"), 175.0),
    )

    @pytest.mark.parametrize(("method", "args", "seconds"), CASES)
    def test_a_client_you_pass_in_is_asked_for_the_long_timeout(
        self, method: str, args: tuple[str, ...], seconds: float
    ) -> None:
        client, recorder = make_client()
        getattr(client, method)(*args)
        assert seen_timeout(recorder)["read"] == seconds
        assert seen_timeout(recorder)["connect"] == 10.0

    def test_ai_answers_wait_longer_than_the_150_seconds_the_api_may_take(self) -> None:
        client, recorder = make_client()
        client.ai_answers("p", "b")
        assert (seen_timeout(recorder)["read"] or 0) > 150.0

    def test_page_and_search_calls_keep_the_default(self) -> None:
        client, recorder = make_client()
        client.scrape("https://example.com")
        assert seen_timeout(recorder)["read"] == 5.0  # httpx's own default: not overridden

    def test_with_your_own_client_an_explicit_timeout_is_left_to_that_client(self) -> None:
        client, recorder = make_client(timeout=7)
        client.ai_answers("p", "b")
        # The temporary-client path is covered below; with your own client the explicit value is
        # left to that client, as for every other call.
        assert seen_timeout(recorder)["read"] == 5.0

    @pytest.mark.parametrize(("method", "args", "seconds"), CASES)
    def test_the_temporary_client_gets_the_long_timeout(
        self,
        monkeypatch: pytest.MonkeyPatch,
        method: str,
        args: tuple[str, ...],
        seconds: float,
    ) -> None:
        seen: dict[str, Any] = {}
        real = httpx.Client

        def spy(**kwargs: Any) -> httpx.Client:
            seen.update(kwargs)
            return real(transport=httpx.MockTransport(lambda r: success()))

        monkeypatch.setattr("langchain_vera._client.httpx.Client", spy)
        getattr(VeraClient("vera_live_x"), method)(*args)
        assert seen["timeout"] == httpx.Timeout(seconds, connect=10.0)

    def test_an_explicit_timeout_beats_the_long_default_on_the_temporary_client(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seen: dict[str, Any] = {}
        real = httpx.Client

        def spy(**kwargs: Any) -> httpx.Client:
            seen.update(kwargs)
            return real(transport=httpx.MockTransport(lambda r: success()))

        monkeypatch.setattr("langchain_vera._client.httpx.Client", spy)
        VeraClient("vera_live_x", timeout=7).ai_answers("p", "b")
        assert seen["timeout"] == httpx.Timeout(7)

    async def test_the_async_temporary_client_gets_the_long_timeout(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seen: dict[str, Any] = {}
        real = httpx.AsyncClient

        def spy(**kwargs: Any) -> httpx.AsyncClient:
            seen.update(kwargs)
            return real(transport=httpx.MockTransport(lambda r: success()))

        monkeypatch.setattr("langchain_vera._client.httpx.AsyncClient", spy)
        await VeraClient("vera_live_x").aai_answers("p", "b")
        assert seen["timeout"] == httpx.Timeout(175.0, connect=10.0)

    def test_a_timeout_is_a_typed_error(self) -> None:
        def stall(request: httpx.Request) -> httpx.Response:
            raise httpx.ReadTimeout("too slow", request=request)

        client, _ = make_client(stall)
        with pytest.raises(VeraTimeoutError):
            client.ai_answers("p", "b")


class TestContractWithPublishedExamples:
    """The documented example of each route is what this client sends, and nothing else."""

    @staticmethod
    def sent(call: Any) -> dict[str, Any]:
        client, recorder = make_client()
        call(client)
        return recorder.body

    def test_serp_examples(self) -> None:
        examples = TOOLS["/v1/serp/{engine}"]["examples"]
        assert (
            self.sent(lambda c: c.serp("best coffee in dubai", gl="ae", hl="en", depth=10))
            == examples["search"]
        )
        assert (
            self.sent(
                lambda c: c.serp(
                    "coffee shop",
                    location="Dubai",
                    gl="ae",
                    hl="en",
                    depth=10,
                    features=["organic", "paa", "ai_overview", "local_pack"],
                )
            )
            == examples["with_local_pack"]
        )

    def test_keyword_example(self) -> None:
        example = TOOLS["/v1/keywords/ideas"]["examples"]["ideas"]
        assert (
            self.sent(
                lambda c: c.keyword_ideas(
                    "specialty coffee", gl="ae", hl="en", limit=50, expand="questions"
                )
            )
            == example
        )

    def test_ai_answers_example(self) -> None:
        example = TOOLS["/v1/ai/answers"]["examples"]["answer"]
        assert (
            self.sent(
                lambda c: c.ai_answers(
                    "What are the best specialty coffee roasters in Dubai?",
                    "Roasters",
                    brand_domain="roasters.ae",
                    competitors=["Other Roasters"],
                    country="ae",
                    language="en",
                    samples=1,
                    engines=["chatgpt"],
                )
            )
            == example
        )

    def test_radar_and_footprint_examples(self) -> None:
        radar = TOOLS["/v1/geo/radar/query"]["examples"]["radar"]
        assert (
            self.sent(
                lambda c: c.geo_radar_query(
                    "best specialty coffee roasters in dubai",
                    brand="Roasters",
                    domain="roasters.ae",
                    location="Dubai",
                    gl="ae",
                    hl="en",
                    depth=30,
                )
            )
            == radar
        )
        footprint = TOOLS["/v1/company/footprint"]["examples"]["footprint"]
        assert (
            self.sent(
                lambda c: c.company_footprint(
                    "Roasters",
                    domain="roasters.ae",
                    location="Dubai",
                    gl="ae",
                    hl="en",
                    depth=20,
                )
            )
            == footprint
        )

    def test_only_field_names_the_services_accept_are_sent(self) -> None:
        bodies = {
            "/v1/serp/{engine}": build_serp_request(
                "q",
                gl="us",
                hl="en",
                location="Austin",
                depth=100,
                features=["organic", "local_pack", "shopping"],
                fresh=True,
            ),
            "/v1/keywords/ideas": build_keyword_ideas_request(
                "s", gl="ae", hl="ar", location="Dubai", limit=200, expand="all", fresh=True
            ),
            "/v1/ai/answers": build_ai_answers_request(
                "p",
                "b",
                brand_domain="b.com",
                competitors=["c"],
                engines=["chatgpt", "claude", "gemini"],
                samples=5,
                country="ae",
                language="en",
                city="Dubai",
            ),
            "/v1/geo/radar/query": build_geo_radar_request(
                "q", brand="b", domain="b.com", location="Dubai", gl="ae", hl="en", fresh=True
            ),
            "/v1/company/footprint": build_company_footprint_request(
                "c", domain="c.com", location="Dubai", gl="ae", hl="en", fresh=True
            ),
        }
        for route, body in bodies.items():
            for key in body:
                assert key in TOOLS[route]["knownFields"], f"{key} is not a field of {route}"

    def test_the_snapshot_lists_every_route(self) -> None:
        assert set(TOOLS) == {
            "/v1/serp/{engine}",
            "/v1/keywords/ideas",
            "/v1/ai/answers",
            "/v1/geo/radar/query",
            "/v1/company/footprint",
        }
