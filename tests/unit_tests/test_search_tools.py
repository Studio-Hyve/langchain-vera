"""The search intelligence tools, with HTTP mocked."""

from __future__ import annotations

from typing import Any

import httpx
import pytest
from langchain_core.messages import ToolMessage
from langchain_core.tools import ToolException
from pydantic import ValidationError

from langchain_vera import (
    VeraAiAnswersTool,
    VeraCompanyFootprintTool,
    VeraGeoRadarTool,
    VeraInsufficientCreditsError,
    VeraKeywordIdeasTool,
    VeraRateLimitError,
    VeraSerpTool,
    VeraServerError,
)
from langchain_vera.tools import format_ai_answers, format_keyword_ideas
from tests.unit_tests.conftest import failure, make_client, success

SERP = {
    "engine": "google",
    "status": "success",
    "query": "best coffee in dubai",
    "depth": {"requested": 10, "delivered": 2, "has_more": False},
    "organic": [
        {
            "position": 1,
            "title": "Best Coffee in Dubai",
            "link": "https://example.com/coffee",
            "url": "https://example.com/coffee",
            "snippet": "A guide to the city's roasters.",
            "domain": "example.com",
        },
        {
            "position": 2,
            "title": "Roasters",
            "link": "https://roasters.ae/",
            "url": "https://roasters.ae/",
            "snippet": "Specialty coffee.",
        },
    ],
    "llm_markdown": (
        "# SERP Brief: best coffee in dubai\n\n## Top Organic Rankings\n### 1. Best Coffee"
    ),
    "provenance": {"credits_used": 3, "cache_hit": False},
}

IDEAS = {
    "status": "success",
    "seed": "specialty coffee",
    "count": 4,
    "ideas": [
        {
            "keyword": "how to brew specialty coffee",
            "sources": ["autocomplete"],
            "intent": "question",
        },
        {
            "keyword": "best specialty coffee beans",
            "sources": ["autocomplete"],
            "intent": "commercial",
        },
        {
            "keyword": "specialty coffee vs regular",
            "sources": ["related_searches"],
            "intent": "comparison",
        },
        {"keyword": "specialty coffee dubai", "sources": ["people_also_ask"], "intent": "local"},
    ],
    "notes": [],
    "provenance": {"credits_used": 30},
}

ANSWERS = {
    "status": "success",
    "label": "AI answers",
    "prompt": "What are the best specialty coffee roasters in Dubai?",
    "brand": "Roasters",
    "brand_domain": "roasters.ae",
    "competitors": ["Other Roasters"],
    "engines": [
        {
            "engine": "chatgpt",
            "status": "ok",
            "model": "gpt",
            "aggregate": {
                "samples_ok": 2,
                "samples_requested": 2,
                "samples_failed": 0,
                "mentions": 1,
                "mention_rate": 0.5,
                "share_of_voice": 0.4,
                "first_mention_rank_avg": 2.0,
                "brand_domain_cited_rate": 0.5,
                "total_citations": 8,
                "competitors": [
                    {
                        "name": "Other Roasters",
                        "mentions": 2,
                        "mention_rate": 1.0,
                        "first_mention_rank_avg": 1.0,
                    }
                ],
                "citation_share_by_domain": {"example.com": 0.5, "roasters.ae": 0.25},
            },
            "samples": [],
        },
        {"engine": "claude", "status": "not_configured", "aggregate": None, "samples": []},
        {
            "engine": "gemini",
            "status": "error",
            "aggregate": {"samples_ok": 0},
            "samples": [{"status": "error", "error": {"code": "timeout", "message": "No answer."}}],
        },
    ],
    "notes": ["One assistant is not set up."],
}

RADAR = {
    "engine": "google",
    "query": "best roasters",
    "citation_probability_score": 42,
    "llm_markdown": "# GEO Visibility Radar: best roasters\n- Citation probability: 42",
}

FOOTPRINT = {
    "company": "Roasters",
    "owned_properties": [],
    "llm_markdown": "# Company footprint: Roasters\n## Owned properties",
}


def call(name: str, **args: Any) -> dict[str, Any]:
    """A `ToolCall`, which makes `invoke` return a ToolMessage with the artifact."""
    return {"type": "tool_call", "name": name, "id": "call_1", "args": args}


class TestSerpTool:
    def test_returns_the_brief_and_the_artifact_with_the_charge(self) -> None:
        client, recorder = make_client(success(SERP, charged=3))
        message = VeraSerpTool(client=client).invoke(
            call("vera_serp", query="best coffee in dubai")
        )
        assert isinstance(message, ToolMessage)
        assert str(message.content).startswith("# SERP Brief: best coffee in dubai")
        assert message.artifact["organic"][0]["title"] == "Best Coffee in Dubai"
        assert message.artifact["credits_used"] == 3
        assert message.artifact["request_id"] == "req_test"
        assert recorder.last.url.path == "/v1/serp/google"
        assert recorder.body == {"q": "best coffee in dubai", "depth": 10}

    def test_settings_are_sent_and_the_model_only_chooses_the_query(self) -> None:
        client, recorder = make_client(success(SERP, charged=6))
        tool = VeraSerpTool(
            client=client,
            engine="bing",
            gl="ae",
            hl="ar",
            location="Dubai",
            depth=20,
            fresh=True,
        )
        assert set(tool.args) == {"query"}
        tool.invoke({"query": "قهوة"})
        assert recorder.last.url.path == "/v1/serp/bing"
        assert recorder.body == {
            "q": "قهوة",
            "depth": 20,
            "gl": "ae",
            "hl": "ar",
            "location": "Dubai",
            "fresh": True,
        }

    def test_add_ons_become_features_on_top_of_the_defaults(self) -> None:
        client, recorder = make_client(success(SERP, charged=9))
        VeraSerpTool(client=client, local_pack=True, shopping=True).invoke({"query": "coffee"})
        assert recorder.body["features"] == [
            "organic",
            "paa",
            "ai_overview",
            "local_pack",
            "shopping",
        ]
        VeraSerpTool(client=client, local_pack=True).invoke({"query": "coffee"})
        assert recorder.body["features"] == ["organic", "paa", "ai_overview", "local_pack"]
        VeraSerpTool(client=client).invoke({"query": "coffee"})
        assert "features" not in recorder.body

    def test_add_ons_on_another_engine_fail_at_construction(self) -> None:
        with pytest.raises(ValidationError, match="Google add-ons"):
            VeraSerpTool(api_key="vera_live_test", engine="bing", local_pack=True)

    @pytest.mark.parametrize(
        "kwargs",
        [{"depth": 0}, {"depth": 101}, {"gl": "USA"}, {"hl": "e"}, {"engine": "ask"}],
    )
    def test_settings_are_validated_at_construction(self, kwargs: dict[str, Any]) -> None:
        with pytest.raises(ValidationError):
            VeraSerpTool(api_key="vera_live_test", **kwargs)

    def test_max_characters_truncates_with_a_notice(self) -> None:
        client, _ = make_client(success({**SERP, "llm_markdown": "x" * 500}, charged=3))
        text = VeraSerpTool(client=client, max_characters=100).invoke({"query": "q"})
        assert text.startswith("x" * 100)
        assert "[Truncated: showing the first 100 of 500 characters.]" in text

    def test_without_a_brief_it_lists_the_organic_results(self) -> None:
        data = {key: value for key, value in SERP.items() if key != "llm_markdown"}
        client, _ = make_client(success(data, charged=3))
        text = VeraSerpTool(client=client).invoke({"query": "q"})
        assert text.splitlines()[0] == "1. Best Coffee in Dubai"
        assert "https://example.com/coffee" in text

    def test_no_results(self) -> None:
        client, _ = make_client(success({"organic": []}, charged=3))
        assert VeraSerpTool(client=client).invoke({"query": "zzz"}) == "No results found for 'zzz'."

    async def test_async(self) -> None:
        client, recorder = make_client(success(SERP, charged=3))
        message = await VeraSerpTool(client=client, engine="yahoo").ainvoke(
            call("vera_serp", query="vera")
        )
        assert message.artifact["credits_used"] == 3
        assert recorder.last.url.path == "/v1/serp/yahoo"

    def test_a_refused_request_goes_back_to_the_model(self) -> None:
        client, _ = make_client(
            failure(422, "depth: too large", headers={"x-waggle-credits-used": "0"})
        )
        message = VeraSerpTool(client=client).invoke(call("vera_serp", query="q"))
        assert message.status == "error"
        assert "depth: too large" in str(message.content)
        assert "No credits were charged." in str(message.content)

    def test_the_timeout_default_is_long_enough_for_a_results_page(self) -> None:
        tool = VeraSerpTool(api_key="vera_live_test")
        assert tool.timeout == 50.0
        assert tool._vera.timeout == httpx.Timeout(50.0, connect=10.0)


class TestKeywordIdeasTool:
    def test_groups_ideas_by_intent(self) -> None:
        client, recorder = make_client(success(IDEAS, charged=30))
        message = VeraKeywordIdeasTool(client=client).invoke(
            call("vera_keyword_ideas", seed="specialty coffee")
        )
        assert message.content == (
            "Keyword ideas for 'specialty coffee' (4 ideas)\n"
            "\nquestion:\n- how to brew specialty coffee"
            "\n\ncomparison:\n- specialty coffee vs regular"
            "\n\ncommercial:\n- best specialty coffee beans"
            "\n\nlocal:\n- specialty coffee dubai"
        )
        assert message.artifact["ideas"][0]["intent"] == "question"
        assert message.artifact["credits_used"] == 30
        assert recorder.last.url.path == "/v1/keywords/ideas"
        assert recorder.body == {"seed": "specialty coffee", "limit": 50, "expand": "questions"}

    def test_settings_are_sent(self) -> None:
        client, recorder = make_client(success(IDEAS, charged=6))
        tool = VeraKeywordIdeasTool(
            client=client, gl="ae", hl="ar", location="Dubai", limit=10, expand="none", fresh=True
        )
        assert set(tool.args) == {"seed"}
        tool.invoke({"seed": "قهوة"})
        assert recorder.body == {
            "seed": "قهوة",
            "limit": 10,
            "expand": "none",
            "gl": "ae",
            "hl": "ar",
            "location": "Dubai",
            "fresh": True,
        }

    @pytest.mark.parametrize(
        "kwargs", [{"limit": 0}, {"limit": 201}, {"expand": "everything"}, {"gl": "ae1"}]
    )
    def test_settings_are_validated_at_construction(self, kwargs: dict[str, Any]) -> None:
        with pytest.raises(ValidationError):
            VeraKeywordIdeasTool(api_key="vera_live_test", **kwargs)

    def test_no_ideas(self) -> None:
        client, _ = make_client(success({"ideas": []}, charged=3))
        assert (
            VeraKeywordIdeasTool(client=client).invoke({"seed": "zzz"})
            == "No keyword ideas found for 'zzz'."
        )

    def test_a_partial_answer_shows_its_notes(self) -> None:
        text = format_keyword_ideas({**IDEAS, "status": "partial", "notes": ["Budget ran out."]})
        assert text.endswith("Notes: Budget ran out.")

    async def test_async(self) -> None:
        client, recorder = make_client(success(IDEAS, charged=30))
        message = await VeraKeywordIdeasTool(client=client, expand="all").ainvoke(
            call("vera_keyword_ideas", seed="s")
        )
        assert message.artifact["count"] == 4
        assert recorder.body["expand"] == "all"

    def test_max_characters(self) -> None:
        client, _ = make_client(success(IDEAS, charged=30))
        text = VeraKeywordIdeasTool(client=client, max_characters=40).invoke({"seed": "s"})
        assert "[Truncated" in text

    def test_the_timeout_default(self) -> None:
        assert VeraKeywordIdeasTool(api_key="vera_live_test").timeout == 80.0


class TestAiAnswersTool:
    def test_summarises_each_assistant(self) -> None:
        client, recorder = make_client(success(ANSWERS, charged=400))
        message = VeraAiAnswersTool(
            client=client,
            brand="Roasters",
            brand_domain="roasters.ae",
            competitors=["Other Roasters"],
        ).invoke(call("vera_ai_answers", prompt="What are the best specialty coffee roasters?"))
        text = str(message.content)
        assert text.startswith(
            "AI answers for 'What are the best specialty coffee roasters in Dubai?'"
        )
        assert (
            "- ChatGPT: Roasters named in 1 of 2 answers (mention rate 50%, share of voice 40%, "
            "average first rank 2, brand site cited in 50% of answers, 8 citations)"
        ) in text
        assert "Competitors: Other Roasters: named in 2 of 2, average rank 1" in text
        assert "Most cited domains: example.com 50%, roasters.ae 25%" in text
        assert "- Claude: not set up, so it did not answer" in text
        assert "- Gemini: did not answer (No answer.)." in text
        assert text.endswith("Notes: One assistant is not set up.")
        assert message.artifact["credits_used"] == 400
        assert message.artifact["engines"][0]["engine"] == "chatgpt"
        assert recorder.last.url.path == "/v1/ai/answers"
        assert recorder.body == {
            "prompt": "What are the best specialty coffee roasters?",
            "brand": "Roasters",
            "engines": ["chatgpt"],
            "samples": 1,
            "brand_domain": "roasters.ae",
            "competitors": ["Other Roasters"],
        }

    def test_a_default_run_asks_one_assistant_once(self) -> None:
        tool = VeraAiAnswersTool(api_key="vera_live_test")
        assert tool.engines == ["chatgpt"]
        assert tool.samples == 1

    def test_the_model_can_give_the_brand_competitors_and_domain(self) -> None:
        client, recorder = make_client(success(ANSWERS, charged=200))
        tool = VeraAiAnswersTool(
            client=client, brand="Default Brand", competitors=["Default Rival"]
        )
        tool.invoke(
            {
                "prompt": "p",
                "brand": "Asked Brand",
                "brand_domain": "asked.com",
                "competitors": ["Asked Rival"],
            }
        )
        assert recorder.body["brand"] == "Asked Brand"
        assert recorder.body["brand_domain"] == "asked.com"
        assert recorder.body["competitors"] == ["Asked Rival"]

    def test_settings_are_sent(self) -> None:
        client, recorder = make_client(success(ANSWERS, charged=200))
        VeraAiAnswersTool(
            client=client,
            brand="B",
            engines=["claude", "gemini"],
            samples=3,
            country="ae",
            language="ar",
            city="Dubai",
        ).invoke({"prompt": "p"})
        assert recorder.body["engines"] == ["claude", "gemini"]
        assert recorder.body["samples"] == 3
        assert (recorder.body["country"], recorder.body["language"]) == ("ae", "ar")
        assert recorder.body["city"] == "Dubai"

    def test_no_brand_comes_back_to_the_model_without_a_request(self) -> None:
        client, recorder = make_client()
        message = VeraAiAnswersTool(client=client).invoke(call("vera_ai_answers", prompt="p"))
        assert message.status == "error"
        assert "No brand was given" in str(message.content)
        assert recorder.requests == []

    def test_an_empty_engine_list_is_refused_at_construction(self) -> None:
        with pytest.raises(ValidationError, match="engines must not be empty"):
            VeraAiAnswersTool(api_key="vera_live_test", engines=[])

    @pytest.mark.parametrize(
        "kwargs", [{"engines": ["bard"]}, {"samples": 0}, {"samples": 6}, {"country": "ARE"}]
    )
    def test_settings_are_validated_at_construction(self, kwargs: dict[str, Any]) -> None:
        with pytest.raises(ValidationError):
            VeraAiAnswersTool(api_key="vera_live_test", **kwargs)

    def test_the_description_warns_about_the_cost(self) -> None:
        assert "Expensive" in VeraAiAnswersTool(api_key="vera_live_test").description

    def test_the_timeout_waits_longer_than_the_150_seconds_the_api_may_take(self) -> None:
        tool = VeraAiAnswersTool(api_key="vera_live_test")
        assert tool.timeout == 175.0
        assert tool._vera.timeout == httpx.Timeout(175.0, connect=10.0)

    async def test_async(self) -> None:
        client, recorder = make_client(success(ANSWERS, charged=200))
        message = await VeraAiAnswersTool(client=client, brand="B").ainvoke(
            call("vera_ai_answers", prompt="p")
        )
        assert message.artifact["credits_used"] == 200
        assert recorder.body["brand"] == "B"

    def test_402_raises_with_the_pricing_link(self) -> None:
        client, _ = make_client(failure(402, "This request requires 500 credits."))
        with pytest.raises(VeraInsufficientCreditsError, match=r"vera\.waggle\.ae/pricing"):
            VeraAiAnswersTool(client=client, brand="B", engines=["claude"]).invoke({"prompt": "p"})

    def test_a_free_account_that_hit_its_daily_pace_raises_the_rate_limit(self) -> None:
        client, _ = make_client(
            failure(429, "Daily free limit reached.", headers={"retry-after": "60"})
        )
        with pytest.raises(VeraRateLimitError) as caught:
            VeraAiAnswersTool(client=client, brand="B").invoke({"prompt": "p"})
        assert caught.value.retry_after == 60.0

    def test_503_and_504_raise_and_say_they_were_not_charged(self) -> None:
        for status in (503, 504):
            client, _ = make_client(failure(status, "x", headers={"x-waggle-credits-used": "0"}))
            with pytest.raises(VeraServerError, match=r"No credits were charged\."):
                VeraAiAnswersTool(client=client, brand="B").invoke({"prompt": "p"})

    def test_422_goes_back_to_the_model_with_the_message(self) -> None:
        client, _ = make_client(failure(422, "prompt: String should have at most 1000 characters"))
        message = VeraAiAnswersTool(client=client, brand="B").invoke(
            call("vera_ai_answers", prompt="p")
        )
        assert message.status == "error"
        assert "Invalid input" in str(message.content)
        assert "at most 1000 characters" in str(message.content)

    def test_it_never_prints_a_key(self) -> None:
        tool = VeraAiAnswersTool(api_key="vera_live_super_secret")
        assert "vera_live_super_secret" not in repr(tool)
        assert "vera_live_super_secret" not in str(tool.model_dump())


class TestFormatAiAnswers:
    def test_an_empty_report(self) -> None:
        assert format_ai_answers({}) == "No assistant answered."
        assert format_ai_answers({"engines": []}) == "No assistant answered."

    def test_a_single_answer_that_names_the_brand(self) -> None:
        text = format_ai_answers(
            {
                "prompt": "p",
                "brand": "B",
                "engines": [
                    {
                        "engine": "gemini",
                        "status": "ok",
                        "aggregate": {"samples_ok": 1, "mentions": 1, "mention_rate": 1.0},
                    }
                ],
            }
        )
        assert text.splitlines()[1] == "- Gemini: B named in 1 of 1 answer"

    def test_tolerates_odd_shapes(self) -> None:
        assert format_ai_answers({"engines": ["junk", {"engine": "claude"}]}).endswith(
            "- Claude: did not answer."
        )


class TestGeoRadarTool:
    def test_returns_the_radar_brief(self) -> None:
        client, recorder = make_client(success(RADAR, charged=6))
        message = VeraGeoRadarTool(
            client=client, brand="Roasters", domain="roasters.ae", location="Dubai", gl="ae"
        ).invoke(call("vera_geo_radar", query="best roasters"))
        assert str(message.content).startswith("# GEO Visibility Radar: best roasters")
        assert message.artifact["citation_probability_score"] == 42
        assert message.artifact["credits_used"] == 6
        assert recorder.last.url.path == "/v1/geo/radar/query"
        assert recorder.body == {
            "query": "best roasters",
            "depth": 30,
            "brand": "Roasters",
            "domain": "roasters.ae",
            "location": "Dubai",
            "gl": "ae",
        }

    def test_the_model_can_give_the_brand_and_domain(self) -> None:
        client, recorder = make_client(success(RADAR, charged=6))
        VeraGeoRadarTool(client=client, brand="Default").invoke(
            {"query": "q", "brand": "Asked", "domain": "asked.com"}
        )
        assert recorder.body["brand"] == "Asked"
        assert recorder.body["domain"] == "asked.com"

    def test_without_a_brief_it_falls_back_to_the_json(self) -> None:
        client, _ = make_client(success({"citation_probability_score": 7}, charged=6))
        assert '"citation_probability_score": 7' in VeraGeoRadarTool(client=client).invoke(
            {"query": "q"}
        )

    def test_settings_and_timeout(self) -> None:
        with pytest.raises(ValidationError):
            VeraGeoRadarTool(api_key="vera_live_test", depth=101)
        assert VeraGeoRadarTool(api_key="vera_live_test").timeout == 80.0

    async def test_async(self) -> None:
        client, recorder = make_client(success(RADAR, charged=6))
        message = await VeraGeoRadarTool(client=client, depth=10).ainvoke(
            call("vera_geo_radar", query="q")
        )
        assert message.artifact["credits_used"] == 6
        assert recorder.body["depth"] == 10


class TestCompanyFootprintTool:
    def test_returns_the_footprint_brief(self) -> None:
        client, recorder = make_client(success(FOOTPRINT, charged=3))
        message = VeraCompanyFootprintTool(client=client, location="Dubai", gl="ae").invoke(
            call("vera_company_footprint", company="Roasters", domain="roasters.ae")
        )
        assert str(message.content).startswith("# Company footprint: Roasters")
        assert message.artifact["credits_used"] == 3
        assert recorder.last.url.path == "/v1/company/footprint"
        assert recorder.body == {
            "company": "Roasters",
            "depth": 20,
            "domain": "roasters.ae",
            "location": "Dubai",
            "gl": "ae",
        }

    def test_the_model_chooses_only_the_company_and_domain(self) -> None:
        tool = VeraCompanyFootprintTool(api_key="vera_live_test")
        assert set(tool.args) == {"company", "domain"}
        assert tool.timeout == 65.0

    def test_max_characters(self) -> None:
        client, _ = make_client(success({"llm_markdown": "y" * 300}, charged=3))
        text = VeraCompanyFootprintTool(client=client, max_characters=50).invoke({"company": "c"})
        assert text.startswith("y" * 50)
        assert "[Truncated" in text

    async def test_async(self) -> None:
        client, recorder = make_client(success(FOOTPRINT, charged=3))
        message = await VeraCompanyFootprintTool(client=client).ainvoke(
            call("vera_company_footprint", company="c")
        )
        assert message.artifact["credits_used"] == 3
        assert recorder.body["company"] == "c"

    def test_a_bad_company_raises_before_a_request(self) -> None:
        client, recorder = make_client()
        with pytest.raises(ValueError, match="company is required"):
            VeraCompanyFootprintTool(client=client).invoke({"company": "  "})
        assert recorder.requests == []


class TestSetupOfEveryTool:
    ALL = (
        VeraSerpTool,
        VeraKeywordIdeasTool,
        VeraAiAnswersTool,
        VeraGeoRadarTool,
        VeraCompanyFootprintTool,
    )

    @pytest.mark.parametrize("tool_class", ALL)
    def test_key_from_the_environment(
        self, tool_class: type, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("VERA_API_KEY", "vera_live_from_env")
        tool = tool_class()
        assert tool.api_key is not None
        assert tool.api_key.get_secret_value() == "vera_live_from_env"

    @pytest.mark.parametrize("tool_class", ALL)
    def test_a_missing_key_fails_at_construction(self, tool_class: type) -> None:
        with pytest.raises(ValidationError, match="VERA_API_KEY"):
            tool_class()

    @pytest.mark.parametrize("tool_class", ALL)
    def test_the_default_base_url_is_the_direct_api(self, tool_class: type) -> None:
        assert tool_class(api_key="vera_live_test")._vera.base_url == "https://vera.waggle.ae/v1"

    @pytest.mark.parametrize("tool_class", ALL)
    def test_the_base_url_is_configurable(self, tool_class: type) -> None:
        tool = tool_class(api_key="vera_live_test", base_url="https://staging.vera.example/v1/")
        assert tool._vera.base_url == "https://staging.vera.example/v1"

    @pytest.mark.parametrize("tool_class", ALL)
    def test_the_artifact_format_is_on(self, tool_class: type) -> None:
        assert tool_class(api_key="vera_live_test").response_format == "content_and_artifact"

    @pytest.mark.parametrize("tool_class", ALL)
    def test_agents_get_a_real_description(self, tool_class: type) -> None:
        tool = tool_class(api_key="vera_live_test")
        assert tool.name.startswith("vera_")
        assert len(tool.description) > 80

    def test_a_refusal_is_a_tool_exception_when_handling_is_off(self) -> None:
        client, _ = make_client(failure(422, "Nope."))
        tool = VeraSerpTool(client=client, handle_tool_error=False)
        with pytest.raises(ToolException, match=r"Nope\."):
            tool.invoke({"query": "q"})
