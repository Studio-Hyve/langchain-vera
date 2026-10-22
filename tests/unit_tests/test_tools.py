"""VeraScrapeTool and VeraSearchTool, with HTTP mocked."""

from __future__ import annotations

from typing import Any

import httpx
import pytest
from langchain_core.messages import ToolMessage
from langchain_core.tools import ToolException
from pydantic import ValidationError

from langchain_vera import (
    VeraAuthenticationError,
    VeraInsufficientCreditsError,
    VeraRateLimitError,
    VeraScrapeTool,
    VeraSearchTool,
    VeraServerError,
)
from langchain_vera.tools import VeraScrapeInput, format_results
from tests.unit_tests.conftest import failure, make_client, success

PAGE = {
    "url": "https://example.com/",
    "title": "Example Domain",
    "markdown": "# Example Domain\n\nThis domain is for use in examples.",
    "text": "Example Domain This domain is for use in examples.",
    "status_code": 200,
}

RESULTS = {
    "query": "vera",
    "type": "web",
    "results": [
        {
            "position": 1,
            "title": "Vera",
            "url": "https://vera.waggle.ae/",
            "snippet": "Web scraping API.",
        },
        {
            "position": 2,
            "title": "Docs",
            "url": "https://vera.waggle.ae/docs",
            "snippet": "API reference.",
            "source": "waggle.ae",
            "date": "2026-01-01",
        },
    ],
}


def call(name: str, **args: Any) -> dict[str, Any]:
    """A `ToolCall`, which makes `invoke` return a ToolMessage with the artifact."""
    return {"type": "tool_call", "name": name, "id": "call_1", "args": args}


class TestScrapeTool:
    def test_returns_markdown(self) -> None:
        client, recorder = make_client(success(PAGE))
        tool = VeraScrapeTool(client=client)
        assert tool.invoke({"url": "https://example.com"}) == PAGE["markdown"]
        assert recorder.body == {"url": "https://example.com", "response_format": "markdown"}

    def test_a_tool_call_also_returns_the_artifact(self) -> None:
        client, _ = make_client(success(PAGE, charged=1))
        message = VeraScrapeTool(client=client).invoke(
            call("vera_scrape", url="https://example.com")
        )
        assert isinstance(message, ToolMessage)
        assert message.content == PAGE["markdown"]
        assert message.status == "success"
        assert message.artifact == {
            "url": "https://example.com/",
            "title": "Example Domain",
            "status_code": 200,
            "credits_used": 1,
            "request_id": "req_test",
            "elapsed_ms": 123,
        }

    def test_render_is_a_tool_setting_not_something_the_model_chooses(self) -> None:
        client, recorder = make_client(success(PAGE, charged=5))
        tool = VeraScrapeTool(client=client, render=True)
        tool.invoke({"url": "https://spa.example.com"})
        assert recorder.body["render"] is True
        assert set(tool.args) == {"url"}
        assert VeraScrapeInput.model_json_schema()["required"] == ["url"]

    def test_max_characters_truncates_with_a_notice(self) -> None:
        client, _ = make_client(success({**PAGE, "markdown": "x" * 100}))
        text = VeraScrapeTool(client=client, max_characters=10).invoke(
            {"url": "https://example.com"}
        )
        assert text.startswith(
            "x" * 10 + "\n\n[Truncated: showing the first 10 of 100 characters.]"
        )

    def test_short_pages_are_not_truncated(self) -> None:
        client, _ = make_client(success({**PAGE, "markdown": "short"}))
        assert (
            VeraScrapeTool(client=client, max_characters=10).invoke({"url": "https://e.co/page"})
            == "short"
        )

    def test_falls_back_to_text_when_there_is_no_markdown(self) -> None:
        client, _ = make_client(success({"text": "plain words", "url": "https://example.com/"}))
        assert VeraScrapeTool(client=client).invoke({"url": "https://example.com"}) == "plain words"

    def test_an_empty_page_is_explained(self) -> None:
        client, _ = make_client(success({"markdown": "  ", "text": ""}))
        text = VeraScrapeTool(client=client).invoke({"url": "https://example.com"})
        assert text == "No readable content was found at https://example.com."

    async def test_async(self) -> None:
        client, recorder = make_client(success(PAGE))
        tool = VeraScrapeTool(client=client)
        assert await tool.ainvoke({"url": "https://example.com"}) == PAGE["markdown"]
        message = await tool.ainvoke(call("vera_scrape", url="https://example.com"))
        assert message.artifact["credits_used"] == 1
        assert recorder.body["response_format"] == "markdown"

    def test_a_refused_page_goes_back_to_the_model_as_text(self) -> None:
        client, _ = make_client(
            failure(
                422,
                "This page is disallowed by robots.txt.",
                code="robots_disallowed",
                headers={"x-waggle-credits-used": "0"},
            )
        )
        message = VeraScrapeTool(client=client).invoke(
            call("vera_scrape", url="https://example.com")
        )
        assert isinstance(message, ToolMessage)
        assert message.status == "error"
        assert "disallowed by robots.txt" in str(message.content)
        assert "No credits were charged." in str(message.content)

    def test_a_refused_page_raises_a_tool_exception_when_handling_is_off(self) -> None:
        client, _ = make_client(failure(422, "Nope."))
        tool = VeraScrapeTool(client=client, handle_tool_error=False)
        with pytest.raises(ToolException, match=r"Nope\."):
            tool.invoke({"url": "https://example.com"})

    @pytest.mark.parametrize(
        ("status", "error_class"),
        [
            (401, VeraAuthenticationError),
            (402, VeraInsufficientCreditsError),
            (429, VeraRateLimitError),
            (500, VeraServerError),
        ],
    )
    def test_errors_the_model_cannot_fix_are_raised_typed(
        self, status: int, error_class: type[Exception]
    ) -> None:
        client, _ = make_client(failure(status, "x"))
        with pytest.raises(error_class):
            VeraScrapeTool(client=client).invoke({"url": "https://example.com"})

    async def test_async_refusal_and_errors(self) -> None:
        client, _ = make_client(failure(422, "Nope."))
        message = await VeraScrapeTool(client=client).ainvoke(
            call("vera_scrape", url="https://example.com")
        )
        assert message.status == "error"
        client, _ = make_client(failure(402, "x"))
        with pytest.raises(VeraInsufficientCreditsError):
            await VeraScrapeTool(client=client).ainvoke({"url": "https://example.com"})

    def test_a_bad_url_is_refused_without_a_request_and_not_raised_as_a_crash(self) -> None:
        client, recorder = make_client()
        with pytest.raises(ValueError, match="url must be"):
            VeraScrapeTool(client=client).invoke({"url": "not a url"})
        assert recorder.requests == []

    def test_metadata_for_models(self) -> None:
        tool = VeraScrapeTool(api_key="vera_live_test")
        assert tool.name == "vera_scrape"
        assert "Markdown" in tool.description
        assert tool.response_format == "content_and_artifact"


class TestSearchTool:
    def test_formats_a_numbered_list(self) -> None:
        client, recorder = make_client(success(RESULTS, charged=3))
        text = VeraSearchTool(client=client).invoke({"query": "vera"})
        assert text == (
            "1. Vera\n"
            "   https://vera.waggle.ae/\n"
            "   Web scraping API.\n"
            "2. Docs\n"
            "   https://vera.waggle.ae/docs\n"
            "   (waggle.ae | 2026-01-01)\n"
            "   API reference."
        )
        assert recorder.last.url.path.endswith("/search")
        assert recorder.body == {"query": "vera", "limit": 10}

    def test_a_tool_call_returns_the_full_data_as_the_artifact(self) -> None:
        client, _ = make_client(success(RESULTS, charged=3))
        message = VeraSearchTool(client=client).invoke(call("vera_search", query="vera"))
        assert isinstance(message, ToolMessage)
        assert message.artifact["results"] == RESULTS["results"]
        assert message.artifact["credits_used"] == 3
        assert message.artifact["request_id"] == "req_test"

    @pytest.mark.parametrize(
        ("search_type", "path"),
        [("web", "/search"), ("news", "/search/news"), ("videos", "/search/videos")],
    )
    def test_search_types(self, search_type: str, path: str) -> None:
        client, recorder = make_client(success(RESULTS, charged=3))
        VeraSearchTool(client=client, search_type=search_type).invoke({"query": "q"})
        assert recorder.last.url.path == "/v1" + path

    def test_settings_are_sent(self) -> None:
        client, recorder = make_client(success(RESULTS, charged=3))
        tool = VeraSearchTool(client=client, max_results=25, country="ae", language="ar")
        tool.invoke({"query": "q"})
        assert recorder.body == {"query": "q", "limit": 25, "country": "ae", "language": "ar"}

    def test_the_model_only_chooses_the_query(self) -> None:
        tool = VeraSearchTool(api_key="vera_live_test")
        assert set(tool.args) == {"query"}

    @pytest.mark.parametrize(
        "kwargs", [{"max_results": 0}, {"max_results": 101}, {"country": "US"}, {"language": "eng"}]
    )
    def test_settings_are_validated_at_construction(self, kwargs: dict[str, Any]) -> None:
        with pytest.raises(ValidationError):
            VeraSearchTool(api_key="vera_live_test", **kwargs)

    def test_no_results(self) -> None:
        client, _ = make_client(success({"results": []}, charged=3))
        assert (
            VeraSearchTool(client=client).invoke({"query": "zzz"}) == "No results found for 'zzz'."
        )

    async def test_async(self) -> None:
        client, recorder = make_client(success(RESULTS, charged=3))
        message = await VeraSearchTool(client=client, search_type="news").ainvoke(
            call("vera_search", query="vera")
        )
        assert message.artifact["credits_used"] == 3
        assert recorder.last.url.path.endswith("/search/news")

    def test_errors(self) -> None:
        client, _ = make_client(failure(429, "Too many requests.", headers={"retry-after": "5"}))
        with pytest.raises(VeraRateLimitError) as caught:
            VeraSearchTool(client=client).invoke({"query": "q"})
        assert caught.value.retry_after == 5.0
        client, _ = make_client(failure(422, "Check these fields: query"))
        message = VeraSearchTool(client=client).invoke(call("vera_search", query="q"))
        assert message.status == "error"

    async def test_async_errors(self) -> None:
        client, _ = make_client(
            failure(502, "Search is temporarily unavailable.", code="fetch_failed")
        )
        with pytest.raises(VeraServerError):
            await VeraSearchTool(client=client).ainvoke({"query": "q"})
        client, _ = make_client(failure(422, "Nope."))
        message = await VeraSearchTool(client=client).ainvoke(call("vera_search", query="q"))
        assert message.status == "error"


class TestFormatResults:
    def test_videos_show_channel_duration_and_description(self) -> None:
        text = format_results(
            [
                {
                    "title": "Intro",
                    "url": "https://youtu.be/abc",
                    "channel": "Waggle",
                    "duration": "3:21",
                    "description": "d" * 400,
                }
            ],
            "videos",
        )
        assert "(Waggle | 3:21)" in text
        assert text.splitlines()[-1] == "   " + "d" * 300

    def test_tolerates_odd_rows(self) -> None:
        assert format_results(["junk", {"url": "https://a.example"}, {}]) == (
            "2. https://a.example\n   https://a.example\n3. Untitled"
        )


class TestSetup:
    @pytest.mark.parametrize("tool_class", [VeraScrapeTool, VeraSearchTool])
    def test_key_from_the_environment(
        self, tool_class: type, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("VERA_API_KEY", "vera_live_from_env")
        tool = tool_class()
        assert tool.api_key is not None
        assert tool.api_key.get_secret_value() == "vera_live_from_env"

    @pytest.mark.parametrize("tool_class", [VeraScrapeTool, VeraSearchTool])
    def test_a_missing_key_fails_at_construction(self, tool_class: type) -> None:
        with pytest.raises(ValidationError, match="VERA_API_KEY"):
            tool_class()

    @pytest.mark.parametrize("tool_class", [VeraScrapeTool, VeraSearchTool])
    def test_the_key_is_never_shown(self, tool_class: type) -> None:
        tool = tool_class(api_key="vera_live_super_secret")
        assert "vera_live_super_secret" not in repr(tool)
        assert "vera_live_super_secret" not in str(tool.model_dump())

    def test_the_timeout_and_base_url_reach_the_client(self) -> None:
        tool = VeraScrapeTool(
            api_key="vera_live_test", timeout=12, base_url="https://vera.example/v1/"
        )
        assert tool._vera.base_url == "https://vera.example/v1"
        assert tool._vera.timeout == httpx.Timeout(12, connect=10)
