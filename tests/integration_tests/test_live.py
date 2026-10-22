"""Live calls against the real Vera API.

Skipped by default: they spend credits and need a key. Run them with

    VERA_API_KEY=vera_live_... uv run pytest -m live

The AI answers call is far more expensive than the others, so it needs a second opt-in:
VERA_LIVE_EXPENSIVE=1.
"""

from __future__ import annotations

import os

import pytest

from langchain_vera import VeraAiAnswersTool, VeraKeywordIdeasTool, VeraScrapeTool, VeraSerpTool

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(not os.environ.get("VERA_API_KEY"), reason="VERA_API_KEY is not set"),
]

expensive = pytest.mark.skipif(
    not os.environ.get("VERA_LIVE_EXPENSIVE"),
    reason="VERA_LIVE_EXPENSIVE is not set (an AI answer costs far more than a search)",
)


def test_scrape_tool_reads_example_com() -> None:
    """One scrape through the tool: key, request shape, response parsing and credits."""
    message = VeraScrapeTool().invoke(
        {
            "type": "tool_call",
            "name": "vera_scrape",
            "id": "live",
            "args": {"url": "https://example.com"},
        }
    )
    assert "Example Domain" in message.content
    assert message.artifact["credits_used"] == 1


def test_serp_tool_reads_a_results_page() -> None:
    """One live results page: the direct API route, the brief and the charge."""
    message = VeraSerpTool(gl="us", hl="en", depth=10).invoke(
        {"type": "tool_call", "name": "vera_serp", "id": "live", "args": {"query": "coffee"}}
    )
    assert message.artifact["organic"]
    assert message.artifact["credits_used"] >= 1


def test_keyword_ideas_tool_finds_ideas() -> None:
    """Keyword ideas with no autocomplete expansion: the cheapest way to prove the route."""
    message = VeraKeywordIdeasTool(expand="none", limit=5).invoke(
        {
            "type": "tool_call",
            "name": "vera_keyword_ideas",
            "id": "live",
            "args": {"seed": "coffee"},
        }
    )
    assert message.artifact["ideas"]


@expensive
def test_ai_answers_tool_asks_one_assistant_once() -> None:
    """One ChatGPT answer: the long timeout, the metrics and the charge."""
    message = VeraAiAnswersTool(brand="Starbucks", engines=["chatgpt"], samples=1).invoke(
        {
            "type": "tool_call",
            "name": "vera_ai_answers",
            "id": "live",
            "args": {"prompt": "What are the best coffee chains in the world?"},
        }
    )
    assert message.artifact["engines"]
    assert message.artifact["credits_used"] >= 0
