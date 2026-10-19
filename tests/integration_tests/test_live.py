"""One live call against the real Vera API.

Skipped by default: it spends a credit and needs a key. Run it with

    VERA_API_KEY=vera_live_... uv run pytest -m live
"""

from __future__ import annotations

import os

import pytest

from langchain_vera import VeraScrapeTool

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(not os.environ.get("VERA_API_KEY"), reason="VERA_API_KEY is not set"),
]


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
