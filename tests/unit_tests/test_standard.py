"""LangChain's standard tool unit tests, run against both tools."""

from __future__ import annotations

from typing import Any

from langchain_tests.unit_tests import ToolsUnitTests

from langchain_vera import VeraScrapeTool, VeraSearchTool


class TestVeraScrapeToolStandard(ToolsUnitTests):
    @property
    def tool_constructor(self) -> type[VeraScrapeTool]:
        return VeraScrapeTool

    @property
    def tool_constructor_params(self) -> dict[str, Any]:
        return {"api_key": "vera_live_standard_tests"}

    @property
    def tool_invoke_params_example(self) -> dict[str, Any]:
        return {"url": "https://example.com"}

    @property
    def init_from_env_params(self) -> tuple[dict[str, str], dict[str, Any], dict[str, Any]]:
        return {"VERA_API_KEY": "vera_live_from_env"}, {}, {"api_key": "vera_live_from_env"}


class TestVeraSearchToolStandard(ToolsUnitTests):
    @property
    def tool_constructor(self) -> type[VeraSearchTool]:
        return VeraSearchTool

    @property
    def tool_constructor_params(self) -> dict[str, Any]:
        return {"api_key": "vera_live_standard_tests"}

    @property
    def tool_invoke_params_example(self) -> dict[str, Any]:
        return {"query": "open source web scrapers"}

    @property
    def init_from_env_params(self) -> tuple[dict[str, str], dict[str, Any], dict[str, Any]]:
        return {"VERA_API_KEY": "vera_live_from_env"}, {}, {"api_key": "vera_live_from_env"}
