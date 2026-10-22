"""LangChain's standard tool unit tests, run against both tools."""

from __future__ import annotations

from typing import Any

from langchain_tests.unit_tests import ToolsUnitTests

from langchain_vera import (
    VeraAiAnswersTool,
    VeraCompanyFootprintTool,
    VeraGeoRadarTool,
    VeraKeywordIdeasTool,
    VeraScrapeTool,
    VeraSearchTool,
    VeraSerpTool,
)


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


class TestVeraSerpToolStandard(ToolsUnitTests):
    @property
    def tool_constructor(self) -> type[VeraSerpTool]:
        return VeraSerpTool

    @property
    def tool_constructor_params(self) -> dict[str, Any]:
        return {"api_key": "vera_live_standard_tests"}

    @property
    def tool_invoke_params_example(self) -> dict[str, Any]:
        return {"query": "best coffee in dubai"}

    @property
    def init_from_env_params(self) -> tuple[dict[str, str], dict[str, Any], dict[str, Any]]:
        return {"VERA_API_KEY": "vera_live_from_env"}, {}, {"api_key": "vera_live_from_env"}


class TestVeraKeywordIdeasToolStandard(ToolsUnitTests):
    @property
    def tool_constructor(self) -> type[VeraKeywordIdeasTool]:
        return VeraKeywordIdeasTool

    @property
    def tool_constructor_params(self) -> dict[str, Any]:
        return {"api_key": "vera_live_standard_tests"}

    @property
    def tool_invoke_params_example(self) -> dict[str, Any]:
        return {"seed": "specialty coffee"}

    @property
    def init_from_env_params(self) -> tuple[dict[str, str], dict[str, Any], dict[str, Any]]:
        return {"VERA_API_KEY": "vera_live_from_env"}, {}, {"api_key": "vera_live_from_env"}


class TestVeraAiAnswersToolStandard(ToolsUnitTests):
    @property
    def tool_constructor(self) -> type[VeraAiAnswersTool]:
        return VeraAiAnswersTool

    @property
    def tool_constructor_params(self) -> dict[str, Any]:
        return {"api_key": "vera_live_standard_tests"}

    @property
    def tool_invoke_params_example(self) -> dict[str, Any]:
        return {"prompt": "What are the best coffee roasters in Dubai?", "brand": "Roasters"}

    @property
    def init_from_env_params(self) -> tuple[dict[str, str], dict[str, Any], dict[str, Any]]:
        return {"VERA_API_KEY": "vera_live_from_env"}, {}, {"api_key": "vera_live_from_env"}


class TestVeraGeoRadarToolStandard(ToolsUnitTests):
    @property
    def tool_constructor(self) -> type[VeraGeoRadarTool]:
        return VeraGeoRadarTool

    @property
    def tool_constructor_params(self) -> dict[str, Any]:
        return {"api_key": "vera_live_standard_tests"}

    @property
    def tool_invoke_params_example(self) -> dict[str, Any]:
        return {"query": "best specialty coffee roasters in dubai"}

    @property
    def init_from_env_params(self) -> tuple[dict[str, str], dict[str, Any], dict[str, Any]]:
        return {"VERA_API_KEY": "vera_live_from_env"}, {}, {"api_key": "vera_live_from_env"}


class TestVeraCompanyFootprintToolStandard(ToolsUnitTests):
    @property
    def tool_constructor(self) -> type[VeraCompanyFootprintTool]:
        return VeraCompanyFootprintTool

    @property
    def tool_constructor_params(self) -> dict[str, Any]:
        return {"api_key": "vera_live_standard_tests"}

    @property
    def tool_invoke_params_example(self) -> dict[str, Any]:
        return {"company": "Roasters"}

    @property
    def init_from_env_params(self) -> tuple[dict[str, str], dict[str, Any], dict[str, Any]]:
        return {"VERA_API_KEY": "vera_live_from_env"}, {}, {"api_key": "vera_live_from_env"}
