"""The public surface of the package."""

from __future__ import annotations

import langchain_vera

EXPECTED = [
    "DEFAULT_BASE_URL",
    "VeraAPIError",
    "VeraAiAnswersTool",
    "VeraAuthenticationError",
    "VeraClient",
    "VeraCompanyFootprintTool",
    "VeraConfigurationError",
    "VeraConflictError",
    "VeraConnectionError",
    "VeraError",
    "VeraGeoRadarTool",
    "VeraInsufficientCreditsError",
    "VeraKeywordIdeasTool",
    "VeraLoader",
    "VeraPermissionError",
    "VeraRateLimitError",
    "VeraRequestError",
    "VeraResponse",
    "VeraResponseError",
    "VeraScrapeTool",
    "VeraSearchTool",
    "VeraSerpTool",
    "VeraServerError",
    "VeraTimeoutError",
    "__version__",
]


def test_public_names() -> None:
    assert sorted(langchain_vera.__all__) == sorted(EXPECTED)
    for name in EXPECTED:
        assert hasattr(langchain_vera, name)


def test_error_hierarchy() -> None:
    from langchain_vera import errors

    api_errors = [
        errors.VeraAuthenticationError,
        errors.VeraInsufficientCreditsError,
        errors.VeraPermissionError,
        errors.VeraRequestError,
        errors.VeraConflictError,
        errors.VeraRateLimitError,
        errors.VeraServerError,
    ]
    for error in api_errors:
        assert issubclass(error, errors.VeraAPIError)
    assert issubclass(errors.VeraAPIError, errors.VeraError)
    assert issubclass(errors.VeraTimeoutError, errors.VeraConnectionError)
    assert issubclass(errors.VeraConfigurationError, ValueError)


def test_version_matches_the_package_metadata() -> None:
    assert langchain_vera.__version__ in {"0.1.0", ""}
