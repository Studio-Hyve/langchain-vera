"""LangChain integration for Vera by Waggle: tools and a document loader for scraping and search."""

from importlib import metadata

from langchain_vera._client import DEFAULT_BASE_URL, VeraClient, VeraResponse
from langchain_vera.document_loaders import VeraLoader
from langchain_vera.errors import (
    VeraAPIError,
    VeraAuthenticationError,
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
from langchain_vera.tools import VeraScrapeTool, VeraSearchTool

try:
    __version__ = metadata.version("langchain-vera")
except metadata.PackageNotFoundError:  # pragma: no cover - running from a source tree
    __version__ = ""
del metadata

__all__ = [
    "DEFAULT_BASE_URL",
    "VeraAPIError",
    "VeraAuthenticationError",
    "VeraClient",
    "VeraConfigurationError",
    "VeraConflictError",
    "VeraConnectionError",
    "VeraError",
    "VeraInsufficientCreditsError",
    "VeraLoader",
    "VeraPermissionError",
    "VeraRateLimitError",
    "VeraRequestError",
    "VeraResponse",
    "VeraResponseError",
    "VeraScrapeTool",
    "VeraSearchTool",
    "VeraServerError",
    "VeraTimeoutError",
    "__version__",
]
