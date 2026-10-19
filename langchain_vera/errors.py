"""Typed errors raised by langchain-vera.

Every error the package raises on purpose derives from `VeraError`. Errors that come from an HTTP
answer derive from `VeraAPIError` and carry the status code, the machine code and request id Vera
sent, and whether the call was charged.

The HTTP status is the dependable signal. Vera's `error.code` is only meaningful for failed fetches
and searches (422 and 502), so it is kept on `code` but should not be used to tell a 401 from a 402.
"""

from __future__ import annotations


class VeraError(Exception):
    """Base class for every error this package raises on purpose."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        code: str | None = None,
        request_id: str | None = None,
        credits_used: int | None = None,
        retry_after: float | None = None,
    ) -> None:
        """Create the error.

        Args:
            message: What went wrong, in a sentence a person can act on.
            status_code: The HTTP status of the answer, when there was one.
            code: Vera's machine code for a failed fetch or search (for example
                `robots_disallowed`), when it sent one.
            request_id: Vera's request id, useful when contacting support.
            credits_used: Credits the call was charged. `0` means it was not charged.
            retry_after: Seconds to wait before retrying, from the Retry-After header.
        """
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.code = code
        self.request_id = request_id
        self.credits_used = credits_used
        self.retry_after = retry_after


class VeraConfigurationError(VeraError, ValueError):
    """The client cannot be set up, for example because no API key was given."""


class VeraConnectionError(VeraError):
    """No answer arrived: DNS, TLS, connection failure or a dropped connection."""


class VeraTimeoutError(VeraConnectionError):
    """No answer arrived within the timeout."""


class VeraResponseError(VeraError):
    """Vera answered 2xx but the body was not the JSON this package expects."""


class VeraAPIError(VeraError):
    """Vera answered with a non-2xx status."""


class VeraAuthenticationError(VeraAPIError):
    """401: the API key is missing, invalid, expired or revoked."""


class VeraInsufficientCreditsError(VeraAPIError):
    """402: the account has too few credits, or needs a paid plan."""


class VeraPermissionError(VeraAPIError):
    """403: the account must act first, for example by confirming its email address."""


class VeraRequestError(VeraAPIError):
    """The request was refused: invalid parameters or a page that cannot be read (422, other 4xx).

    A refused call is not charged. An agent can often recover from this one by trying a different
    URL or query, so the tools turn it into a `ToolException`.
    """


class VeraConflictError(VeraAPIError):
    """409 or 410: an idempotency key was reused, or its request is still running."""


class VeraRateLimitError(VeraAPIError):
    """429: too many requests, or the free-account daily limit. See `retry_after`."""


class VeraServerError(VeraAPIError):
    """5xx: Vera or its search provider is temporarily unavailable."""
