"""LLM generator module for n8n Forge.

This module handles all communication with the OpenAI Chat Completions API.
It accepts a list of pre-built messages (system + user), sends them to the
specified model, and returns the raw response text for downstream parsing.

The module is intentionally thin — it does not build prompts or parse JSON.
Those responsibilities belong to :mod:`n8n_forge.prompt_builder` and
:mod:`n8n_forge.parser` respectively.

Example usage::

    from n8n_forge.generator import generate_workflow_response

    raw_text = generate_workflow_response(
        messages=[
            {"role": "system", "content": "You are an n8n expert..."},
            {"role": "user", "content": "Generate a workflow for..."},
        ],
        model="gpt-4o-mini",
        api_key="sk-...",
        temperature=0.2,
    )
    # raw_text is a string containing the LLM's response (expected to include JSON)
"""

from __future__ import annotations

import logging
import os
from typing import Any

import openai
from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Default model to use when none is specified.
DEFAULT_MODEL = "gpt-4o-mini"

#: Default sampling temperature — low for deterministic JSON generation.
DEFAULT_TEMPERATURE = 0.2

#: Default maximum tokens for the completion response.
DEFAULT_MAX_TOKENS = 4096

#: Default request timeout in seconds.
DEFAULT_TIMEOUT = 120.0


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class GeneratorError(Exception):
    """Raised when the OpenAI API call fails or returns an unusable response.

    Attributes:
        message: Human-readable description of what went wrong.
        cause: The underlying exception, if any.
    """

    def __init__(self, message: str, cause: BaseException | None = None) -> None:
        """Initialise with a message and optional root cause.

        Args:
            message: Description of the failure.
            cause: The original exception that caused this error, if any.
        """
        super().__init__(message)
        self.message = message
        self.cause = cause

    def __str__(self) -> str:
        """Return a formatted error string."""
        if self.cause:
            return f"{self.message}: {self.cause}"
        return self.message


class EmptyResponseError(GeneratorError):
    """Raised when the LLM returns an empty or whitespace-only response."""


class RateLimitError(GeneratorError):
    """Raised when the OpenAI API returns a 429 rate-limit response."""


class AuthenticationError(GeneratorError):
    """Raised when the OpenAI API key is invalid or missing."""


# ---------------------------------------------------------------------------
# Client factory
# ---------------------------------------------------------------------------


def _create_client(api_key: str | None, timeout: float) -> OpenAI:
    """Create and return a configured :class:`openai.OpenAI` client.

    Resolution order for the API key:

    1. The ``api_key`` argument (if not ``None``).
    2. The ``OPENAI_API_KEY`` environment variable.

    Args:
        api_key: Explicit API key string, or ``None`` to fall back to the
            environment variable.
        timeout: Request timeout in seconds.

    Returns:
        A configured :class:`openai.OpenAI` client instance.

    Raises:
        AuthenticationError: If no API key is available from either source.
    """
    resolved_key = api_key or os.environ.get("OPENAI_API_KEY")
    if not resolved_key or not resolved_key.strip():
        raise AuthenticationError(
            "OpenAI API key is required. Set the OPENAI_API_KEY environment "
            "variable or pass --api-key on the command line."
        )
    return OpenAI(api_key=resolved_key.strip(), timeout=timeout)


# ---------------------------------------------------------------------------
# Core generation function
# ---------------------------------------------------------------------------


def generate_workflow_response(
    messages: list[dict[str, str]],
    model: str = DEFAULT_MODEL,
    api_key: str | None = None,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    timeout: float = DEFAULT_TIMEOUT,
) -> str:
    """Call the OpenAI Chat Completions API and return the raw response text.

    Sends a list of pre-built chat messages to the specified OpenAI model
    and returns the content of the first choice's message as a string.
    Detailed error mapping converts OpenAI SDK exceptions into
    project-specific :class:`GeneratorError` subclasses.

    Args:
        messages: List of message dicts in OpenAI chat format, each with
            ``"role"`` and ``"content"`` keys. Typically produced by
            :func:`n8n_forge.prompt_builder.build_messages`.
        model: OpenAI model identifier to use for generation.
            Defaults to ``"gpt-4o-mini"``.
        api_key: OpenAI API key. If ``None``, the ``OPENAI_API_KEY``
            environment variable is used.
        temperature: Sampling temperature in the range ``[0.0, 2.0]``.
            Lower values produce more deterministic output.
            Defaults to ``0.2``.
        max_tokens: Maximum number of tokens to generate in the response.
            Defaults to ``4096``.
        timeout: HTTP request timeout in seconds.
            Defaults to ``120.0``.

    Returns:
        The raw string content of the LLM's response message, which is
        expected to contain a JSON code block for downstream parsing.

    Raises:
        ValueError: If ``messages`` is empty or contains invalid entries.
        AuthenticationError: If no valid API key is found.
        RateLimitError: If the API returns a 429 Too Many Requests error.
        GeneratorError: For all other API errors (connection, timeout, server
            errors, empty responses, etc.).

    Example::

        from n8n_forge.prompt_builder import build_messages
        from n8n_forge.generator import generate_workflow_response

        messages = build_messages("Send a Slack alert every hour")
        raw = generate_workflow_response(messages, model="gpt-4o-mini")
    """
    if not messages:
        raise ValueError("messages list must not be empty.")

    for idx, msg in enumerate(messages):
        if not isinstance(msg, dict) or "role" not in msg or "content" not in msg:
            raise ValueError(
                f"Message at index {idx} must be a dict with 'role' and 'content' keys."
            )

    if not (0.0 <= temperature <= 2.0):
        raise ValueError(f"temperature must be in [0.0, 2.0], got {temperature}.")

    if max_tokens < 1:
        raise ValueError(f"max_tokens must be >= 1, got {max_tokens}.")

    logger.debug(
        "Sending request to OpenAI: model=%s, temperature=%s, max_tokens=%s, "
        "num_messages=%d",
        model,
        temperature,
        max_tokens,
        len(messages),
    )

    client = _create_client(api_key=api_key, timeout=timeout)

    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,  # type: ignore[arg-type]
            temperature=temperature,
            max_tokens=max_tokens,
        )
    except openai.AuthenticationError as exc:
        raise AuthenticationError(
            "OpenAI authentication failed. Check your API key.", cause=exc
        ) from exc
    except openai.RateLimitError as exc:
        raise RateLimitError(
            "OpenAI rate limit exceeded. Please wait before retrying.", cause=exc
        ) from exc
    except APITimeoutError as exc:
        raise GeneratorError(
            f"OpenAI request timed out after {timeout}s. "
            "Try again or increase the timeout.",
            cause=exc,
        ) from exc
    except APIConnectionError as exc:
        raise GeneratorError(
            "Failed to connect to the OpenAI API. Check your internet connection.",
            cause=exc,
        ) from exc
    except APIStatusError as exc:
        raise GeneratorError(
            f"OpenAI API returned an error (HTTP {exc.status_code}): {exc.message}",
            cause=exc,
        ) from exc
    except Exception as exc:  # noqa: BLE001
        raise GeneratorError(
            f"Unexpected error during OpenAI API call: {exc}", cause=exc
        ) from exc

    # Extract content from the response
    try:
        choice = response.choices[0]
        content = choice.message.content
    except (IndexError, AttributeError) as exc:
        raise GeneratorError(
            "OpenAI response was missing expected choice/message structure.",
            cause=exc,
        ) from exc

    if content is None or not content.strip():
        finish_reason = getattr(
            getattr(response.choices[0], "finish_reason", None), "value", None
        ) if response.choices else "unknown"
        raise EmptyResponseError(
            f"OpenAI returned an empty response (finish_reason={finish_reason}). "
            "The model may have hit a context or content filter limit."
        )

    logger.debug(
        "Received response: finish_reason=%s, content_length=%d",
        getattr(response.choices[0], "finish_reason", "unknown"),
        len(content),
    )

    return content


# ---------------------------------------------------------------------------
# Usage metadata helper
# ---------------------------------------------------------------------------


def get_response_metadata(response: Any) -> dict[str, Any]:
    """Extract token usage and finish-reason metadata from an OpenAI response.

    This is a utility function for logging and diagnostics. It is not called
    by :func:`generate_workflow_response` internally but is available for
    callers that retain the response object.

    Args:
        response: An ``openai.types.chat.ChatCompletion`` object returned by
            the OpenAI SDK.

    Returns:
        A dict with the following keys (all may be ``None`` if unavailable):

        - ``prompt_tokens``: Number of tokens in the prompt.
        - ``completion_tokens``: Number of tokens in the completion.
        - ``total_tokens``: Total tokens used.
        - ``finish_reason``: Why the model stopped generating.
        - ``model``: The model that was actually used.
    """
    metadata: dict[str, Any] = {
        "prompt_tokens": None,
        "completion_tokens": None,
        "total_tokens": None,
        "finish_reason": None,
        "model": None,
    }
    try:
        if hasattr(response, "usage") and response.usage:
            metadata["prompt_tokens"] = getattr(response.usage, "prompt_tokens", None)
            metadata["completion_tokens"] = getattr(
                response.usage, "completion_tokens", None
            )
            metadata["total_tokens"] = getattr(response.usage, "total_tokens", None)
        if hasattr(response, "choices") and response.choices:
            finish = response.choices[0].finish_reason
            metadata["finish_reason"] = (
                finish.value if hasattr(finish, "value") else finish
            )
        if hasattr(response, "model"):
            metadata["model"] = response.model
    except Exception:  # noqa: BLE001
        pass  # Best-effort — never raise from metadata extraction
    return metadata
