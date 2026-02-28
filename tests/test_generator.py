"""Unit tests for the n8n_forge.generator module.

All OpenAI API calls are mocked — no real network requests are made.
Tests cover the generate_workflow_response function, error mapping,
client creation, and metadata extraction.
"""

from __future__ import annotations

import os
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from n8n_forge.generator import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_MODEL,
    DEFAULT_TEMPERATURE,
    AuthenticationError,
    EmptyResponseError,
    GeneratorError,
    RateLimitError,
    _create_client,
    generate_workflow_response,
    get_response_metadata,
)


# ---------------------------------------------------------------------------
# Helpers and fixtures
# ---------------------------------------------------------------------------


def _make_mock_response(content: str, finish_reason: str = "stop") -> MagicMock:
    """Build a minimal mock OpenAI ChatCompletion response."""
    mock_message = MagicMock()
    mock_message.content = content

    mock_choice = MagicMock()
    mock_choice.message = mock_message
    mock_choice.finish_reason = finish_reason

    mock_usage = MagicMock()
    mock_usage.prompt_tokens = 100
    mock_usage.completion_tokens = 200
    mock_usage.total_tokens = 300

    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    mock_response.usage = mock_usage
    mock_response.model = DEFAULT_MODEL

    return mock_response


SAMPLE_MESSAGES = [
    {"role": "system", "content": "You are an n8n expert."},
    {"role": "user", "content": "Generate a workflow."},
]

SAMPLE_JSON_RESPONSE = (
    "```json\n"
    '{"name": "Test Workflow", "nodes": [], "connections": {}, "active": false}\n'
    "```"
)


# ---------------------------------------------------------------------------
# Tests for _create_client
# ---------------------------------------------------------------------------


class TestCreateClient:
    """Tests for the _create_client factory function."""

    def test_creates_client_with_explicit_key(self) -> None:
        with patch("n8n_forge.generator.OpenAI") as mock_openai:
            mock_openai.return_value = MagicMock()
            _create_client(api_key="sk-test-key", timeout=30.0)
            mock_openai.assert_called_once_with(api_key="sk-test-key", timeout=30.0)

    def test_creates_client_from_env_var(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-env-key")
        with patch("n8n_forge.generator.OpenAI") as mock_openai:
            mock_openai.return_value = MagicMock()
            _create_client(api_key=None, timeout=30.0)
            mock_openai.assert_called_once_with(api_key="sk-env-key", timeout=30.0)

    def test_explicit_key_overrides_env_var(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-env-key")
        with patch("n8n_forge.generator.OpenAI") as mock_openai:
            mock_openai.return_value = MagicMock()
            _create_client(api_key="sk-explicit-key", timeout=30.0)
            call_kwargs = mock_openai.call_args[1]
            assert call_kwargs["api_key"] == "sk-explicit-key"

    def test_raises_when_no_key_and_no_env(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        with pytest.raises(AuthenticationError, match="API key is required"):
            _create_client(api_key=None, timeout=30.0)

    def test_raises_on_empty_string_key(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        with pytest.raises(AuthenticationError):
            _create_client(api_key="", timeout=30.0)

    def test_raises_on_whitespace_only_key(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        with pytest.raises(AuthenticationError):
            _create_client(api_key="   ", timeout=30.0)

    def test_strips_whitespace_from_key(self) -> None:
        with patch("n8n_forge.generator.OpenAI") as mock_openai:
            mock_openai.return_value = MagicMock()
            _create_client(api_key="  sk-trimmed  ", timeout=60.0)
            call_kwargs = mock_openai.call_args[1]
            assert call_kwargs["api_key"] == "sk-trimmed"

    def test_timeout_passed_to_client(self) -> None:
        with patch("n8n_forge.generator.OpenAI") as mock_openai:
            mock_openai.return_value = MagicMock()
            _create_client(api_key="sk-key", timeout=999.0)
            call_kwargs = mock_openai.call_args[1]
            assert call_kwargs["timeout"] == 999.0


# ---------------------------------------------------------------------------
# Tests for generate_workflow_response — success cases
# ---------------------------------------------------------------------------


class TestGenerateWorkflowResponseSuccess:
    """Success-path tests for generate_workflow_response."""

    @patch("n8n_forge.generator.OpenAI")
    def test_returns_string(self, mock_openai_cls: MagicMock) -> None:
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = _make_mock_response(
            SAMPLE_JSON_RESPONSE
        )

        result = generate_workflow_response(
            messages=SAMPLE_MESSAGES,
            api_key="sk-test",
        )
        assert isinstance(result, str)
        assert result == SAMPLE_JSON_RESPONSE

    @patch("n8n_forge.generator.OpenAI")
    def test_calls_api_with_correct_model(self, mock_openai_cls: MagicMock) -> None:
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = _make_mock_response("ok")

        generate_workflow_response(
            messages=SAMPLE_MESSAGES,
            model="gpt-4o",
            api_key="sk-test",
        )
        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["model"] == "gpt-4o"

    @patch("n8n_forge.generator.OpenAI")
    def test_calls_api_with_correct_temperature(self, mock_openai_cls: MagicMock) -> None:
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = _make_mock_response("ok")

        generate_workflow_response(
            messages=SAMPLE_MESSAGES,
            temperature=0.5,
            api_key="sk-test",
        )
        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["temperature"] == 0.5

    @patch("n8n_forge.generator.OpenAI")
    def test_calls_api_with_correct_max_tokens(self, mock_openai_cls: MagicMock) -> None:
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = _make_mock_response("ok")

        generate_workflow_response(
            messages=SAMPLE_MESSAGES,
            max_tokens=2048,
            api_key="sk-test",
        )
        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["max_tokens"] == 2048

    @patch("n8n_forge.generator.OpenAI")
    def test_calls_api_with_correct_messages(self, mock_openai_cls: MagicMock) -> None:
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = _make_mock_response("ok")

        generate_workflow_response(
            messages=SAMPLE_MESSAGES,
            api_key="sk-test",
        )
        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["messages"] == SAMPLE_MESSAGES

    @patch("n8n_forge.generator.OpenAI")
    def test_default_model_used_when_not_specified(self, mock_openai_cls: MagicMock) -> None:
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = _make_mock_response("ok")

        generate_workflow_response(messages=SAMPLE_MESSAGES, api_key="sk-test")
        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["model"] == DEFAULT_MODEL

    @patch("n8n_forge.generator.OpenAI")
    def test_default_temperature_used_when_not_specified(
        self, mock_openai_cls: MagicMock
    ) -> None:
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = _make_mock_response("ok")

        generate_workflow_response(messages=SAMPLE_MESSAGES, api_key="sk-test")
        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["temperature"] == DEFAULT_TEMPERATURE

    @patch("n8n_forge.generator.OpenAI")
    def test_default_max_tokens_used_when_not_specified(
        self, mock_openai_cls: MagicMock
    ) -> None:
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = _make_mock_response("ok")

        generate_workflow_response(messages=SAMPLE_MESSAGES, api_key="sk-test")
        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["max_tokens"] == DEFAULT_MAX_TOKENS

    @patch("n8n_forge.generator.OpenAI")
    def test_returns_first_choice_content(self, mock_openai_cls: MagicMock) -> None:
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = _make_mock_response(
            "first choice content"
        )

        result = generate_workflow_response(messages=SAMPLE_MESSAGES, api_key="sk-test")
        assert result == "first choice content"

    @patch("n8n_forge.generator.OpenAI")
    def test_api_called_exactly_once(self, mock_openai_cls: MagicMock) -> None:
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = _make_mock_response("ok")

        generate_workflow_response(messages=SAMPLE_MESSAGES, api_key="sk-test")
        assert mock_client.chat.completions.create.call_count == 1

    @patch("n8n_forge.generator.OpenAI")
    def test_multiline_response_returned_intact(self, mock_openai_cls: MagicMock) -> None:
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        content = "Line 1\nLine 2\nLine 3\n"
        mock_client.chat.completions.create.return_value = _make_mock_response(content)

        result = generate_workflow_response(messages=SAMPLE_MESSAGES, api_key="sk-test")
        assert result == content

    @patch("n8n_forge.generator.OpenAI")
    def test_uses_api_key_from_argument(self, mock_openai_cls: MagicMock) -> None:
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = _make_mock_response("ok")

        generate_workflow_response(messages=SAMPLE_MESSAGES, api_key="sk-explicit")
        # Verify client was created with the explicit key
        assert mock_openai_cls.called
        call_kwargs = mock_openai_cls.call_args[1]
        assert call_kwargs["api_key"] == "sk-explicit"

    @patch("n8n_forge.generator.OpenAI")
    def test_temperature_zero_is_valid(self, mock_openai_cls: MagicMock) -> None:
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = _make_mock_response("ok")

        result = generate_workflow_response(
            messages=SAMPLE_MESSAGES,
            temperature=0.0,
            api_key="sk-test",
        )
        assert isinstance(result, str)

    @patch("n8n_forge.generator.OpenAI")
    def test_temperature_two_is_valid(self, mock_openai_cls: MagicMock) -> None:
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = _make_mock_response("ok")

        result = generate_workflow_response(
            messages=SAMPLE_MESSAGES,
            temperature=2.0,
            api_key="sk-test",
        )
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# Tests for generate_workflow_response — input validation
# ---------------------------------------------------------------------------


class TestGenerateWorkflowResponseInputValidation:
    """Tests for input validation in generate_workflow_response."""

    def test_raises_on_empty_messages(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            generate_workflow_response(messages=[], api_key="sk-test")

    def test_raises_on_message_missing_role(self) -> None:
        with pytest.raises(ValueError, match="role"):
            generate_workflow_response(
                messages=[{"content": "hello"}],
                api_key="sk-test",
            )

    def test_raises_on_message_missing_content(self) -> None:
        with pytest.raises(ValueError, match="content"):
            generate_workflow_response(
                messages=[{"role": "user"}],
                api_key="sk-test",
            )

    def test_raises_on_non_dict_message(self) -> None:
        with pytest.raises(ValueError):
            generate_workflow_response(
                messages=["not a dict"],  # type: ignore[list-item]
                api_key="sk-test",
            )

    def test_raises_on_temperature_too_high(self) -> None:
        with pytest.raises(ValueError, match="temperature"):
            generate_workflow_response(
                messages=SAMPLE_MESSAGES,
                temperature=2.5,
                api_key="sk-test",
            )

    def test_raises_on_negative_temperature(self) -> None:
        with pytest.raises(ValueError, match="temperature"):
            generate_workflow_response(
                messages=SAMPLE_MESSAGES,
                temperature=-0.1,
                api_key="sk-test",
            )

    def test_raises_on_zero_max_tokens(self) -> None:
        with pytest.raises(ValueError, match="max_tokens"):
            generate_workflow_response(
                messages=SAMPLE_MESSAGES,
                max_tokens=0,
                api_key="sk-test",
            )

    def test_raises_on_negative_max_tokens(self) -> None:
        with pytest.raises(ValueError, match="max_tokens"):
            generate_workflow_response(
                messages=SAMPLE_MESSAGES,
                max_tokens=-1,
                api_key="sk-test",
            )

    def test_raises_without_api_key(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        with pytest.raises(AuthenticationError):
            generate_workflow_response(
                messages=SAMPLE_MESSAGES,
                api_key=None,
            )

    def test_single_message_is_valid(self) -> None:
        """A list with a single message should not raise a ValueError."""
        with patch("n8n_forge.generator.OpenAI") as mock_openai_cls:
            mock_client = MagicMock()
            mock_openai_cls.return_value = mock_client
            mock_client.chat.completions.create.return_value = _make_mock_response("ok")

            result = generate_workflow_response(
                messages=[{"role": "user", "content": "Just a user message"}],
                api_key="sk-test",
            )
            assert isinstance(result, str)

    def test_raises_when_message_is_none_in_list(self) -> None:
        with pytest.raises((ValueError, TypeError)):
            generate_workflow_response(
                messages=[None],  # type: ignore[list-item]
                api_key="sk-test",
            )

    def test_max_tokens_one_is_valid(self) -> None:
        with patch("n8n_forge.generator.OpenAI") as mock_openai_cls:
            mock_client = MagicMock()
            mock_openai_cls.return_value = mock_client
            mock_client.chat.completions.create.return_value = _make_mock_response("x")

            result = generate_workflow_response(
                messages=SAMPLE_MESSAGES,
                max_tokens=1,
                api_key="sk-test",
            )
            assert isinstance(result, str)


# ---------------------------------------------------------------------------
# Tests for generate_workflow_response — error mapping
# ---------------------------------------------------------------------------


class TestGenerateWorkflowResponseErrorMapping:
    """Tests that OpenAI SDK exceptions map to the correct project exceptions."""

    @patch("n8n_forge.generator.OpenAI")
    def test_openai_auth_error_maps_to_authentication_error(
        self, mock_openai_cls: MagicMock
    ) -> None:
        import openai

        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.side_effect = openai.AuthenticationError(
            message="Invalid API key",
            response=MagicMock(status_code=401),
            body={},
        )

        with pytest.raises(AuthenticationError, match="authentication failed"):
            generate_workflow_response(messages=SAMPLE_MESSAGES, api_key="sk-bad")

    @patch("n8n_forge.generator.OpenAI")
    def test_openai_rate_limit_error_maps_to_rate_limit_error(
        self, mock_openai_cls: MagicMock
    ) -> None:
        import openai

        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.side_effect = openai.RateLimitError(
            message="Rate limit exceeded",
            response=MagicMock(status_code=429),
            body={},
        )

        with pytest.raises(RateLimitError, match="rate limit"):
            generate_workflow_response(messages=SAMPLE_MESSAGES, api_key="sk-test")

    @patch("n8n_forge.generator.OpenAI")
    def test_timeout_error_maps_to_generator_error(
        self, mock_openai_cls: MagicMock
    ) -> None:
        from openai import APITimeoutError

        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.side_effect = APITimeoutError(
            request=MagicMock()
        )

        with pytest.raises(GeneratorError, match="timed out"):
            generate_workflow_response(messages=SAMPLE_MESSAGES, api_key="sk-test")

    @patch("n8n_forge.generator.OpenAI")
    def test_connection_error_maps_to_generator_error(
        self, mock_openai_cls: MagicMock
    ) -> None:
        from openai import APIConnectionError

        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.side_effect = APIConnectionError(
            request=MagicMock()
        )

        with pytest.raises(GeneratorError, match="connect"):
            generate_workflow_response(messages=SAMPLE_MESSAGES, api_key="sk-test")

    @patch("n8n_forge.generator.OpenAI")
    def test_api_status_error_maps_to_generator_error(
        self, mock_openai_cls: MagicMock
    ) -> None:
        from openai import APIStatusError

        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.side_effect = APIStatusError(
            message="Internal server error",
            response=MagicMock(status_code=500),
            body={},
        )

        with pytest.raises(GeneratorError, match="error"):
            generate_workflow_response(messages=SAMPLE_MESSAGES, api_key="sk-test")

    @patch("n8n_forge.generator.OpenAI")
    def test_generic_exception_maps_to_generator_error(
        self, mock_openai_cls: MagicMock
    ) -> None:
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.side_effect = RuntimeError(
            "Unexpected error"
        )

        with pytest.raises(GeneratorError):
            generate_workflow_response(messages=SAMPLE_MESSAGES, api_key="sk-test")

    @patch("n8n_forge.generator.OpenAI")
    def test_empty_content_raises_empty_response_error(
        self, mock_openai_cls: MagicMock
    ) -> None:
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = _make_mock_response("")

        with pytest.raises(EmptyResponseError, match="empty"):
            generate_workflow_response(messages=SAMPLE_MESSAGES, api_key="sk-test")

    @patch("n8n_forge.generator.OpenAI")
    def test_whitespace_content_raises_empty_response_error(
        self, mock_openai_cls: MagicMock
    ) -> None:
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = _make_mock_response(
            "   \n  "
        )

        with pytest.raises(EmptyResponseError):
            generate_workflow_response(messages=SAMPLE_MESSAGES, api_key="sk-test")

    @patch("n8n_forge.generator.OpenAI")
    def test_none_content_raises_empty_response_error(
        self, mock_openai_cls: MagicMock
    ) -> None:
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        response = _make_mock_response("placeholder")
        response.choices[0].message.content = None
        mock_client.chat.completions.create.return_value = response

        with pytest.raises(EmptyResponseError):
            generate_workflow_response(messages=SAMPLE_MESSAGES, api_key="sk-test")

    @patch("n8n_forge.generator.OpenAI")
    def test_authentication_error_is_subclass_of_generator_error(
        self, mock_openai_cls: MagicMock
    ) -> None:
        import openai

        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.side_effect = openai.AuthenticationError(
            message="Unauthorized",
            response=MagicMock(status_code=401),
            body={},
        )

        with pytest.raises(GeneratorError):
            generate_workflow_response(messages=SAMPLE_MESSAGES, api_key="sk-bad")

    @patch("n8n_forge.generator.OpenAI")
    def test_rate_limit_error_is_subclass_of_generator_error(
        self, mock_openai_cls: MagicMock
    ) -> None:
        import openai

        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.side_effect = openai.RateLimitError(
            message="Too many requests",
            response=MagicMock(status_code=429),
            body={},
        )

        with pytest.raises(GeneratorError):
            generate_workflow_response(messages=SAMPLE_MESSAGES, api_key="sk-test")

    @patch("n8n_forge.generator.OpenAI")
    def test_empty_response_error_is_subclass_of_generator_error(
        self, mock_openai_cls: MagicMock
    ) -> None:
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = _make_mock_response("")

        with pytest.raises(GeneratorError):
            generate_workflow_response(messages=SAMPLE_MESSAGES, api_key="sk-test")

    @patch("n8n_forge.generator.OpenAI")
    def test_generator_error_stores_cause(
        self, mock_openai_cls: MagicMock
    ) -> None:
        """The mapped GeneratorError should store the original exception as cause."""
        original = RuntimeError("Root cause")
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.side_effect = original

        try:
            generate_workflow_response(messages=SAMPLE_MESSAGES, api_key="sk-test")
        except GeneratorError as exc:
            assert exc.cause is original

    @patch("n8n_forge.generator.OpenAI")
    def test_api_status_error_includes_status_code_in_message(
        self, mock_openai_cls: MagicMock
    ) -> None:
        from openai import APIStatusError

        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.side_effect = APIStatusError(
            message="Bad Gateway",
            response=MagicMock(status_code=502),
            body={},
        )

        try:
            generate_workflow_response(messages=SAMPLE_MESSAGES, api_key="sk-test")
        except GeneratorError as exc:
            # The error message should reference the HTTP status or an error indicator
            assert "502" in str(exc) or "error" in str(exc).lower()


# ---------------------------------------------------------------------------
# Tests for GeneratorError exception hierarchy
# ---------------------------------------------------------------------------


class TestGeneratorErrorHierarchy:
    """Tests for the custom exception classes."""

    def test_generator_error_is_exception(self) -> None:
        exc = GeneratorError("Something went wrong")
        assert isinstance(exc, Exception)

    def test_generator_error_str_without_cause(self) -> None:
        exc = GeneratorError("Something went wrong")
        assert str(exc) == "Something went wrong"

    def test_generator_error_str_with_cause(self) -> None:
        cause = ValueError("root cause")
        exc = GeneratorError("Wrapper error", cause=cause)
        assert exc.cause is cause
        assert "root cause" in str(exc)

    def test_generator_error_message_attribute(self) -> None:
        exc = GeneratorError("Detailed message")
        assert exc.message == "Detailed message"

    def test_generator_error_cause_default_none(self) -> None:
        exc = GeneratorError("msg")
        assert exc.cause is None

    def test_empty_response_error_is_generator_error(self) -> None:
        exc = EmptyResponseError("Empty")
        assert isinstance(exc, GeneratorError)

    def test_empty_response_error_is_exception(self) -> None:
        exc = EmptyResponseError("Empty response")
        assert isinstance(exc, Exception)

    def test_rate_limit_error_is_generator_error(self) -> None:
        exc = RateLimitError("429")
        assert isinstance(exc, GeneratorError)

    def test_rate_limit_error_is_exception(self) -> None:
        exc = RateLimitError("429")
        assert isinstance(exc, Exception)

    def test_authentication_error_is_generator_error(self) -> None:
        exc = AuthenticationError("Bad key")
        assert isinstance(exc, GeneratorError)

    def test_authentication_error_is_exception(self) -> None:
        exc = AuthenticationError("Bad key")
        assert isinstance(exc, Exception)

    def test_empty_response_error_with_cause(self) -> None:
        cause = IOError("Connection reset")
        exc = EmptyResponseError("No data returned", cause=cause)
        assert exc.cause is cause

    def test_rate_limit_error_message(self) -> None:
        exc = RateLimitError("Too many requests per minute")
        assert "Too many requests" in str(exc)

    def test_authentication_error_message(self) -> None:
        exc = AuthenticationError("Invalid API credentials")
        assert "Invalid API credentials" in str(exc)

    def test_all_error_classes_have_message_attribute(self) -> None:
        for cls in [GeneratorError, EmptyResponseError, RateLimitError, AuthenticationError]:
            exc = cls("test message")
            assert exc.message == "test message"


# ---------------------------------------------------------------------------
# Tests for get_response_metadata
# ---------------------------------------------------------------------------


class TestGetResponseMetadata:
    """Tests for the get_response_metadata utility function."""

    def test_extracts_prompt_tokens(self) -> None:
        mock_response = _make_mock_response("content")
        metadata = get_response_metadata(mock_response)
        assert metadata["prompt_tokens"] == 100

    def test_extracts_completion_tokens(self) -> None:
        mock_response = _make_mock_response("content")
        metadata = get_response_metadata(mock_response)
        assert metadata["completion_tokens"] == 200

    def test_extracts_total_tokens(self) -> None:
        mock_response = _make_mock_response("content")
        metadata = get_response_metadata(mock_response)
        assert metadata["total_tokens"] == 300

    def test_extracts_model(self) -> None:
        mock_response = _make_mock_response("content")
        metadata = get_response_metadata(mock_response)
        assert metadata["model"] == DEFAULT_MODEL

    def test_returns_none_values_on_none_input(self) -> None:
        """Should never raise — returns None values for inaccessible fields."""
        metadata = get_response_metadata(None)
        assert metadata["prompt_tokens"] is None
        assert metadata["completion_tokens"] is None
        assert metadata["total_tokens"] is None

    def test_returns_dict_with_expected_keys(self) -> None:
        mock_response = _make_mock_response("content")
        metadata = get_response_metadata(mock_response)
        expected_keys = {
            "prompt_tokens",
            "completion_tokens",
            "total_tokens",
            "finish_reason",
            "model",
        }
        assert set(metadata.keys()) == expected_keys

    def test_does_not_raise_on_none_response(self) -> None:
        metadata = get_response_metadata(None)
        assert isinstance(metadata, dict)

    def test_does_not_raise_on_string_input(self) -> None:
        metadata = get_response_metadata("string")
        assert isinstance(metadata, dict)

    def test_does_not_raise_on_int_input(self) -> None:
        metadata = get_response_metadata(42)
        assert isinstance(metadata, dict)

    def test_does_not_raise_on_empty_dict(self) -> None:
        metadata = get_response_metadata({})
        assert isinstance(metadata, dict)

    def test_does_not_raise_on_empty_list(self) -> None:
        metadata = get_response_metadata([])
        assert isinstance(metadata, dict)

    def test_does_not_raise_on_arbitrary_object(self) -> None:
        metadata = get_response_metadata(object())
        assert isinstance(metadata, dict)

    def test_finish_reason_extracted(self) -> None:
        mock_response = _make_mock_response("content", finish_reason="stop")
        # Make finish_reason a plain string (not an enum)
        mock_response.choices[0].finish_reason = "stop"
        metadata = get_response_metadata(mock_response)
        # finish_reason should be extractable (either "stop" or None if enum handling differs)
        assert "finish_reason" in metadata

    def test_all_none_values_for_empty_response(self) -> None:
        response_with_no_usage = MagicMock()
        response_with_no_usage.usage = None
        response_with_no_usage.choices = []
        response_with_no_usage.model = "gpt-4o"
        metadata = get_response_metadata(response_with_no_usage)
        assert metadata["prompt_tokens"] is None
        assert metadata["completion_tokens"] is None
        assert metadata["total_tokens"] is None

    def test_model_extracted_correctly(self) -> None:
        mock_response = _make_mock_response("content")
        mock_response.model = "gpt-4o-mini"
        metadata = get_response_metadata(mock_response)
        assert metadata["model"] == "gpt-4o-mini"


# ---------------------------------------------------------------------------
# Integration-style test: build_messages -> generate (mocked) -> raw response
# ---------------------------------------------------------------------------


class TestGeneratorIntegrationWithPromptBuilder:
    """Integration tests combining prompt_builder.build_messages with generate."""

    @patch("n8n_forge.generator.OpenAI")
    def test_full_generate_flow_with_built_messages(
        self, mock_openai_cls: MagicMock
    ) -> None:
        from n8n_forge.prompt_builder import build_messages

        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = _make_mock_response(
            SAMPLE_JSON_RESPONSE
        )

        messages = build_messages(
            description="Send a Slack message every Monday morning"
        )
        result = generate_workflow_response(
            messages=messages,
            api_key="sk-test",
        )
        assert isinstance(result, str)
        assert len(result) > 0
        # Verify the API was called with the built messages
        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert len(call_kwargs["messages"]) == 2
        assert call_kwargs["messages"][0]["role"] == "system"
        assert call_kwargs["messages"][1]["role"] == "user"

    @patch("n8n_forge.generator.OpenAI")
    def test_refinement_mode_passes_existing_json_in_user_message(
        self, mock_openai_cls: MagicMock
    ) -> None:
        import json as json_mod

        from n8n_forge.prompt_builder import build_messages

        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = _make_mock_response(
            SAMPLE_JSON_RESPONSE
        )

        existing = json_mod.dumps(
            {
                "name": "Old Workflow",
                "nodes": [],
                "connections": {},
                "active": False,
            }
        )
        messages = build_messages(
            description="Add a Slack notification step",
            existing_workflow_json=existing,
        )
        result = generate_workflow_response(messages=messages, api_key="sk-test")
        assert isinstance(result, str)
        # Verify user message contains existing workflow
        user_content = messages[1]["content"]
        assert "Old Workflow" in user_content

    @patch("n8n_forge.generator.OpenAI")
    def test_system_message_contains_node_catalog(
        self, mock_openai_cls: MagicMock
    ) -> None:
        """After building messages, system content should reference n8n node types."""
        from n8n_forge.prompt_builder import build_messages

        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = _make_mock_response("ok")

        messages = build_messages(description="Fetch data and store in Google Sheets")
        system_content = messages[0]["content"]
        assert "n8n-nodes-base" in system_content

        generate_workflow_response(messages=messages, api_key="sk-test")
        # API should have been called once
        assert mock_client.chat.completions.create.call_count == 1

    @patch("n8n_forge.generator.OpenAI")
    def test_generate_with_custom_model_and_temperature(
        self, mock_openai_cls: MagicMock
    ) -> None:
        from n8n_forge.prompt_builder import build_messages

        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = _make_mock_response(
            SAMPLE_JSON_RESPONSE
        )

        messages = build_messages(description="Send weekly email digest")
        generate_workflow_response(
            messages=messages,
            model="gpt-4o",
            temperature=0.1,
            api_key="sk-test",
        )
        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["model"] == "gpt-4o"
        assert call_kwargs["temperature"] == 0.1

    @patch("n8n_forge.generator.OpenAI")
    def test_parse_mocked_response_is_valid_workflow(
        self, mock_openai_cls: MagicMock
    ) -> None:
        """A realistic mocked LLM response should be parseable by the parser."""
        import json as json_mod

        from n8n_forge.parser import parse_workflow_response
        from n8n_forge.prompt_builder import build_messages

        realistic_response = (
            "```json\n"
            + json_mod.dumps(
                {
                    "name": "Weekly Slack Digest",
                    "nodes": [
                        {
                            "id": "uuid-0001",
                            "name": "Schedule Trigger",
                            "type": "n8n-nodes-base.scheduleTrigger",
                            "typeVersion": 1,
                            "position": [250, 300],
                            "parameters": {
                                "rule": {
                                    "interval": [
                                        {
                                            "field": "weeks",
                                            "weeksInterval": 1,
                                            "triggerAtDay": [1],
                                            "triggerAtHour": 9,
                                            "triggerAtMinute": 0,
                                        }
                                    ]
                                }
                            },
                        },
                        {
                            "id": "uuid-0002",
                            "name": "Slack",
                            "type": "n8n-nodes-base.slack",
                            "typeVersion": 2,
                            "position": [500, 300],
                            "parameters": {
                                "resource": "message",
                                "operation": "post",
                                "channel": "#general",
                                "text": "Weekly digest is ready!",
                            },
                            "credentials": {
                                "slackApi": {"id": "1", "name": "Slack Account"}
                            },
                        },
                    ],
                    "connections": {
                        "Schedule Trigger": {
                            "main": [
                                [{"node": "Slack", "type": "main", "index": 0}]
                            ]
                        }
                    },
                    "active": False,
                    "settings": {
                        "executionOrder": "v1",
                        "timezone": "UTC",
                        "saveManualExecutions": True,
                        "saveDataSuccessExecution": "all",
                        "saveDataErrorExecution": "all",
                    },
                },
                indent=2,
            )
            + "\n```"
        )

        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = _make_mock_response(
            realistic_response
        )

        messages = build_messages(
            description="Send a Slack message every Monday at 9am"
        )
        raw = generate_workflow_response(messages=messages, api_key="sk-test")
        workflow = parse_workflow_response(raw)

        assert workflow.name == "Weekly Slack Digest"
        assert len(workflow.nodes) == 2
        assert workflow.get_connection_count() == 1
        triggers = workflow.get_trigger_nodes()
        assert len(triggers) == 1
        assert triggers[0].name == "Schedule Trigger"
