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
            client = _create_client(api_key="sk-test-key", timeout=30.0)
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
    def test_default_temperature_used_when_not_specified(self, mock_openai_cls: MagicMock) -> None:
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = _make_mock_response("ok")

        generate_workflow_response(messages=SAMPLE_MESSAGES, api_key="sk-test")
        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["temperature"] == DEFAULT_TEMPERATURE

    @patch("n8n_forge.generator.OpenAI")
    def test_returns_first_choice_content(self, mock_openai_cls: MagicMock) -> None:
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = _make_mock_response(
            "first choice content"
        )

        result = generate_workflow_response(messages=SAMPLE_MESSAGES, api_key="sk-test")
        assert result == "first choice content"


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
        mock_client.chat.completions.create.return_value = _make_mock_response("   \n  ")

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


# ---------------------------------------------------------------------------
# Tests for GeneratorError exception hierarchy
# ---------------------------------------------------------------------------


class TestGeneratorErrorHierarchy:
    """Tests for the custom exception classes."""

    def test_generator_error_is_exception(self) -> None:
        exc = GeneratorError("Something went wrong")
        assert isinstance(exc, Exception)

    def test_generator_error_str(self) -> None:
        exc = GeneratorError("Something went wrong")
        assert str(exc) == "Something went wrong"

    def test_generator_error_with_cause(self) -> None:
        cause = ValueError("root cause")
        exc = GeneratorError("Wrapper error", cause=cause)
        assert exc.cause is cause
        assert "root cause" in str(exc)

    def test_empty_response_error_is_generator_error(self) -> None:
        exc = EmptyResponseError("Empty")
        assert isinstance(exc, GeneratorError)

    def test_rate_limit_error_is_generator_error(self) -> None:
        exc = RateLimitError("429")
        assert isinstance(exc, GeneratorError)

    def test_authentication_error_is_generator_error(self) -> None:
        exc = AuthenticationError("Bad key")
        assert isinstance(exc, GeneratorError)

    def test_generator_error_cause_default_none(self) -> None:
        exc = GeneratorError("msg")
        assert exc.cause is None


# ---------------------------------------------------------------------------
# Tests for get_response_metadata
# ---------------------------------------------------------------------------


class TestGetResponseMetadata:
    """Tests for the get_response_metadata utility function."""

    def test_extracts_token_counts(self) -> None:
        mock_response = _make_mock_response("content")
        metadata = get_response_metadata(mock_response)
        assert metadata["prompt_tokens"] == 100
        assert metadata["completion_tokens"] == 200
        assert metadata["total_tokens"] == 300

    def test_extracts_model(self) -> None:
        mock_response = _make_mock_response("content")
        metadata = get_response_metadata(mock_response)
        assert metadata["model"] == DEFAULT_MODEL

    def test_returns_none_values_on_bad_response(self) -> None:
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

    def test_does_not_raise_on_malformed_response(self) -> None:
        """get_response_metadata must never raise regardless of input."""
        for bad_input in [None, "string", 42, {}, [], object()]:
            metadata = get_response_metadata(bad_input)
            assert isinstance(metadata, dict)


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
