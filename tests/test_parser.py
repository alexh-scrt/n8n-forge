"""Unit tests for the n8n_forge.parser module.

Covers JSON extraction, cleaning, normalisation, schema validation,
and the full parse_workflow_response pipeline with various LLM response formats.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from n8n_forge.parser import (
    JSONDecodeError,
    JSONExtractionError,
    ParserError,
    _ensure_required_fields,
    _extract_outermost_object,
    _normalise_connections,
    _normalise_node_ids,
    _normalise_node_positions,
    _normalise_parameters,
    _normalise_type_versions,
    clean_json_string,
    extract_json_string,
    normalise_workflow_dict,
    parse_json_string,
    parse_workflow_response,
    workflow_to_json_string,
)
from n8n_forge.schema import WorkflowSchema, WorkflowValidationError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_minimal_workflow_dict(**overrides: Any) -> dict[str, Any]:
    """Return a minimal valid workflow dict."""
    base: dict[str, Any] = {
        "name": "Test Workflow",
        "nodes": [
            {
                "id": "node-trigger-001",
                "name": "Schedule Trigger",
                "type": "n8n-nodes-base.scheduleTrigger",
                "typeVersion": 1,
                "position": [250, 300],
                "parameters": {},
            },
            {
                "id": "node-action-002",
                "name": "HTTP Request",
                "type": "n8n-nodes-base.httpRequest",
                "typeVersion": 4,
                "position": [500, 300],
                "parameters": {"url": "https://example.com"},
            },
        ],
        "connections": {
            "Schedule Trigger": {
                "main": [
                    [{"node": "HTTP Request", "type": "main", "index": 0}]
                ]
            }
        },
        "active": False,
    }
    base.update(overrides)
    return base


def _make_minimal_workflow_json(**overrides: Any) -> str:
    """Return a minimal valid workflow JSON string."""
    return json.dumps(_make_minimal_workflow_dict(**overrides), indent=2)


def _wrap_in_fenced_block(json_str: str) -> str:
    """Wrap a JSON string in a fenced code block."""
    return f"```json\n{json_str}\n```"


# ---------------------------------------------------------------------------
# Tests for extract_json_string
# ---------------------------------------------------------------------------


class TestExtractJsonString:
    """Tests for the extract_json_string function."""

    def test_extracts_from_fenced_json_block(self) -> None:
        json_str = '{"name": "My Workflow"}'
        raw = f"Here is your workflow:\n```json\n{json_str}\n```"
        result = extract_json_string(raw)
        assert result == json_str

    def test_extracts_from_fenced_block_without_language_tag(self) -> None:
        json_str = '{"name": "My Workflow"}'
        raw = f"```\n{json_str}\n```"
        result = extract_json_string(raw)
        assert result == json_str

    def test_extracts_bare_json_object(self) -> None:
        json_str = '{"name": "My Workflow", "nodes": []}'
        result = extract_json_string(json_str)
        assert "{" in result
        assert "My Workflow" in result

    def test_extracts_json_from_mixed_prose_and_json(self) -> None:
        json_str = '{"name": "Workflow"}'
        raw = f"Here is the workflow for you:\n{json_str}\nHope this helps!"
        result = extract_json_string(raw)
        assert "Workflow" in result

    def test_extracts_nested_json_correctly(self) -> None:
        data = {"a": {"b": {"c": 1}}}
        raw = json.dumps(data)
        result = extract_json_string(raw)
        parsed = json.loads(result)
        assert parsed == data

    def test_raises_on_empty_response(self) -> None:
        with pytest.raises(JSONExtractionError, match="empty"):
            extract_json_string("")

    def test_raises_on_whitespace_only_response(self) -> None:
        with pytest.raises(JSONExtractionError):
            extract_json_string("   \n  ")

    def test_raises_when_no_json_present(self) -> None:
        with pytest.raises(JSONExtractionError, match="No JSON"):
            extract_json_string("Here is some plain text without any JSON at all.")

    def test_raises_when_only_array_present(self) -> None:
        """Arrays (not objects) should not be extracted as top-level."""
        with pytest.raises(JSONExtractionError):
            extract_json_string("[1, 2, 3]")

    def test_fenced_block_with_prose_before_and_after(self) -> None:
        json_str = '{"name": "Wf"}'
        raw = (
            "I've analysed your request. Here is the workflow:\n"
            f"```json\n{json_str}\n```\n"
            "Let me know if you need changes!"
        )
        result = extract_json_string(raw)
        assert "Wf" in result

    def test_multiline_fenced_json(self) -> None:
        data = _make_minimal_workflow_dict()
        json_str = json.dumps(data, indent=2)
        raw = _wrap_in_fenced_block(json_str)
        result = extract_json_string(raw)
        parsed = json.loads(result)
        assert parsed["name"] == "Test Workflow"


# ---------------------------------------------------------------------------
# Tests for _extract_outermost_object
# ---------------------------------------------------------------------------


class TestExtractOutermostObject:
    """Tests for the internal _extract_outermost_object helper."""

    def test_simple_object(self) -> None:
        result = _extract_outermost_object('{"a": 1}')
        assert result == '{"a": 1}'

    def test_nested_object(self) -> None:
        result = _extract_outermost_object('{"a": {"b": 2}}')
        assert result == '{"a": {"b": 2}}'

    def test_returns_none_when_no_object(self) -> None:
        result = _extract_outermost_object("no braces here")
        assert result is None

    def test_ignores_braces_in_strings(self) -> None:
        result = _extract_outermost_object('{"a": "value with } brace"}')
        assert result is not None
        parsed = json.loads(result)
        assert parsed["a"] == "value with } brace"

    def test_unbalanced_braces_returns_none(self) -> None:
        result = _extract_outermost_object("{")
        assert result is None

    def test_text_before_and_after_object(self) -> None:
        result = _extract_outermost_object('prefix {"key": "val"} suffix')
        assert result == '{"key": "val"}'


# ---------------------------------------------------------------------------
# Tests for clean_json_string
# ---------------------------------------------------------------------------


class TestCleanJsonString:
    """Tests for the clean_json_string function."""

    def test_removes_trailing_comma_before_closing_brace(self) -> None:
        raw = '{"a": 1,}'
        cleaned = clean_json_string(raw)
        assert cleaned == '{"a": 1}'

    def test_removes_trailing_comma_before_closing_bracket(self) -> None:
        raw = '[1, 2, 3,]'
        cleaned = clean_json_string(raw)
        assert cleaned == '[1, 2, 3]'

    def test_removes_nested_trailing_commas(self) -> None:
        raw = '{"a": [1, 2,], "b": {"c": 3,},}'
        cleaned = clean_json_string(raw)
        # Should be parseable after cleaning
        parsed = json.loads(cleaned)
        assert parsed["a"] == [1, 2]
        assert parsed["b"] == {"c": 3}

    def test_removes_single_line_js_comments(self) -> None:
        raw = '{"a": 1, // this is a comment\n"b": 2}'
        cleaned = clean_json_string(raw)
        assert "// this is a comment" not in cleaned

    def test_removes_multi_line_js_comments(self) -> None:
        raw = '{"a": /* inline comment */ 1}'
        cleaned = clean_json_string(raw)
        assert "/* inline comment */" not in cleaned

    def test_strips_leading_bom(self) -> None:
        raw = "\ufeff{\"a\": 1}"
        cleaned = clean_json_string(raw)
        assert not cleaned.startswith("\ufeff")
        assert cleaned.startswith("{")

    def test_strips_surrounding_whitespace(self) -> None:
        raw = "   {\"a\": 1}   "
        cleaned = clean_json_string(raw)
        assert cleaned == '{"a": 1}'

    def test_valid_json_unchanged(self) -> None:
        raw = '{"name": "Test", "active": false}'
        cleaned = clean_json_string(raw)
        assert json.loads(cleaned) == {"name": "Test", "active": False}

    def test_complex_workflow_json_cleaned_and_parseable(self) -> None:
        raw = json.dumps(_make_minimal_workflow_dict())
        cleaned = clean_json_string(raw)
        parsed = json.loads(cleaned)
        assert parsed["name"] == "Test Workflow"


# ---------------------------------------------------------------------------
# Tests for parse_json_string
# ---------------------------------------------------------------------------


class TestParseJsonString:
    """Tests for the parse_json_string function."""

    def test_parses_valid_json_object(self) -> None:
        result = parse_json_string('{"a": 1}')
        assert result == {"a": 1}

    def test_parses_nested_object(self) -> None:
        data = {"a": {"b": [1, 2, 3]}}
        result = parse_json_string(json.dumps(data))
        assert result == data

    def test_raises_on_invalid_json(self) -> None:
        with pytest.raises(JSONDecodeError, match="Failed to parse"):
            parse_json_string("{invalid json}")

    def test_raises_on_json_array(self) -> None:
        with pytest.raises(ParserError, match="dict"):
            parse_json_string("[1, 2, 3]")

    def test_raises_on_json_string(self) -> None:
        with pytest.raises(ParserError, match="dict"):
            parse_json_string('"just a string"')

    def test_raises_on_json_number(self) -> None:
        with pytest.raises(ParserError, match="dict"):
            parse_json_string("42")

    def test_raises_on_empty_string(self) -> None:
        with pytest.raises(JSONDecodeError):
            parse_json_string("")

    def test_raw_response_included_in_error(self) -> None:
        try:
            parse_json_string("{bad}", raw_response="original raw response")
        except JSONDecodeError as exc:
            assert exc.raw_response == "original raw response"


# ---------------------------------------------------------------------------
# Tests for normalisation helpers
# ---------------------------------------------------------------------------


class TestEnsureRequiredFields:
    """Tests for _ensure_required_fields."""

    def test_adds_missing_name(self) -> None:
        result = _ensure_required_fields({})
        assert result["name"] == "Generated Workflow"

    def test_adds_missing_nodes(self) -> None:
        result = _ensure_required_fields({})
        assert result["nodes"] == []

    def test_adds_missing_connections(self) -> None:
        result = _ensure_required_fields({})
        assert result["connections"] == {}

    def test_adds_missing_active(self) -> None:
        result = _ensure_required_fields({})
        assert result["active"] is False

    def test_does_not_overwrite_existing_fields(self) -> None:
        data = {"name": "My Custom Name", "active": True}
        result = _ensure_required_fields(data)
        assert result["name"] == "My Custom Name"
        assert result["active"] is True

    def test_adds_settings_when_missing(self) -> None:
        result = _ensure_required_fields({})
        assert "settings" in result
        assert result["settings"]["executionOrder"] == "v1"


class TestNormaliseNodeIds:
    """Tests for _normalise_node_ids."""

    def test_keeps_valid_ids(self) -> None:
        data = {"nodes": [{"id": "my-id", "name": "A"}]}
        result = _normalise_node_ids(data)
        assert result["nodes"][0]["id"] == "my-id"

    def test_generates_id_when_missing(self) -> None:
        data = {"nodes": [{"name": "A"}]}
        result = _normalise_node_ids(data)
        assert "id" in result["nodes"][0]
        assert result["nodes"][0]["id"]  # non-empty

    def test_generates_id_when_empty_string(self) -> None:
        data = {"nodes": [{"id": "", "name": "A"}]}
        result = _normalise_node_ids(data)
        assert result["nodes"][0]["id"]  # non-empty

    def test_generates_unique_ids_for_duplicates(self) -> None:
        data = {
            "nodes": [
                {"id": "same-id", "name": "A"},
                {"id": "same-id", "name": "B"},
            ]
        }
        result = _normalise_node_ids(data)
        ids = [n["id"] for n in result["nodes"]]
        assert len(ids) == len(set(ids))

    def test_empty_nodes_list_unchanged(self) -> None:
        data = {"nodes": []}
        result = _normalise_node_ids(data)
        assert result["nodes"] == []

    def test_non_list_nodes_unchanged(self) -> None:
        data = {"nodes": "not a list"}
        result = _normalise_node_ids(data)
        assert result["nodes"] == "not a list"


class TestNormaliseNodePositions:
    """Tests for _normalise_node_positions."""

    def test_keeps_valid_position(self) -> None:
        data = {"nodes": [{"position": [100, 200], "name": "A"}]}
        result = _normalise_node_positions(data)
        assert result["nodes"][0]["position"] == [100, 200]

    def test_fixes_missing_position(self) -> None:
        data = {"nodes": [{"name": "A"}]}
        result = _normalise_node_positions(data)
        pos = result["nodes"][0]["position"]
        assert isinstance(pos, list)
        assert len(pos) == 2

    def test_fixes_single_element_position(self) -> None:
        data = {"nodes": [{"position": [100], "name": "A"}]}
        result = _normalise_node_positions(data)
        pos = result["nodes"][0]["position"]
        assert len(pos) == 2

    def test_positions_spaced_horizontally(self) -> None:
        data = {
            "nodes": [
                {"name": "A"},
                {"name": "B"},
            ]
        }
        result = _normalise_node_positions(data)
        x0 = result["nodes"][0]["position"][0]
        x1 = result["nodes"][1]["position"][0]
        assert x1 > x0

    def test_fixes_non_numeric_position(self) -> None:
        data = {"nodes": [{"position": ["a", "b"], "name": "A"}]}
        result = _normalise_node_positions(data)
        pos = result["nodes"][0]["position"]
        assert all(isinstance(v, float) for v in pos)


class TestNormaliseTypeVersions:
    """Tests for _normalise_type_versions."""

    def test_keeps_valid_version(self) -> None:
        data = {"nodes": [{"typeVersion": 4}]}
        result = _normalise_type_versions(data)
        assert result["nodes"][0]["typeVersion"] == 4

    def test_defaults_missing_version_to_1(self) -> None:
        data = {"nodes": [{"name": "A"}]}
        result = _normalise_type_versions(data)
        assert result["nodes"][0]["typeVersion"] == 1

    def test_defaults_zero_version_to_1(self) -> None:
        data = {"nodes": [{"typeVersion": 0}]}
        result = _normalise_type_versions(data)
        assert result["nodes"][0]["typeVersion"] == 1

    def test_defaults_negative_version_to_1(self) -> None:
        data = {"nodes": [{"typeVersion": -1}]}
        result = _normalise_type_versions(data)
        assert result["nodes"][0]["typeVersion"] == 1

    def test_defaults_string_version_to_1(self) -> None:
        data = {"nodes": [{"typeVersion": "2"}]}
        result = _normalise_type_versions(data)
        assert result["nodes"][0]["typeVersion"] == 1


class TestNormaliseParameters:
    """Tests for _normalise_parameters."""

    def test_keeps_existing_parameters(self) -> None:
        data = {"nodes": [{"parameters": {"url": "https://example.com"}}]}
        result = _normalise_parameters(data)
        assert result["nodes"][0]["parameters"]["url"] == "https://example.com"

    def test_defaults_missing_parameters_to_empty_dict(self) -> None:
        data = {"nodes": [{"name": "A"}]}
        result = _normalise_parameters(data)
        assert result["nodes"][0]["parameters"] == {}

    def test_defaults_none_parameters_to_empty_dict(self) -> None:
        data = {"nodes": [{"parameters": None}]}
        result = _normalise_parameters(data)
        assert result["nodes"][0]["parameters"] == {}

    def test_defaults_list_parameters_to_empty_dict(self) -> None:
        data = {"nodes": [{"parameters": []}]}
        result = _normalise_parameters(data)
        assert result["nodes"][0]["parameters"] == {}


class TestNormaliseConnections:
    """Tests for _normalise_connections."""

    def _base_data(self) -> dict[str, Any]:
        return {
            "nodes": [
                {"name": "Schedule Trigger"},
                {"name": "HTTP Request"},
            ],
            "connections": {
                "Schedule Trigger": {
                    "main": [
                        [{"node": "HTTP Request", "type": "main", "index": 0}]
                    ]
                }
            },
        }

    def test_valid_connections_preserved(self) -> None:
        data = self._base_data()
        result = _normalise_connections(data)
        assert "Schedule Trigger" in result["connections"]
        assert len(result["connections"]["Schedule Trigger"]["main"][0]) == 1

    def test_drops_unknown_source_node(self) -> None:
        data = self._base_data()
        data["connections"]["Unknown Node"] = {
            "main": [[{"node": "HTTP Request", "type": "main", "index": 0}]]
        }
        result = _normalise_connections(data)
        assert "Unknown Node" not in result["connections"]

    def test_drops_unknown_target_node(self) -> None:
        data = self._base_data()
        data["connections"]["Schedule Trigger"]["main"][0].append(
            {"node": "Ghost Node", "type": "main", "index": 0}
        )
        result = _normalise_connections(data)
        target_nodes = [
            c["node"]
            for c in result["connections"]["Schedule Trigger"]["main"][0]
        ]
        assert "Ghost Node" not in target_nodes

    def test_empty_connections_unchanged(self) -> None:
        data = {
            "nodes": [{"name": "A"}],
            "connections": {},
        }
        result = _normalise_connections(data)
        assert result["connections"] == {}

    def test_adds_default_type_and_index_when_missing(self) -> None:
        data = {
            "nodes": [
                {"name": "Trigger"},
                {"name": "Action"},
            ],
            "connections": {
                "Trigger": {"main": [[{"node": "Action"}]]}
            },
        }
        result = _normalise_connections(data)
        conn = result["connections"]["Trigger"]["main"][0][0]
        assert conn["type"] == "main"
        assert conn["index"] == 0


class TestNormaliseWorkflowDict:
    """Tests for the composite normalise_workflow_dict function."""

    def test_fully_valid_dict_passes_through(self) -> None:
        data = _make_minimal_workflow_dict()
        result = normalise_workflow_dict(data)
        assert result["name"] == "Test Workflow"

    def test_empty_dict_gets_defaults(self) -> None:
        result = normalise_workflow_dict({})
        assert result["name"] == "Generated Workflow"
        assert result["nodes"] == []
        assert result["connections"] == {}
        assert result["active"] is False

    def test_nodes_without_ids_get_ids(self) -> None:
        data = {
            "nodes": [{"name": "A", "type": "n8n-nodes-base.manualTrigger"}]
        }
        result = normalise_workflow_dict(data)
        assert result["nodes"][0]["id"]

    def test_nodes_without_positions_get_positions(self) -> None:
        data = {
            "nodes": [{"id": "1", "name": "A", "type": "n8n-nodes-base.manualTrigger"}]
        }
        result = normalise_workflow_dict(data)
        pos = result["nodes"][0]["position"]
        assert isinstance(pos, list) and len(pos) == 2

    def test_unknown_connection_source_dropped(self) -> None:
        data = _make_minimal_workflow_dict()
        data["connections"]["Nonexistent"] = {"main": [[{"node": "HTTP Request", "type": "main", "index": 0}]]}
        result = normalise_workflow_dict(data)
        assert "Nonexistent" not in result["connections"]


# ---------------------------------------------------------------------------
# Tests for parse_workflow_response
# ---------------------------------------------------------------------------


class TestParseWorkflowResponse:
    """Tests for the top-level parse_workflow_response function."""

    def _make_raw_fenced(self, **overrides: Any) -> str:
        """Return a fenced-code-block LLM response with a minimal valid workflow."""
        return _wrap_in_fenced_block(_make_minimal_workflow_json(**overrides))

    def test_parses_fenced_json_response(self) -> None:
        raw = self._make_raw_fenced()
        workflow = parse_workflow_response(raw)
        assert isinstance(workflow, WorkflowSchema)
        assert workflow.name == "Test Workflow"

    def test_parses_bare_json_response(self) -> None:
        raw = _make_minimal_workflow_json()
        workflow = parse_workflow_response(raw)
        assert isinstance(workflow, WorkflowSchema)

    def test_parses_json_with_prose_around_it(self) -> None:
        json_str = _make_minimal_workflow_json()
        raw = f"Here is the workflow:\n```json\n{json_str}\n```\nEnjoy!"
        workflow = parse_workflow_response(raw)
        assert workflow.name == "Test Workflow"

    def test_returns_workflow_schema_instance(self) -> None:
        workflow = parse_workflow_response(self._make_raw_fenced())
        assert isinstance(workflow, WorkflowSchema)

    def test_workflow_has_correct_node_count(self) -> None:
        workflow = parse_workflow_response(self._make_raw_fenced())
        assert len(workflow.nodes) == 2

    def test_workflow_has_correct_connection_count(self) -> None:
        workflow = parse_workflow_response(self._make_raw_fenced())
        assert workflow.get_connection_count() == 1

    def test_trigger_nodes_detected(self) -> None:
        workflow = parse_workflow_response(self._make_raw_fenced())
        triggers = workflow.get_trigger_nodes()
        assert len(triggers) == 1
        assert triggers[0].name == "Schedule Trigger"

    def test_raises_on_empty_response(self) -> None:
        with pytest.raises(JSONExtractionError):
            parse_workflow_response("")

    def test_raises_on_no_json_in_response(self) -> None:
        with pytest.raises(JSONExtractionError):
            parse_workflow_response("Sorry, I cannot generate that workflow.")

    def test_raises_on_invalid_json(self) -> None:
        with pytest.raises(JSONDecodeError):
            parse_workflow_response("```json\n{invalid json here}\n```")

    def test_raises_validation_error_on_duplicate_node_ids(self) -> None:
        data = _make_minimal_workflow_dict()
        # Force duplicate IDs to trigger schema validation failure
        data["nodes"][0]["id"] = "same-id"
        data["nodes"][1]["id"] = "same-id"
        raw = _wrap_in_fenced_block(json.dumps(data))
        # Normalisation should resolve this, but if strict=True it should fail
        with pytest.raises(WorkflowValidationError):
            parse_workflow_response(raw, strict=True)

    def test_handles_trailing_comma_in_json(self) -> None:
        # Inject a trailing comma
        data = _make_minimal_workflow_dict()
        json_str = json.dumps(data, indent=2)
        # Manually inject a trailing comma
        json_with_comma = json_str.replace(
            '"active": false', '"active": false,'
        )
        raw = _wrap_in_fenced_block(json_with_comma)
        workflow = parse_workflow_response(raw)
        assert workflow.name == "Test Workflow"

    def test_normalises_missing_node_ids(self) -> None:
        data = _make_minimal_workflow_dict()
        del data["nodes"][0]["id"]
        raw = _wrap_in_fenced_block(json.dumps(data))
        workflow = parse_workflow_response(raw)
        assert workflow.nodes[0].id  # Should have been assigned a UUID

    def test_normalises_missing_positions(self) -> None:
        data = _make_minimal_workflow_dict()
        del data["nodes"][0]["position"]
        raw = _wrap_in_fenced_block(json.dumps(data))
        workflow = parse_workflow_response(raw)
        assert workflow.nodes[0].position is not None
        assert len(workflow.nodes[0].position) == 2

    def test_normalises_missing_parameters(self) -> None:
        data = _make_minimal_workflow_dict()
        del data["nodes"][0]["parameters"]
        raw = _wrap_in_fenced_block(json.dumps(data))
        workflow = parse_workflow_response(raw)
        assert workflow.nodes[0].parameters == {}

    def test_strict_mode_raises_on_missing_required_field(self) -> None:
        data = _make_minimal_workflow_dict()
        del data["name"]
        raw = _wrap_in_fenced_block(json.dumps(data))
        with pytest.raises(WorkflowValidationError):
            parse_workflow_response(raw, strict=True)

    def test_strict_mode_false_fills_missing_name(self) -> None:
        data = _make_minimal_workflow_dict()
        del data["name"]
        # Remove connections too so there's no reference validation issue
        data["connections"] = {}
        raw = _wrap_in_fenced_block(json.dumps(data))
        # Without strict mode, missing name gets a default
        workflow = parse_workflow_response(raw, strict=False)
        assert workflow.name == "Generated Workflow"

    def test_workflow_with_no_connections_is_valid(self) -> None:
        data = _make_minimal_workflow_dict()
        data["connections"] = {}
        raw = _wrap_in_fenced_block(json.dumps(data))
        workflow = parse_workflow_response(raw)
        assert workflow.get_connection_count() == 0

    def test_complex_multi_node_workflow(self) -> None:
        data = {
            "name": "Complex Workflow",
            "nodes": [
                {
                    "id": "n1",
                    "name": "Schedule Trigger",
                    "type": "n8n-nodes-base.scheduleTrigger",
                    "typeVersion": 1,
                    "position": [250, 300],
                    "parameters": {"rule": {}},
                },
                {
                    "id": "n2",
                    "name": "HTTP Request",
                    "type": "n8n-nodes-base.httpRequest",
                    "typeVersion": 4,
                    "position": [500, 300],
                    "parameters": {"url": "https://api.example.com"},
                },
                {
                    "id": "n3",
                    "name": "Code",
                    "type": "n8n-nodes-base.code",
                    "typeVersion": 2,
                    "position": [750, 300],
                    "parameters": {"jsCode": "return $input.all();"},
                },
                {
                    "id": "n4",
                    "name": "Slack",
                    "type": "n8n-nodes-base.slack",
                    "typeVersion": 2,
                    "position": [1000, 300],
                    "parameters": {"channel": "#general", "text": "Done"},
                },
            ],
            "connections": {
                "Schedule Trigger": {
                    "main": [[{"node": "HTTP Request", "type": "main", "index": 0}]]
                },
                "HTTP Request": {
                    "main": [[{"node": "Code", "type": "main", "index": 0}]]
                },
                "Code": {
                    "main": [[{"node": "Slack", "type": "main", "index": 0}]]
                },
            },
            "active": False,
        }
        raw = _wrap_in_fenced_block(json.dumps(data, indent=2))
        workflow = parse_workflow_response(raw)
        assert len(workflow.nodes) == 4
        assert workflow.get_connection_count() == 3
        assert workflow.get_trigger_nodes()[0].name == "Schedule Trigger"


# ---------------------------------------------------------------------------
# Tests for workflow_to_json_string
# ---------------------------------------------------------------------------


class TestWorkflowToJsonString:
    """Tests for the workflow_to_json_string helper."""

    def _make_workflow(self) -> WorkflowSchema:
        raw = _wrap_in_fenced_block(_make_minimal_workflow_json())
        return parse_workflow_response(raw)

    def test_returns_string(self) -> None:
        workflow = self._make_workflow()
        result = workflow_to_json_string(workflow)
        assert isinstance(result, str)

    def test_returns_valid_json(self) -> None:
        workflow = self._make_workflow()
        result = workflow_to_json_string(workflow)
        parsed = json.loads(result)
        assert isinstance(parsed, dict)

    def test_contains_workflow_name(self) -> None:
        workflow = self._make_workflow()
        result = workflow_to_json_string(workflow)
        assert "Test Workflow" in result

    def test_contains_nodes(self) -> None:
        workflow = self._make_workflow()
        result = workflow_to_json_string(workflow)
        parsed = json.loads(result)
        assert "nodes" in parsed
        assert len(parsed["nodes"]) == 2

    def test_contains_connections(self) -> None:
        workflow = self._make_workflow()
        result = workflow_to_json_string(workflow)
        parsed = json.loads(result)
        assert "connections" in parsed
        assert "Schedule Trigger" in parsed["connections"]

    def test_default_indent_is_2(self) -> None:
        workflow = self._make_workflow()
        result = workflow_to_json_string(workflow)
        # With indent=2, lines should start with 2-space indentation
        assert "  " in result

    def test_custom_indent(self) -> None:
        workflow = self._make_workflow()
        result_4 = workflow_to_json_string(workflow, indent=4)
        result_2 = workflow_to_json_string(workflow, indent=2)
        # 4-space indent produces longer output
        assert len(result_4) > len(result_2)

    def test_round_trip_produces_equivalent_workflow(self) -> None:
        workflow = self._make_workflow()
        json_str = workflow_to_json_string(workflow)
        # Parse it again
        workflow2 = parse_workflow_response(json_str)
        assert workflow2.name == workflow.name
        assert len(workflow2.nodes) == len(workflow.nodes)
        assert workflow2.get_connection_count() == workflow.get_connection_count()


# ---------------------------------------------------------------------------
# Tests for exception classes
# ---------------------------------------------------------------------------


class TestParserExceptions:
    """Tests for the custom parser exception classes."""

    def test_parser_error_is_exception(self) -> None:
        exc = ParserError("Something went wrong")
        assert isinstance(exc, Exception)

    def test_parser_error_str(self) -> None:
        exc = ParserError("Bad JSON")
        assert str(exc) == "Bad JSON"

    def test_parser_error_raw_response(self) -> None:
        exc = ParserError("Oops", raw_response="raw stuff")
        assert exc.raw_response == "raw stuff"

    def test_json_extraction_error_is_parser_error(self) -> None:
        exc = JSONExtractionError("No JSON found")
        assert isinstance(exc, ParserError)

    def test_json_decode_error_is_parser_error(self) -> None:
        exc = JSONDecodeError("Decode failed")
        assert isinstance(exc, ParserError)

    def test_parser_error_raw_response_default_empty(self) -> None:
        exc = ParserError("msg")
        assert exc.raw_response == ""
