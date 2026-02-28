"""Unit tests for the n8n_forge.prompt_builder module.

Covers prompt construction, node selection, system prompt rendering,
user message building, and error handling.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from n8n_forge.prompt_builder import (
    _get_example_workflow_json,
    _select_relevant_nodes,
    build_messages,
    build_user_message,
    render_system_prompt,
)


# ---------------------------------------------------------------------------
# Tests for _select_relevant_nodes
# ---------------------------------------------------------------------------


class TestSelectRelevantNodes:
    """Tests for the internal _select_relevant_nodes function."""

    def test_always_includes_trigger_nodes(self) -> None:
        nodes = _select_relevant_nodes("do something")
        type_names = [n.type_name for n in nodes]
        assert "n8n-nodes-base.scheduleTrigger" in type_names
        assert "n8n-nodes-base.webhook" in type_names

    def test_slack_description_includes_slack_node(self) -> None:
        nodes = _select_relevant_nodes("Send a Slack message")
        type_names = [n.type_name for n in nodes]
        assert "n8n-nodes-base.slack" in type_names

    def test_google_sheets_description_includes_sheets_node(self) -> None:
        nodes = _select_relevant_nodes("add data to Google Sheets")
        type_names = [n.type_name for n in nodes]
        assert "n8n-nodes-base.googleSheets" in type_names

    def test_returns_list_of_node_catalog_entries(self) -> None:
        from n8n_forge.node_catalog import NodeCatalogEntry
        nodes = _select_relevant_nodes("process emails")
        assert isinstance(nodes, list)
        for node in nodes:
            assert isinstance(node, NodeCatalogEntry)

    def test_result_capped_at_max_catalog_nodes(self) -> None:
        from n8n_forge.prompt_builder import _MAX_CATALOG_NODES
        nodes = _select_relevant_nodes("all nodes in the entire catalog")
        assert len(nodes) <= _MAX_CATALOG_NODES

    def test_no_duplicate_type_names(self) -> None:
        nodes = _select_relevant_nodes("schedule google sheets slack email")
        type_names = [n.type_name for n in nodes]
        assert len(type_names) == len(set(type_names))

    def test_includes_core_utility_nodes(self) -> None:
        nodes = _select_relevant_nodes("complex data transformation")
        type_names = [n.type_name for n in nodes]
        # At minimum some core nodes should be present
        core_types = {
            "n8n-nodes-base.code",
            "n8n-nodes-base.set",
            "n8n-nodes-base.if",
        }
        assert any(t in type_names for t in core_types)


# ---------------------------------------------------------------------------
# Tests for _get_example_workflow_json
# ---------------------------------------------------------------------------


class TestGetExampleWorkflowJson:
    """Tests for the example workflow JSON generator."""

    def test_returns_valid_json_string(self) -> None:
        example = _get_example_workflow_json()
        assert isinstance(example, str)
        # Must be parseable JSON
        parsed = json.loads(example)
        assert isinstance(parsed, dict)

    def test_example_has_required_fields(self) -> None:
        example = json.loads(_get_example_workflow_json())
        assert "name" in example
        assert "nodes" in example
        assert "connections" in example
        assert "active" in example

    def test_example_has_at_least_two_nodes(self) -> None:
        example = json.loads(_get_example_workflow_json())
        assert len(example["nodes"]) >= 2

    def test_example_has_trigger_node(self) -> None:
        example = json.loads(_get_example_workflow_json())
        types = [n["type"] for n in example["nodes"]]
        assert any("trigger" in t.lower() for t in types)

    def test_example_has_connections(self) -> None:
        example = json.loads(_get_example_workflow_json())
        assert len(example["connections"]) > 0

    def test_example_connection_format_is_valid(self) -> None:
        example = json.loads(_get_example_workflow_json())
        for source, output_map in example["connections"].items():
            assert "main" in output_map
            for group in output_map["main"]:
                assert isinstance(group, list)
                for conn in group:
                    assert "node" in conn
                    assert "type" in conn
                    assert "index" in conn

    def test_example_node_ids_are_unique(self) -> None:
        example = json.loads(_get_example_workflow_json())
        ids = [n["id"] for n in example["nodes"]]
        assert len(ids) == len(set(ids))


# ---------------------------------------------------------------------------
# Tests for render_system_prompt
# ---------------------------------------------------------------------------


class TestRenderSystemPrompt:
    """Tests for the render_system_prompt function."""

    def test_returns_string(self) -> None:
        result = render_system_prompt(node_catalog_text="Some catalog text")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_catalog_text_appears_in_output(self) -> None:
        catalog_text = "UNIQUE_CATALOG_MARKER_XYZ"
        result = render_system_prompt(node_catalog_text=catalog_text)
        assert catalog_text in result

    def test_example_json_appears_when_provided(self) -> None:
        example = '{"name": "Test", "nodes": []}'
        result = render_system_prompt(
            node_catalog_text="catalog",
            example_workflow_json=example,
        )
        assert '"name": "Test"' in result

    def test_example_section_absent_when_not_provided(self) -> None:
        result = render_system_prompt(
            node_catalog_text="catalog",
            example_workflow_json=None,
        )
        # The example section should not appear
        assert "EXAMPLE OUTPUT" not in result

    def test_output_contains_n8n_structure_guidance(self) -> None:
        result = render_system_prompt(node_catalog_text="catalog")
        assert "nodes" in result
        assert "connections" in result

    def test_output_contains_json_instruction(self) -> None:
        result = render_system_prompt(node_catalog_text="catalog")
        # Must instruct LLM to output JSON only
        assert "json" in result.lower()

    def test_output_contains_trigger_guidance(self) -> None:
        result = render_system_prompt(node_catalog_text="catalog")
        assert "trigger" in result.lower()

    def test_output_contains_checklist(self) -> None:
        result = render_system_prompt(node_catalog_text="catalog")
        assert "CHECKLIST" in result or "checklist" in result.lower() or "[ ]" in result

    def test_bad_templates_dir_raises_runtime_error(self) -> None:
        import n8n_forge.prompt_builder as pb
        original_dir = pb._TEMPLATES_DIR
        from pathlib import Path
        pb._TEMPLATES_DIR = Path("/nonexistent/path/to/templates")
        try:
            with pytest.raises(RuntimeError, match="Templates directory not found"):
                pb._get_jinja_env()
        finally:
            pb._TEMPLATES_DIR = original_dir


# ---------------------------------------------------------------------------
# Tests for build_user_message
# ---------------------------------------------------------------------------


class TestBuildUserMessage:
    """Tests for the build_user_message function."""

    def test_fresh_generation_contains_description(self) -> None:
        desc = "Send a daily Slack digest"
        msg = build_user_message(description=desc)
        assert desc in msg

    def test_fresh_generation_no_existing_workflow(self) -> None:
        msg = build_user_message(description="Do something")
        # Should not mention existing workflow
        assert "existing" not in msg.lower() or "refine" not in msg.lower()

    def test_refinement_contains_description(self) -> None:
        existing = '{"name": "Old Workflow", "nodes": []}'
        desc = "Also send to email"
        msg = build_user_message(description=desc, existing_workflow_json=existing)
        assert desc in msg

    def test_refinement_contains_existing_json(self) -> None:
        existing = '{"name": "Old Workflow", "nodes": []}'
        msg = build_user_message(
            description="Refine it",
            existing_workflow_json=existing,
        )
        assert "Old Workflow" in msg

    def test_refinement_mentions_refine_or_update(self) -> None:
        existing = '{"name": "Wf", "nodes": []}'
        msg = build_user_message(
            description="Add a filter step",
            existing_workflow_json=existing,
        )
        assert any(word in msg.lower() for word in ["refine", "update", "existing"])

    def test_description_is_stripped(self) -> None:
        msg = build_user_message(description="  Send a report  ")
        assert "Send a report" in msg
        # Should not have extra spaces around the description
        assert "  Send a report  " not in msg


# ---------------------------------------------------------------------------
# Tests for build_messages
# ---------------------------------------------------------------------------


class TestBuildMessages:
    """Tests for the top-level build_messages function."""

    def test_returns_list_of_dicts(self) -> None:
        messages = build_messages(description="Schedule a weekly report")
        assert isinstance(messages, list)
        assert len(messages) == 2

    def test_first_message_is_system(self) -> None:
        messages = build_messages(description="Do something")
        assert messages[0]["role"] == "system"

    def test_second_message_is_user(self) -> None:
        messages = build_messages(description="Do something")
        assert messages[1]["role"] == "user"

    def test_system_message_has_content(self) -> None:
        messages = build_messages(description="Do something")
        assert isinstance(messages[0]["content"], str)
        assert len(messages[0]["content"]) > 0

    def test_user_message_has_content(self) -> None:
        messages = build_messages(description="Do something")
        assert isinstance(messages[1]["content"], str)
        assert len(messages[1]["content"]) > 0

    def test_user_message_contains_description(self) -> None:
        desc = "Check competitor prices weekly"
        messages = build_messages(description=desc)
        assert desc in messages[1]["content"]

    def test_system_message_contains_catalog_text(self) -> None:
        messages = build_messages(description="Send Slack alerts")
        system_content = messages[0]["content"]
        # System prompt should have node catalog information
        assert "n8n-nodes-base" in system_content

    def test_system_message_contains_schedule_trigger(self) -> None:
        messages = build_messages(description="Run every Monday morning")
        system_content = messages[0]["content"]
        assert "scheduleTrigger" in system_content

    def test_empty_description_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            build_messages(description="")

    def test_whitespace_description_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            build_messages(description="   ")

    def test_refinement_mode_includes_existing_json_in_user_message(self) -> None:
        existing = json.dumps({"name": "Old Workflow", "nodes": [], "connections": {}, "active": False})
        messages = build_messages(
            description="Add an email step",
            existing_workflow_json=existing,
        )
        assert "Old Workflow" in messages[1]["content"]

    def test_refinement_mode_system_prompt_unchanged(self) -> None:
        """System prompt should be same structure for both fresh and refine modes."""
        fresh_messages = build_messages(description="Do something")
        refine_messages = build_messages(
            description="Do something",
            existing_workflow_json='{"name": "x", "nodes": [], "connections": {}, "active": false}',
        )
        # System prompts should both contain catalog info
        assert "n8n-nodes-base" in fresh_messages[0]["content"]
        assert "n8n-nodes-base" in refine_messages[0]["content"]

    def test_model_context_hint_affects_node_selection(self) -> None:
        """A context hint should influence which nodes appear in the system prompt."""
        messages_with_hint = build_messages(
            description="Automate sales process",
            model_context_hint="salesforce crm",
        )
        system_content = messages_with_hint[0]["content"]
        # Salesforce-related content should appear
        assert "salesforce" in system_content.lower() or "Salesforce" in system_content

    def test_messages_have_only_role_and_content_keys(self) -> None:
        messages = build_messages(description="Some task")
        for msg in messages:
            assert set(msg.keys()) == {"role", "content"}

    def test_google_sheets_description_has_sheets_in_system(self) -> None:
        messages = build_messages(
            description="Read data from Google Sheets and process it"
        )
        system_content = messages[0]["content"]
        assert "googleSheets" in system_content or "Google Sheets" in system_content

    def test_email_description_has_email_node_in_system(self) -> None:
        messages = build_messages(
            description="Send an email to customers with their invoice"
        )
        system_content = messages[0]["content"]
        assert "email" in system_content.lower()

    def test_system_prompt_contains_connection_structure_docs(self) -> None:
        messages = build_messages(description="Do anything")
        system_content = messages[0]["content"]
        assert "connections" in system_content
        assert "main" in system_content
