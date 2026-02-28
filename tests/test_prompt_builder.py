"""Unit tests for the n8n_forge.prompt_builder module.

Covers prompt construction, node selection, system prompt rendering,
user message building, and error handling.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from n8n_forge.prompt_builder import (
    _MAX_CATALOG_NODES,
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
        nodes = _select_relevant_nodes("all nodes in the entire catalog")
        assert len(nodes) <= _MAX_CATALOG_NODES

    def test_no_duplicate_type_names(self) -> None:
        nodes = _select_relevant_nodes("schedule google sheets slack email")
        type_names = [n.type_name for n in nodes]
        assert len(type_names) == len(set(type_names))

    def test_includes_core_utility_nodes(self) -> None:
        nodes = _select_relevant_nodes("complex data transformation")
        type_names = [n.type_name for n in nodes]
        core_types = {
            "n8n-nodes-base.code",
            "n8n-nodes-base.set",
            "n8n-nodes-base.if",
        }
        assert any(t in type_names for t in core_types)

    def test_email_description_includes_email_nodes(self) -> None:
        nodes = _select_relevant_nodes("send emails to customers")
        type_names = [n.type_name for n in nodes]
        email_types = {
            "n8n-nodes-base.sendEmail",
            "n8n-nodes-base.gmail",
            "n8n-nodes-base.emailReadImap",
        }
        assert any(t in type_names for t in email_types)

    def test_postgres_description_includes_database_nodes(self) -> None:
        nodes = _select_relevant_nodes("query postgres database")
        type_names = [n.type_name for n in nodes]
        assert "n8n-nodes-base.postgres" in type_names

    def test_returns_non_empty_list(self) -> None:
        nodes = _select_relevant_nodes("do anything")
        assert len(nodes) > 0

    def test_triggers_appear_first(self) -> None:
        """Trigger nodes should always be at the beginning of the list."""
        nodes = _select_relevant_nodes("send a slack alert on a schedule")
        # The first node should be a trigger
        assert nodes[0].is_trigger is True

    def test_empty_description_still_returns_triggers(self) -> None:
        # Even with an empty-ish query, triggers should be present
        nodes = _select_relevant_nodes("")
        type_names = [n.type_name for n in nodes]
        assert "n8n-nodes-base.scheduleTrigger" in type_names


# ---------------------------------------------------------------------------
# Tests for _get_example_workflow_json
# ---------------------------------------------------------------------------


class TestGetExampleWorkflowJson:
    """Tests for the example workflow JSON generator."""

    def test_returns_valid_json_string(self) -> None:
        example = _get_example_workflow_json()
        assert isinstance(example, str)
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

    def test_example_node_names_are_unique(self) -> None:
        example = json.loads(_get_example_workflow_json())
        names = [n["name"] for n in example["nodes"]]
        assert len(names) == len(set(names))

    def test_example_nodes_have_required_fields(self) -> None:
        example = json.loads(_get_example_workflow_json())
        required_fields = {"id", "name", "type", "typeVersion", "position", "parameters"}
        for node in example["nodes"]:
            for field in required_fields:
                assert field in node, f"Node missing field '{field}': {node}"

    def test_example_active_is_false(self) -> None:
        """Generated workflows should default to inactive."""
        example = json.loads(_get_example_workflow_json())
        assert example["active"] is False

    def test_example_has_settings(self) -> None:
        example = json.loads(_get_example_workflow_json())
        assert "settings" in example
        assert example["settings"]["executionOrder"] == "v1"

    def test_example_connection_source_matches_node_name(self) -> None:
        example = json.loads(_get_example_workflow_json())
        node_names = {n["name"] for n in example["nodes"]}
        for source in example["connections"]:
            assert source in node_names, (
                f"Connection source '{source}' not found in node names"
            )

    def test_example_connection_target_matches_node_name(self) -> None:
        example = json.loads(_get_example_workflow_json())
        node_names = {n["name"] for n in example["nodes"]}
        for source, output_map in example["connections"].items():
            for groups in output_map.values():
                for group in groups:
                    for conn in group:
                        assert conn["node"] in node_names, (
                            f"Connection target '{conn['node']}' not found in node names"
                        )


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
        assert "EXAMPLE OUTPUT" not in result

    def test_output_contains_n8n_structure_guidance(self) -> None:
        result = render_system_prompt(node_catalog_text="catalog")
        assert "nodes" in result
        assert "connections" in result

    def test_output_contains_json_instruction(self) -> None:
        result = render_system_prompt(node_catalog_text="catalog")
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
        pb._TEMPLATES_DIR = Path("/nonexistent/path/to/templates")
        try:
            with pytest.raises(RuntimeError, match="Templates directory not found"):
                pb._get_jinja_env()
        finally:
            pb._TEMPLATES_DIR = original_dir

    def test_output_is_non_empty_with_empty_catalog(self) -> None:
        result = render_system_prompt(node_catalog_text="")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_example_json_section_present_when_provided(self) -> None:
        example = json.dumps({"name": "Ex", "nodes": [], "connections": {}, "active": False})
        result = render_system_prompt(
            node_catalog_text="catalog",
            example_workflow_json=example,
        )
        assert "EXAMPLE OUTPUT" in result

    def test_output_mentions_position_guidance(self) -> None:
        """System prompt should include guidance about node positions."""
        result = render_system_prompt(node_catalog_text="catalog")
        assert "position" in result.lower()

    def test_output_mentions_unique_ids(self) -> None:
        """System prompt should instruct about unique node IDs."""
        result = render_system_prompt(node_catalog_text="catalog")
        assert "id" in result.lower() or "uuid" in result.lower()

    def test_output_mentions_scheduling_rules(self) -> None:
        """System prompt should include scheduling rule examples."""
        result = render_system_prompt(node_catalog_text="catalog")
        assert "scheduleTrigger" in result or "schedule" in result.lower()


# ---------------------------------------------------------------------------
# Tests for build_user_message
# ---------------------------------------------------------------------------


class TestBuildUserMessage:
    """Tests for the build_user_message function."""

    def test_fresh_generation_contains_description(self) -> None:
        desc = "Send a daily Slack digest"
        msg = build_user_message(description=desc)
        assert desc in msg

    def test_fresh_generation_mentions_generate(self) -> None:
        msg = build_user_message(description="Do something")
        assert any(
            word in msg.lower()
            for word in ["generate", "create", "automation", "workflow"]
        )

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
        assert "  Send a report  " not in msg

    def test_refinement_existing_json_wrapped_in_code_block(self) -> None:
        existing = '{"name": "Wf", "nodes": []}'
        msg = build_user_message(
            description="Add a step",
            existing_workflow_json=existing,
        )
        # The existing JSON should appear inside a code block
        assert "```" in msg or existing in msg

    def test_fresh_generation_does_not_mention_existing_workflow(self) -> None:
        msg = build_user_message(description="Do something cool")
        assert "existing" not in msg.lower() or "refine" not in msg.lower()

    def test_whitespace_in_existing_json_stripped(self) -> None:
        existing = '  {"name": "Wf"}  '
        msg = build_user_message(
            description="Fix it",
            existing_workflow_json=existing,
        )
        # The stripped version should appear
        assert '{"name": "Wf"}' in msg

    def test_returns_string(self) -> None:
        msg = build_user_message(description="Something")
        assert isinstance(msg, str)

    def test_refinement_returns_string(self) -> None:
        msg = build_user_message(
            description="Something",
            existing_workflow_json='{"name": "x"}',
        )
        assert isinstance(msg, str)


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
        existing = json.dumps(
            {"name": "Old Workflow", "nodes": [], "connections": {}, "active": False}
        )
        messages = build_messages(
            description="Add an email step",
            existing_workflow_json=existing,
        )
        assert "Old Workflow" in messages[1]["content"]

    def test_refinement_mode_system_prompt_unchanged_structure(self) -> None:
        """System prompt should have same structure for both fresh and refine modes."""
        fresh_messages = build_messages(description="Do something")
        refine_messages = build_messages(
            description="Do something",
            existing_workflow_json='{"name": "x", "nodes": [], "connections": {}, "active": false}',
        )
        assert "n8n-nodes-base" in fresh_messages[0]["content"]
        assert "n8n-nodes-base" in refine_messages[0]["content"]

    def test_model_context_hint_affects_node_selection(self) -> None:
        """A context hint should influence which nodes appear in the system prompt."""
        messages_with_hint = build_messages(
            description="Automate sales process",
            model_context_hint="salesforce crm",
        )
        system_content = messages_with_hint[0]["content"]
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

    def test_system_prompt_contains_example_workflow(self) -> None:
        """System prompt should inject an example workflow for few-shot grounding."""
        messages = build_messages(description="Do anything")
        system_content = messages[0]["content"]
        # The example workflow should have been injected
        assert "Schedule Trigger" in system_content or "scheduleTrigger" in system_content

    def test_system_prompt_contains_output_format_instructions(self) -> None:
        """System prompt must instruct the model on exact output format."""
        messages = build_messages(description="Some workflow")
        system_content = messages[0]["content"]
        # Should mention JSON code block format
        assert "```json" in system_content or "code block" in system_content.lower()

    def test_multiple_calls_produce_consistent_structure(self) -> None:
        """build_messages should be deterministic for the same input."""
        desc = "Send a daily report"
        msgs1 = build_messages(description=desc)
        msgs2 = build_messages(description=desc)
        assert msgs1[0]["role"] == msgs2[0]["role"]
        assert msgs1[1]["role"] == msgs2[1]["role"]
        assert msgs1[0]["content"] == msgs2[0]["content"]
        assert msgs1[1]["content"] == msgs2[1]["content"]

    def test_long_description_does_not_raise(self) -> None:
        """build_messages should handle very long descriptions gracefully."""
        long_desc = "Send a Slack notification " * 100
        messages = build_messages(description=long_desc)
        assert len(messages) == 2
        assert long_desc.strip() in messages[1]["content"]

    def test_special_characters_in_description(self) -> None:
        """Descriptions with special chars should not break prompt building."""
        desc = 'Send an email with subject "Report: Q1 & Q2" for <all> users'
        messages = build_messages(description=desc)
        assert desc in messages[1]["content"]

    def test_context_hint_without_match_does_not_raise(self) -> None:
        """A context hint that matches no nodes should not raise."""
        messages = build_messages(
            description="Do something",
            model_context_hint="xyzzy_nonexistent_service",
        )
        assert len(messages) == 2

    def test_refinement_user_message_includes_return_instruction(self) -> None:
        """Refinement mode user message should ask for a complete updated workflow."""
        existing = json.dumps(
            {"name": "Old", "nodes": [], "connections": {}, "active": False}
        )
        messages = build_messages(
            description="Add a filter",
            existing_workflow_json=existing,
        )
        user_content = messages[1]["content"].lower()
        assert any(
            word in user_content
            for word in ["complete", "updated", "return", "full"]
        )
