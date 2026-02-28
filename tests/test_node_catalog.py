"""Unit tests for the n8n_forge.node_catalog module.

Covers catalog entry attributes, search functionality, category filtering,
trigger detection, and prompt text generation.
"""

from __future__ import annotations

import pytest

from n8n_forge.node_catalog import (
    NODE_CATALOG,
    NodeCatalogEntry,
    NodeParameter,
    catalog_to_prompt_text,
    get_all_categories,
    get_node_by_type,
    get_nodes_by_category,
    get_trigger_nodes,
    search_nodes,
)


class TestNodeCatalogEntry:
    """Tests for the NodeCatalogEntry dataclass."""

    def test_to_prompt_text_contains_type(self) -> None:
        entry = get_node_by_type("n8n-nodes-base.httpRequest")
        assert entry is not None
        text = entry.to_prompt_text()
        assert "n8n-nodes-base.httpRequest" in text

    def test_to_prompt_text_contains_description(self) -> None:
        entry = get_node_by_type("n8n-nodes-base.slack")
        assert entry is not None
        text = entry.to_prompt_text()
        assert "Slack" in text

    def test_to_prompt_text_contains_use_cases(self) -> None:
        entry = get_node_by_type("n8n-nodes-base.scheduleTrigger")
        assert entry is not None
        text = entry.to_prompt_text()
        assert "Use cases" in text

    def test_to_prompt_text_contains_credentials(self) -> None:
        entry = get_node_by_type("n8n-nodes-base.googleSheets")
        assert entry is not None
        text = entry.to_prompt_text()
        assert "Requires credentials" in text

    def test_to_prompt_text_contains_key_parameters(self) -> None:
        entry = get_node_by_type("n8n-nodes-base.httpRequest")
        assert entry is not None
        text = entry.to_prompt_text()
        assert "Key parameters" in text


class TestNodeParameter:
    """Tests for the NodeParameter dataclass."""

    def test_basic_creation(self) -> None:
        param = NodeParameter(name="url", description="Target URL", required=True)
        assert param.name == "url"
        assert param.required is True
        assert param.default is None

    def test_defaults(self) -> None:
        param = NodeParameter(name="method", description="HTTP method")
        assert param.required is False
        assert param.param_type == "string"


class TestNodeCatalogLookups:
    """Tests for catalog lookup functions."""

    def test_get_node_by_type_returns_entry(self) -> None:
        entry = get_node_by_type("n8n-nodes-base.httpRequest")
        assert entry is not None
        assert entry.type_name == "n8n-nodes-base.httpRequest"
        assert entry.display_name == "HTTP Request"

    def test_get_node_by_type_unknown_returns_none(self) -> None:
        entry = get_node_by_type("n8n-nodes-base.doesNotExist")
        assert entry is None

    def test_get_trigger_nodes_all_are_triggers(self) -> None:
        triggers = get_trigger_nodes()
        assert len(triggers) > 0
        for t in triggers:
            assert t.is_trigger is True

    def test_schedule_trigger_in_trigger_nodes(self) -> None:
        triggers = get_trigger_nodes()
        types = [t.type_name for t in triggers]
        assert "n8n-nodes-base.scheduleTrigger" in types

    def test_webhook_in_trigger_nodes(self) -> None:
        triggers = get_trigger_nodes()
        types = [t.type_name for t in triggers]
        assert "n8n-nodes-base.webhook" in types

    def test_http_request_not_in_trigger_nodes(self) -> None:
        triggers = get_trigger_nodes()
        types = [t.type_name for t in triggers]
        assert "n8n-nodes-base.httpRequest" not in types

    def test_get_all_categories_returns_list(self) -> None:
        categories = get_all_categories()
        assert isinstance(categories, list)
        assert len(categories) > 0

    def test_get_all_categories_contains_trigger(self) -> None:
        categories = get_all_categories()
        assert "Trigger" in categories

    def test_get_all_categories_sorted(self) -> None:
        categories = get_all_categories()
        assert categories == sorted(categories)

    def test_get_nodes_by_category_communication(self) -> None:
        nodes = get_nodes_by_category("Communication")
        assert len(nodes) > 0
        for n in nodes:
            assert n.category.lower() == "communication"

    def test_get_nodes_by_category_case_insensitive(self) -> None:
        upper = get_nodes_by_category("TRIGGER")
        lower = get_nodes_by_category("trigger")
        assert len(upper) == len(lower)

    def test_get_nodes_by_category_unknown_returns_empty(self) -> None:
        result = get_nodes_by_category("NonExistentCategory")
        assert result == []


class TestSearchNodes:
    """Tests for the search_nodes function."""

    def test_search_slack_returns_slack(self) -> None:
        results = search_nodes("slack")
        types = [r.type_name for r in results]
        assert "n8n-nodes-base.slack" in types

    def test_search_empty_returns_all(self) -> None:
        all_nodes = search_nodes("")
        assert len(all_nodes) == len(NODE_CATALOG)

    def test_search_schedule_returns_trigger(self) -> None:
        results = search_nodes("schedule")
        assert len(results) > 0
        types = [r.type_name for r in results]
        assert "n8n-nodes-base.scheduleTrigger" in types

    def test_search_google_returns_google_nodes(self) -> None:
        results = search_nodes("google")
        assert len(results) > 0
        for r in results:
            text = (
                r.display_name + r.type_name + r.description + r.category
                + " ".join(r.use_cases)
            ).lower()
            assert "google" in text

    def test_search_nonexistent_returns_empty(self) -> None:
        results = search_nodes("xyzzy_nonexistent_123")
        assert results == []

    def test_search_name_matches_come_before_desc_matches(self) -> None:
        # "HTTP" appears in the display name of HTTP Request and in descriptions of others
        results = search_nodes("http")
        assert len(results) > 0
        # First result should be a name match
        first = results[0]
        assert "http" in first.display_name.lower() or "http" in first.type_name.lower()


class TestCatalogToPromptText:
    """Tests for the catalog_to_prompt_text helper."""

    def test_returns_string(self) -> None:
        text = catalog_to_prompt_text()
        assert isinstance(text, str)
        assert len(text) > 0

    def test_max_nodes_limits_output(self) -> None:
        text_5 = catalog_to_prompt_text(max_nodes=5)
        text_all = catalog_to_prompt_text()
        # 5-node output must be shorter than full catalog output
        assert len(text_5) < len(text_all)

    def test_specific_entries_rendered(self) -> None:
        slack = get_node_by_type("n8n-nodes-base.slack")
        assert slack is not None
        text = catalog_to_prompt_text(entries=[slack])
        assert "Slack" in text
        assert "n8n-nodes-base.slack" in text

    def test_empty_entries_returns_empty_string(self) -> None:
        text = catalog_to_prompt_text(entries=[])
        assert text == ""

    def test_full_catalog_contains_known_nodes(self) -> None:
        text = catalog_to_prompt_text(max_nodes=100)
        assert "Schedule Trigger" in text
        assert "HTTP Request" in text


class TestNodeCatalogCompleteness:
    """Tests ensuring the catalog contains expected nodes."""

    EXPECTED_TYPES = [
        "n8n-nodes-base.scheduleTrigger",
        "n8n-nodes-base.webhook",
        "n8n-nodes-base.httpRequest",
        "n8n-nodes-base.code",
        "n8n-nodes-base.set",
        "n8n-nodes-base.if",
        "n8n-nodes-base.slack",
        "n8n-nodes-base.sendEmail",
        "n8n-nodes-base.googleSheets",
        "n8n-nodes-base.postgres",
    ]

    @pytest.mark.parametrize("type_name", EXPECTED_TYPES)
    def test_expected_node_exists(self, type_name: str) -> None:
        entry = get_node_by_type(type_name)
        assert entry is not None, f"Expected node '{type_name}' not found in catalog"

    def test_all_entries_have_non_empty_description(self) -> None:
        for type_name, entry in NODE_CATALOG.items():
            assert entry.description.strip(), (
                f"Node '{type_name}' has an empty description"
            )

    def test_all_entries_have_non_empty_display_name(self) -> None:
        for type_name, entry in NODE_CATALOG.items():
            assert entry.display_name.strip(), (
                f"Node '{type_name}' has an empty display_name"
            )

    def test_all_trigger_entries_have_is_trigger_true(self) -> None:
        for type_name, entry in NODE_CATALOG.items():
            if "trigger" in type_name.lower():
                assert entry.is_trigger, (
                    f"Node '{type_name}' looks like a trigger but is_trigger=False"
                )

    def test_catalog_keys_match_type_names(self) -> None:
        for key, entry in NODE_CATALOG.items():
            assert key == entry.type_name, (
                f"Catalog key '{key}' does not match entry type_name '{entry.type_name}'"
            )
