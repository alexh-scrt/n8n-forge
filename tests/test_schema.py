"""Unit tests for the n8n_forge.schema module.

Covers Pydantic model validation, helper methods, serialization,
and the WorkflowValidationError exception class.
"""

from __future__ import annotations

import pytest

from n8n_forge.schema import (
    ConnectionItem,
    Node,
    NodeCredential,
    WorkflowSchema,
    WorkflowSettings,
    WorkflowValidationError,
    validate_workflow_dict,
)


# ---------------------------------------------------------------------------
# Node tests
# ---------------------------------------------------------------------------


class TestNode:
    """Tests for the Node model."""

    def _make_node(self, **kwargs) -> Node:
        defaults = dict(id="node-1", name="HTTP Request", type="n8n-nodes-base.httpRequest")
        defaults.update(kwargs)
        return Node(**defaults)

    def test_basic_creation(self) -> None:
        node = self._make_node()
        assert node.id == "node-1"
        assert node.name == "HTTP Request"
        assert node.type == "n8n-nodes-base.httpRequest"
        assert node.typeVersion == 1
        assert node.position == [250.0, 300.0]
        assert node.disabled is False

    def test_is_trigger_true(self) -> None:
        node = self._make_node(type="n8n-nodes-base.scheduleTrigger")
        assert node.is_trigger() is True

    def test_is_trigger_false(self) -> None:
        node = self._make_node(type="n8n-nodes-base.httpRequest")
        assert node.is_trigger() is False

    def test_is_trigger_case_insensitive(self) -> None:
        node = self._make_node(type="n8n-nodes-base.SCHEDULETRIGGER")
        assert node.is_trigger() is True

    def test_empty_id_raises(self) -> None:
        with pytest.raises(Exception):
            self._make_node(id="")

    def test_whitespace_id_raises(self) -> None:
        with pytest.raises(Exception):
            self._make_node(id="   ")

    def test_empty_type_raises(self) -> None:
        with pytest.raises(Exception):
            self._make_node(type="")

    def test_position_must_be_two_elements(self) -> None:
        with pytest.raises(Exception):
            self._make_node(position=[100.0])

    def test_position_with_three_elements_raises(self) -> None:
        with pytest.raises(Exception):
            self._make_node(position=[100.0, 200.0, 300.0])

    def test_model_dump_n8n_excludes_none(self) -> None:
        node = self._make_node(notes=None, credentials=None)
        data = node.model_dump_n8n()
        assert "notes" not in data
        assert "credentials" not in data

    def test_node_with_credentials(self) -> None:
        cred = NodeCredential(id="cred-123", name="My Google Account")
        node = self._make_node(credentials={"googleSheetsOAuth2Api": cred})
        assert node.credentials is not None
        assert node.credentials["googleSheetsOAuth2Api"].name == "My Google Account"

    def test_default_parameters_empty(self) -> None:
        node = self._make_node()
        assert node.parameters == {}

    def test_custom_parameters(self) -> None:
        node = self._make_node(parameters={"url": "https://example.com", "method": "GET"})
        assert node.parameters["url"] == "https://example.com"


# ---------------------------------------------------------------------------
# ConnectionItem tests
# ---------------------------------------------------------------------------


class TestConnectionItem:
    """Tests for the ConnectionItem model."""

    def test_basic_creation(self) -> None:
        conn = ConnectionItem(node="Slack", type="main", index=0)
        assert conn.node == "Slack"
        assert conn.type == "main"
        assert conn.index == 0

    def test_defaults(self) -> None:
        conn = ConnectionItem(node="Slack")
        assert conn.type == "main"
        assert conn.index == 0

    def test_negative_index_raises(self) -> None:
        with pytest.raises(Exception):
            ConnectionItem(node="Slack", index=-1)


# ---------------------------------------------------------------------------
# WorkflowSettings tests
# ---------------------------------------------------------------------------


class TestWorkflowSettings:
    """Tests for the WorkflowSettings model."""

    def test_defaults(self) -> None:
        settings = WorkflowSettings()
        assert settings.executionOrder == "v1"
        assert settings.timezone == "UTC"
        assert settings.saveManualExecutions is True
        assert settings.saveDataSuccessExecution == "all"

    def test_custom_timezone(self) -> None:
        settings = WorkflowSettings(timezone="America/New_York")
        assert settings.timezone == "America/New_York"


# ---------------------------------------------------------------------------
# WorkflowSchema tests
# ---------------------------------------------------------------------------


def _make_trigger_node(node_id: str = "1", name: str = "Schedule Trigger") -> Node:
    return Node(
        id=node_id,
        name=name,
        type="n8n-nodes-base.scheduleTrigger",
        position=[250.0, 300.0],
    )


def _make_action_node(node_id: str = "2", name: str = "HTTP Request") -> Node:
    return Node(
        id=node_id,
        name=name,
        type="n8n-nodes-base.httpRequest",
        position=[500.0, 300.0],
    )


class TestWorkflowSchema:
    """Tests for the WorkflowSchema model."""

    def test_minimal_workflow(self) -> None:
        wf = WorkflowSchema(name="Test Workflow", nodes=[], connections={})
        assert wf.name == "Test Workflow"
        assert wf.nodes == []
        assert wf.connections == {}
        assert wf.active is False

    def test_workflow_with_nodes(self) -> None:
        trigger = _make_trigger_node()
        action = _make_action_node()
        conn = ConnectionItem(node="HTTP Request")
        wf = WorkflowSchema(
            name="My Workflow",
            nodes=[trigger, action],
            connections={"Schedule Trigger": {"main": [[conn]]}},
        )
        assert len(wf.nodes) == 2
        assert wf.get_connection_count() == 1

    def test_get_trigger_nodes(self) -> None:
        trigger = _make_trigger_node()
        action = _make_action_node()
        wf = WorkflowSchema(name="WF", nodes=[trigger, action], connections={})
        triggers = wf.get_trigger_nodes()
        assert len(triggers) == 1
        assert triggers[0].name == "Schedule Trigger"

    def test_get_connection_count_empty(self) -> None:
        wf = WorkflowSchema(name="WF", nodes=[], connections={})
        assert wf.get_connection_count() == 0

    def test_get_connection_count_multiple(self) -> None:
        t = _make_trigger_node()
        a = _make_action_node("2", "HTTP Request")
        b = _make_action_node("3", "Code")
        conn1 = ConnectionItem(node="HTTP Request")
        conn2 = ConnectionItem(node="Code")
        wf = WorkflowSchema(
            name="WF",
            nodes=[t, a, b],
            connections={
                "Schedule Trigger": {"main": [[conn1, conn2]]}
            },
        )
        assert wf.get_connection_count() == 2

    def test_empty_name_raises(self) -> None:
        with pytest.raises(Exception):
            WorkflowSchema(name="", nodes=[], connections={})

    def test_whitespace_name_raises(self) -> None:
        with pytest.raises(Exception):
            WorkflowSchema(name="   ", nodes=[], connections={})

    def test_duplicate_node_ids_raises(self) -> None:
        t1 = _make_trigger_node(node_id="1", name="Trigger 1")
        t2 = _make_trigger_node(node_id="1", name="Trigger 2")  # duplicate id
        with pytest.raises(Exception, match="Duplicate node id"):
            WorkflowSchema(name="WF", nodes=[t1, t2], connections={})

    def test_duplicate_node_names_raises(self) -> None:
        t1 = _make_trigger_node(node_id="1", name="Same Name")
        t2 = _make_trigger_node(node_id="2", name="Same Name")  # duplicate name
        with pytest.raises(Exception, match="Duplicate node name"):
            WorkflowSchema(name="WF", nodes=[t1, t2], connections={})

    def test_connection_source_not_in_nodes_raises(self) -> None:
        action = _make_action_node()
        conn = ConnectionItem(node="HTTP Request")
        with pytest.raises(Exception, match="source node"):
            WorkflowSchema(
                name="WF",
                nodes=[action],
                connections={"NonExistentTrigger": {"main": [[conn]]}},
            )

    def test_connection_target_not_in_nodes_raises(self) -> None:
        trigger = _make_trigger_node()
        conn = ConnectionItem(node="NonExistentAction")
        with pytest.raises(Exception, match="target node"):
            WorkflowSchema(
                name="WF",
                nodes=[trigger],
                connections={"Schedule Trigger": {"main": [[conn]]}},
            )

    def test_model_dump_n8n_serializes_connections(self) -> None:
        trigger = _make_trigger_node()
        action = _make_action_node()
        conn = ConnectionItem(node="HTTP Request")
        wf = WorkflowSchema(
            name="WF",
            nodes=[trigger, action],
            connections={"Schedule Trigger": {"main": [[conn]]}},
        )
        data = wf.model_dump_n8n()
        assert "connections" in data
        raw_conn = data["connections"]["Schedule Trigger"]["main"][0][0]
        assert raw_conn["node"] == "HTTP Request"
        assert raw_conn["type"] == "main"
        assert raw_conn["index"] == 0

    def test_tags_default_empty(self) -> None:
        wf = WorkflowSchema(name="WF", nodes=[], connections={})
        assert wf.tags == []

    def test_workflow_with_tags(self) -> None:
        wf = WorkflowSchema(name="WF", nodes=[], connections={}, tags=["finance", "weekly"])
        assert "finance" in wf.tags


# ---------------------------------------------------------------------------
# WorkflowValidationError tests
# ---------------------------------------------------------------------------


class TestWorkflowValidationError:
    """Tests for the WorkflowValidationError exception."""

    def test_basic_message(self) -> None:
        err = WorkflowValidationError("Something went wrong")
        assert str(err) == "Something went wrong"
        assert err.errors == []

    def test_with_errors_list(self) -> None:
        errors = [{"loc": ("nodes",), "msg": "field required"}]
        err = WorkflowValidationError("Validation failed", errors=errors)
        assert "nodes" in str(err)
        assert "field required" in str(err)

    def test_is_exception(self) -> None:
        err = WorkflowValidationError("Oops")
        assert isinstance(err, Exception)


# ---------------------------------------------------------------------------
# validate_workflow_dict tests
# ---------------------------------------------------------------------------


class TestValidateWorkflowDict:
    """Tests for the validate_workflow_dict helper function."""

    def _minimal_dict(self) -> dict:
        return {
            "name": "Test",
            "nodes": [
                {
                    "id": "1",
                    "name": "Schedule Trigger",
                    "type": "n8n-nodes-base.scheduleTrigger",
                    "typeVersion": 1,
                    "position": [250, 300],
                    "parameters": {},
                },
                {
                    "id": "2",
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

    def test_valid_dict_returns_schema(self) -> None:
        data = self._minimal_dict()
        wf = validate_workflow_dict(data)
        assert isinstance(wf, WorkflowSchema)
        assert wf.name == "Test"
        assert len(wf.nodes) == 2

    def test_missing_name_raises_validation_error(self) -> None:
        data = self._minimal_dict()
        del data["name"]
        with pytest.raises(WorkflowValidationError):
            validate_workflow_dict(data)

    def test_invalid_node_position_raises(self) -> None:
        data = self._minimal_dict()
        data["nodes"][0]["position"] = [100]  # only one element
        with pytest.raises(WorkflowValidationError):
            validate_workflow_dict(data)

    def test_empty_connections_valid(self) -> None:
        data = self._minimal_dict()
        data["connections"] = {}
        wf = validate_workflow_dict(data)
        assert wf.get_connection_count() == 0

    def test_connection_count_correct(self) -> None:
        data = self._minimal_dict()
        wf = validate_workflow_dict(data)
        assert wf.get_connection_count() == 1
