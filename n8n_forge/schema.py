"""Pydantic-based data models for n8n workflow JSON structure.

This module defines the schema for n8n workflow JSON files, providing
validation and serialization for all components of a workflow including
nodes, connections, and workflow-level metadata.

The schema is based on n8n's internal workflow format and can be used
to validate LLM-generated workflow JSON before importing into n8n.

Example usage::

    from n8n_forge.schema import WorkflowSchema, Node, Connection

    node = Node(
        id="1",
        name="Schedule Trigger",
        type="n8n-nodes-base.scheduleTrigger",
        typeVersion=1,
        position=[250, 300],
        parameters={}
    )

    workflow = WorkflowSchema(
        name="My Workflow",
        nodes=[node],
        connections={},
        active=False
    )

    json_str = workflow.model_dump_json(indent=2)
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator


class NodePosition(BaseModel):
    """Represents the x, y position of a node on the n8n canvas."""

    x: float = Field(..., description="Horizontal position on the canvas")
    y: float = Field(..., description="Vertical position on the canvas")

    def to_list(self) -> list[float]:
        """Convert position to a [x, y] list as used in n8n JSON."""
        return [self.x, self.y]


class NodeCredential(BaseModel):
    """Reference to a named credential set stored in n8n."""

    id: str | None = Field(default=None, description="Credential ID in n8n")
    name: str = Field(..., description="Human-readable credential name")


class Node(BaseModel):
    """Represents a single n8n workflow node.

    Each node corresponds to an action or trigger in the workflow.
    Nodes are connected together via the ``connections`` mapping in
    the parent :class:`WorkflowSchema`.

    Attributes:
        id: Unique identifier for this node within the workflow.
        name: Human-readable display name shown on the canvas.
        type: n8n node type identifier (e.g. ``n8n-nodes-base.httpRequest``).
        typeVersion: Version of the node type to use.
        position: [x, y] coordinates on the n8n canvas.
        parameters: Node-specific configuration parameters.
        credentials: Optional mapping of credential type to credential reference.
        disabled: Whether the node is disabled in the workflow.
        notes: Optional notes displayed on the node in the canvas.
        continueOnFail: Whether the workflow continues even if this node fails.
        retryOnFail: Whether the node automatically retries on failure.
        maxTries: Maximum number of retry attempts.
        waitBetweenTries: Milliseconds to wait between retry attempts.
        alwaysOutputData: Whether to always output data even on error.
        executeOnce: Whether to execute only once regardless of input items.
    """

    id: str = Field(..., description="Unique node identifier")
    name: str = Field(..., description="Display name of the node")
    type: str = Field(..., description="n8n node type identifier")
    typeVersion: int = Field(default=1, description="Node type version", ge=1)
    position: list[float] = Field(
        default_factory=lambda: [250.0, 300.0],
        description="[x, y] canvas position",
        min_length=2,
        max_length=2,
    )
    parameters: dict[str, Any] = Field(
        default_factory=dict, description="Node-specific parameters"
    )
    credentials: dict[str, NodeCredential] | None = Field(
        default=None, description="Credential references keyed by credential type"
    )
    disabled: bool = Field(default=False, description="Whether the node is disabled")
    notes: str | None = Field(default=None, description="Optional canvas notes")
    continueOnFail: bool = Field(
        default=False, description="Continue workflow even if this node fails"
    )
    retryOnFail: bool = Field(default=False, description="Retry node on failure")
    maxTries: int = Field(default=3, description="Maximum retry attempts", ge=1)
    waitBetweenTries: int = Field(
        default=1000, description="Milliseconds between retries", ge=0
    )
    alwaysOutputData: bool = Field(
        default=False, description="Always output data even on error"
    )
    executeOnce: bool = Field(
        default=False, description="Execute only once regardless of input items"
    )

    @field_validator("type")
    @classmethod
    def validate_node_type(cls, v: str) -> str:
        """Ensure node type follows n8n naming conventions."""
        if not v or not v.strip():
            raise ValueError("Node type must not be empty")
        return v.strip()

    @field_validator("position")
    @classmethod
    def validate_position(cls, v: list[float]) -> list[float]:
        """Ensure position has exactly two numeric values."""
        if len(v) != 2:
            raise ValueError("Position must be a list of exactly two numbers [x, y]")
        return v

    @field_validator("id")
    @classmethod
    def validate_id(cls, v: str) -> str:
        """Ensure node ID is not empty."""
        if not v or not v.strip():
            raise ValueError("Node id must not be empty")
        return v.strip()

    def is_trigger(self) -> bool:
        """Return True if this node is a trigger (starts a workflow).

        Returns:
            True if the node type string contains 'trigger' (case-insensitive).
        """
        return "trigger" in self.type.lower()

    def model_dump_n8n(self) -> dict[str, Any]:
        """Serialize node to n8n-compatible dict format.

        Returns:
            Dictionary representation matching n8n's expected JSON structure,
            with ``None`` values excluded.
        """
        data = self.model_dump(exclude_none=True)
        return data


class ConnectionItem(BaseModel):
    """A single connection reference pointing to a target node input.

    Attributes:
        node: The name of the destination node.
        type: Connection type, typically ``"main"``.
        index: The input index on the destination node (usually 0).
    """

    node: str = Field(..., description="Target node name")
    type: str = Field(default="main", description="Connection type")
    index: int = Field(default=0, description="Target node input index", ge=0)


class WorkflowSettings(BaseModel):
    """Workflow-level execution settings.

    Attributes:
        executionOrder: Execution order strategy (``"v1"`` is the default).
        saveManualExecutions: Whether to save manual execution logs.
        callerPolicy: Who can call this workflow as a sub-workflow.
        errorWorkflow: ID of a workflow to trigger on error.
        timezone: Timezone for schedule-based triggers.
        saveDataSuccessExecution: When to save successful execution data.
        saveDataErrorExecution: When to save failed execution data.
        saveExecutionProgress: Whether to save execution progress.
    """

    executionOrder: str = Field(default="v1", description="Execution order strategy")
    saveManualExecutions: bool = Field(
        default=True, description="Save manual execution logs"
    )
    callerPolicy: str = Field(
        default="workflowsFromSameOwner",
        description="Sub-workflow caller policy",
    )
    errorWorkflow: str | None = Field(
        default=None, description="Error workflow ID"
    )
    timezone: str = Field(default="UTC", description="Timezone for schedule triggers")
    saveDataSuccessExecution: str = Field(
        default="all", description="When to save successful execution data"
    )
    saveDataErrorExecution: str = Field(
        default="all", description="When to save failed execution data"
    )
    saveExecutionProgress: bool = Field(
        default=False, description="Save execution progress"
    )


class WorkflowSchema(BaseModel):
    """Top-level n8n workflow schema.

    This model represents a complete n8n workflow that can be imported
    directly into n8n via the UI or API.

    Attributes:
        name: Human-readable workflow name.
        nodes: List of all nodes in the workflow.
        connections: Adjacency mapping describing node connections.
            Format: ``{"SourceNodeName": {"main": [[ConnectionItem, ...]]}}`
        active: Whether the workflow is active (running on schedule/trigger).
        settings: Workflow-level execution settings.
        id: Optional n8n-assigned workflow ID (set after import).
        meta: Optional workflow metadata dict.
        tags: List of tag labels for the workflow.
        pinData: Optional pinned test data keyed by node name.
        staticData: Optional static data persisted between executions.
        versionId: Optional version identifier assigned by n8n.

    Example::

        workflow = WorkflowSchema(
            name="HN to Slack",
            nodes=[...],
            connections={"Schedule Trigger": {"main": [[{"node": "HTTP Request", "type": "main", "index": 0}]]}},
            active=False,
        )
    """

    name: str = Field(..., description="Workflow display name")
    nodes: list[Node] = Field(
        default_factory=list, description="All nodes in the workflow"
    )
    connections: dict[str, dict[str, list[list[ConnectionItem]]]] = Field(
        default_factory=dict,
        description="Node connection adjacency map",
    )
    active: bool = Field(default=False, description="Whether the workflow is active")
    settings: WorkflowSettings = Field(
        default_factory=WorkflowSettings,
        description="Workflow execution settings",
    )
    id: str | None = Field(default=None, description="n8n-assigned workflow ID")
    meta: dict[str, Any] | None = Field(
        default=None, description="Workflow metadata"
    )
    tags: list[str] = Field(default_factory=list, description="Workflow tag labels")
    pinData: dict[str, Any] | None = Field(
        default=None, description="Pinned test data by node name"
    )
    staticData: dict[str, Any] | None = Field(
        default=None, description="Static data persisted between executions"
    )
    versionId: str | None = Field(
        default=None, description="Version identifier assigned by n8n"
    )

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Ensure workflow name is non-empty."""
        if not v or not v.strip():
            raise ValueError("Workflow name must not be empty")
        return v.strip()

    @field_validator("nodes")
    @classmethod
    def validate_nodes(cls, v: list[Node]) -> list[Node]:
        """Ensure node IDs and names are unique within the workflow."""
        seen_ids: set[str] = set()
        seen_names: set[str] = set()
        for node in v:
            if node.id in seen_ids:
                raise ValueError(f"Duplicate node id found: '{node.id}'")
            if node.name in seen_names:
                raise ValueError(f"Duplicate node name found: '{node.name}'")
            seen_ids.add(node.id)
            seen_names.add(node.name)
        return v

    @model_validator(mode="after")
    def validate_connections_reference_existing_nodes(self) -> "WorkflowSchema":
        """Ensure all connection source and target node names exist in nodes list."""
        node_names = {node.name for node in self.nodes}
        for source_name, output_map in self.connections.items():
            if source_name not in node_names:
                raise ValueError(
                    f"Connection source node '{source_name}' not found in nodes list"
                )
            for _output_type, output_groups in output_map.items():
                for group in output_groups:
                    for conn in group:
                        if conn.node not in node_names:
                            raise ValueError(
                                f"Connection target node '{conn.node}' not found "
                                f"in nodes list"
                            )
        return self

    def get_trigger_nodes(self) -> list[Node]:
        """Return all trigger nodes in the workflow.

        Returns:
            List of :class:`Node` instances whose type contains 'trigger'.
        """
        return [node for node in self.nodes if node.is_trigger()]

    def get_connection_count(self) -> int:
        """Return the total number of individual connections in the workflow.

        Returns:
            Sum of all :class:`ConnectionItem` instances across all outputs.
        """
        count = 0
        for output_map in self.connections.values():
            for output_groups in output_map.values():
                for group in output_groups:
                    count += len(group)
        return count

    def model_dump_n8n(self) -> dict[str, Any]:
        """Serialize the workflow to a dict compatible with n8n's import format.

        Returns:
            A dictionary matching n8n's expected workflow JSON structure,
            with ``None`` values excluded and connection items serialized.
        """
        data = self.model_dump(exclude_none=True)
        # Serialize connections with ConnectionItem dicts
        serialized_connections: dict[str, dict[str, list[list[dict[str, Any]]]]] = {}
        for source, output_map in self.connections.items():
            serialized_connections[source] = {}
            for out_type, groups in output_map.items():
                serialized_connections[source][out_type] = [
                    [item.model_dump() for item in group] for group in groups
                ]
        data["connections"] = serialized_connections
        return data


class WorkflowValidationError(Exception):
    """Raised when a workflow JSON fails schema validation.

    Attributes:
        message: Human-readable description of the validation failure.
        errors: List of specific Pydantic validation error details.
    """

    def __init__(self, message: str, errors: list[dict[str, Any]] | None = None) -> None:
        """Initialize with a message and optional list of error details.

        Args:
            message: Description of what validation failed.
            errors: Optional list of Pydantic error detail dicts.
        """
        super().__init__(message)
        self.message = message
        self.errors = errors or []

    def __str__(self) -> str:
        """Return formatted error string."""
        if self.errors:
            error_details = "; ".join(
                f"{'.'.join(str(loc) for loc in e.get('loc', []))}: {e.get('msg', '')}"
                for e in self.errors
            )
            return f"{self.message} — {error_details}"
        return self.message


def validate_workflow_dict(data: dict[str, Any]) -> WorkflowSchema:
    """Validate a raw dictionary against the WorkflowSchema.

    Converts n8n's raw JSON connection format (dicts) into proper
    :class:`ConnectionItem` objects before validation.

    Args:
        data: Raw dictionary parsed from n8n workflow JSON.

    Returns:
        A validated :class:`WorkflowSchema` instance.

    Raises:
        WorkflowValidationError: If the dictionary does not conform to the
            expected n8n workflow structure.
    """
    from pydantic import ValidationError

    # Normalize connections: convert raw dicts to ConnectionItem-compatible dicts
    if "connections" in data and isinstance(data["connections"], dict):
        normalized: dict[str, Any] = {}
        for source, output_map in data["connections"].items():
            if not isinstance(output_map, dict):
                continue
            normalized[source] = {}
            for out_type, groups in output_map.items():
                if not isinstance(groups, list):
                    normalized[source][out_type] = []
                    continue
                normalized_groups = []
                for group in groups:
                    if not isinstance(group, list):
                        normalized_groups.append([])
                        continue
                    normalized_group = []
                    for item in group:
                        if isinstance(item, dict):
                            normalized_group.append(item)
                        elif isinstance(item, ConnectionItem):
                            normalized_group.append(item.model_dump())
                    normalized_groups.append(normalized_group)
                normalized[source][out_type] = normalized_groups
        data = {**data, "connections": normalized}

    try:
        return WorkflowSchema.model_validate(data)
    except ValidationError as exc:
        errors = exc.errors()
        raise WorkflowValidationError(
            f"Workflow JSON failed schema validation ({len(errors)} error(s))",
            errors=errors,
        ) from exc
