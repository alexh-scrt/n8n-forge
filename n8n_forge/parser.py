"""JSON extraction, cleaning, and schema validation for n8n workflow responses.

This module is responsible for taking the raw string output from the LLM
and producing a validated :class:`~n8n_forge.schema.WorkflowSchema` object
(or a clean JSON string) that is safe to import into n8n.

The pipeline applied to each raw response is:

1. **Extraction** — Find and extract the JSON content from the LLM response,
   handling fenced code blocks (\`\`\`json ... \`\`\`) as well as bare JSON.
2. **Cleaning** — Strip common LLM artefacts such as trailing commas,
   JavaScript-style comments, and BOM characters.
3. **Parsing** — Parse the cleaned string into a Python ``dict`` via
   :mod:`json`.
4. **Validation** — Validate the parsed dict against
   :class:`~n8n_forge.schema.WorkflowSchema` using
   :func:`~n8n_forge.schema.validate_workflow_dict`.

Example usage::

    from n8n_forge.parser import parse_workflow_response

    raw_llm_output = '''
    Here is your workflow:
    ```json
    {"name": "My Workflow", "nodes": [...], "connections": {}, "active": false}
    ```
    '''
    workflow = parse_workflow_response(raw_llm_output)
    print(workflow.name)  # "My Workflow"
    print(workflow.model_dump_json(indent=2))  # Clean JSON string
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from n8n_forge.schema import WorkflowSchema, WorkflowValidationError, validate_workflow_dict

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class ParserError(Exception):
    """Raised when the LLM response cannot be parsed into a valid workflow.

    Attributes:
        message: Human-readable description of the parse failure.
        raw_response: The original LLM response string that failed to parse.
    """

    def __init__(self, message: str, raw_response: str = "") -> None:
        """Initialise with a descriptive message and optional raw response.

        Args:
            message: What went wrong during parsing.
            raw_response: The raw LLM response string (used for diagnostics).
        """
        super().__init__(message)
        self.message = message
        self.raw_response = raw_response

    def __str__(self) -> str:
        """Return the error message."""
        return self.message


class JSONExtractionError(ParserError):
    """Raised when no JSON object can be located in the LLM response."""


class JSONDecodeError(ParserError):
    """Raised when extracted text is found but cannot be decoded as JSON."""


# ---------------------------------------------------------------------------
# Step 1: JSON Extraction
# ---------------------------------------------------------------------------

# Regex for fenced code blocks: ```json ... ``` or ``` ... ```
_FENCED_JSON_RE = re.compile(
    r"```(?:json)?\s*\n?(.*?)\n?\s*```",
    re.DOTALL | re.IGNORECASE,
)

# Regex to find a JSON object starting with `{` and ending with `}` (greedy, outermost)
_BARE_JSON_OBJECT_RE = re.compile(
    r"(\{.*\})",
    re.DOTALL,
)


def extract_json_string(raw_response: str) -> str:
    """Extract the JSON string from the raw LLM response.

    Attempts extraction strategies in the following order:

    1. Fenced code block: \`\`\`json ... \`\`\` or \`\`\` ... \`\`\`
    2. Bare JSON object (first ``{`` to last ``}`` in the response).

    Args:
        raw_response: The raw string returned by the LLM.

    Returns:
        The extracted JSON substring (not yet parsed or validated).

    Raises:
        JSONExtractionError: If no JSON object can be located in the response.
    """
    if not raw_response or not raw_response.strip():
        raise JSONExtractionError(
            "LLM response is empty — cannot extract JSON.",
            raw_response=raw_response,
        )

    # Strategy 1: Look for a fenced code block
    fenced_matches = _FENCED_JSON_RE.findall(raw_response)
    for candidate in fenced_matches:
        candidate = candidate.strip()
        if candidate.startswith("{"):
            logger.debug("Extracted JSON from fenced code block (%d chars).", len(candidate))
            return candidate

    # Strategy 2: Find the outermost JSON object in the response
    # Use a bracket-matching approach for robustness with nested objects
    extracted = _extract_outermost_object(raw_response)
    if extracted:
        logger.debug("Extracted JSON via bracket matching (%d chars).", len(extracted))
        return extracted

    # Strategy 3: Fall back to regex match (handles edge cases)
    bare_match = _BARE_JSON_OBJECT_RE.search(raw_response)
    if bare_match:
        candidate = bare_match.group(1).strip()
        logger.debug("Extracted JSON via bare regex (%d chars).", len(candidate))
        return candidate

    raise JSONExtractionError(
        "No JSON object found in the LLM response. "
        "The model may not have returned valid JSON. "
        "Try again or use a more capable model.",
        raw_response=raw_response,
    )


def _extract_outermost_object(text: str) -> str | None:
    """Extract the outermost JSON object using bracket matching.

    Finds the first ``{`` character, then scans forward tracking nesting
    depth until the matching ``}`` is found. Handles quoted strings
    containing braces correctly.

    Args:
        text: String to search for a JSON object.

    Returns:
        The substring from the first ``{`` to its matching ``}``,
        or ``None`` if no balanced object is found.
    """
    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escape_next = False

    for i, ch in enumerate(text[start:], start=start):
        if escape_next:
            escape_next = False
            continue
        if ch == "\\" and in_string:
            escape_next = True
            continue
        if ch == '"' and not escape_next:
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]

    return None


# ---------------------------------------------------------------------------
# Step 2: JSON Cleaning
# ---------------------------------------------------------------------------

# Match trailing commas before a closing } or ] — common LLM error
_TRAILING_COMMA_RE = re.compile(
    r",\s*([}\]])",
    re.MULTILINE,
)

# Match single-line JavaScript comments (// ...) outside of strings
_SINGLE_LINE_COMMENT_RE = re.compile(
    r"//[^\n]*",
)

# Match multi-line JavaScript comments (/* ... */)
_MULTI_LINE_COMMENT_RE = re.compile(
    r"/\*.*?\*/",
    re.DOTALL,
)


def clean_json_string(raw_json: str) -> str:
    """Clean common LLM JSON artefacts from a JSON string.

    Applies the following transformations in order:

    1. Strip leading/trailing whitespace and BOM characters.
    2. Remove JavaScript-style single-line comments (``// ...``).
    3. Remove JavaScript-style multi-line comments (``/* ... */``).
    4. Remove trailing commas before ``}`` or ``]``.

    .. warning::
        Comment removal is done via regex and does not correctly handle
        comment-like sequences inside JSON string values. In practice this
        is rarely an issue for LLM-generated n8n workflow JSON.

    Args:
        raw_json: The extracted JSON string, potentially containing artefacts.

    Returns:
        A cleaned JSON string that is more likely to parse successfully.
    """
    # Strip BOM and surrounding whitespace
    cleaned = raw_json.strip().lstrip("\ufeff")

    # Remove JS-style comments — these are invalid JSON but LLMs sometimes add them
    cleaned = _MULTI_LINE_COMMENT_RE.sub("", cleaned)
    cleaned = _SINGLE_LINE_COMMENT_RE.sub("", cleaned)

    # Remove trailing commas before closing brackets/braces
    # Apply multiple times in case of nested trailing commas
    for _ in range(5):
        new_cleaned = _TRAILING_COMMA_RE.sub(r"\1", cleaned)
        if new_cleaned == cleaned:
            break
        cleaned = new_cleaned

    return cleaned.strip()


# ---------------------------------------------------------------------------
# Step 3: JSON Parsing
# ---------------------------------------------------------------------------


def parse_json_string(json_string: str, raw_response: str = "") -> dict[str, Any]:
    """Parse a cleaned JSON string into a Python dictionary.

    Args:
        json_string: A (possibly cleaned) JSON string to parse.
        raw_response: The original raw LLM response, used for error context.

    Returns:
        The parsed dictionary.

    Raises:
        JSONDecodeError: If the string cannot be parsed as valid JSON.
        ParserError: If the parsed value is not a JSON object (dict).
    """
    try:
        parsed = json.loads(json_string)
    except json.JSONDecodeError as exc:
        snippet = json_string[:200] + ("..." if len(json_string) > 200 else "")
        raise JSONDecodeError(
            f"Failed to parse extracted text as JSON: {exc}. "
            f"Snippet: {snippet!r}",
            raw_response=raw_response,
        ) from exc

    if not isinstance(parsed, dict):
        raise ParserError(
            f"Expected a JSON object (dict) but got {type(parsed).__name__}. "
            "The LLM may have returned a JSON array or primitive.",
            raw_response=raw_response,
        )

    return parsed


# ---------------------------------------------------------------------------
# Step 4: Structural normalisation helpers
# ---------------------------------------------------------------------------


def _ensure_required_fields(data: dict[str, Any]) -> dict[str, Any]:
    """Add missing top-level fields with sensible defaults.

    Fills in fields that n8n requires but the LLM might omit, without
    overwriting any values the LLM did provide.

    Args:
        data: Raw parsed workflow dict from the LLM.

    Returns:
        A new dict with required fields guaranteed to be present.
    """
    defaults: dict[str, Any] = {
        "name": "Generated Workflow",
        "nodes": [],
        "connections": {},
        "active": False,
        "settings": {
            "executionOrder": "v1",
            "timezone": "UTC",
            "saveManualExecutions": True,
            "saveDataSuccessExecution": "all",
            "saveDataErrorExecution": "all",
        },
    }
    result = {**data}
    for key, default_value in defaults.items():
        if key not in result:
            logger.debug("Adding missing field '%s' with default value.", key)
            result[key] = default_value
    return result


def _normalise_node_ids(data: dict[str, Any]) -> dict[str, Any]:
    """Ensure every node has a unique non-empty ``id`` field.

    If a node is missing an ``id`` or has an empty string id, a unique
    placeholder ID is generated. This prevents schema validation failures
    when the LLM omits node IDs.

    Args:
        data: Parsed workflow dict (may be mutated).

    Returns:
        The workflow dict with all node IDs normalised.
    """
    import uuid

    nodes = data.get("nodes", [])
    if not isinstance(nodes, list):
        return data

    seen_ids: set[str] = set()
    result_nodes = []
    for node in nodes:
        if not isinstance(node, dict):
            result_nodes.append(node)
            continue
        node_id = node.get("id", "")
        if not isinstance(node_id, str) or not node_id.strip() or node_id in seen_ids:
            node_id = str(uuid.uuid4())
            logger.debug(
                "Generated new node id '%s' for node '%s'.",
                node_id,
                node.get("name", "<unnamed>"),
            )
        seen_ids.add(node_id)
        result_nodes.append({**node, "id": node_id})

    return {**data, "nodes": result_nodes}


def _normalise_node_positions(data: dict[str, Any]) -> dict[str, Any]:
    """Ensure every node has a valid two-element ``position`` array.

    Nodes missing a position or with a non-list position receive a
    calculated default position based on their index in the nodes list.

    Args:
        data: Parsed workflow dict.

    Returns:
        The workflow dict with all node positions normalised.
    """
    nodes = data.get("nodes", [])
    if not isinstance(nodes, list):
        return data

    result_nodes = []
    for idx, node in enumerate(nodes):
        if not isinstance(node, dict):
            result_nodes.append(node)
            continue
        position = node.get("position")
        if (
            not isinstance(position, list)
            or len(position) != 2
            or not all(isinstance(v, (int, float)) for v in position)
        ):
            default_position = [250.0 + idx * 250.0, 300.0]
            logger.debug(
                "Normalising position for node '%s' to %s.",
                node.get("name", "<unnamed>"),
                default_position,
            )
            node = {**node, "position": default_position}
        result_nodes.append(node)

    return {**data, "nodes": result_nodes}


def _normalise_type_versions(data: dict[str, Any]) -> dict[str, Any]:
    """Ensure every node has a valid integer ``typeVersion`` field.

    If ``typeVersion`` is missing, ``None``, or not a positive integer,
    it is set to ``1``.

    Args:
        data: Parsed workflow dict.

    Returns:
        The workflow dict with all typeVersion fields normalised.
    """
    nodes = data.get("nodes", [])
    if not isinstance(nodes, list):
        return data

    result_nodes = []
    for node in nodes:
        if not isinstance(node, dict):
            result_nodes.append(node)
            continue
        tv = node.get("typeVersion")
        if not isinstance(tv, int) or tv < 1:
            node = {**node, "typeVersion": 1}
        result_nodes.append(node)

    return {**data, "nodes": result_nodes}


def _normalise_parameters(data: dict[str, Any]) -> dict[str, Any]:
    """Ensure every node has a ``parameters`` dict (never ``None`` or absent).

    Args:
        data: Parsed workflow dict.

    Returns:
        The workflow dict with all node parameters normalised.
    """
    nodes = data.get("nodes", [])
    if not isinstance(nodes, list):
        return data

    result_nodes = []
    for node in nodes:
        if not isinstance(node, dict):
            result_nodes.append(node)
            continue
        params = node.get("parameters")
        if not isinstance(params, dict):
            node = {**node, "parameters": {}}
        result_nodes.append(node)

    return {**data, "nodes": result_nodes}


def _normalise_connections(data: dict[str, Any]) -> dict[str, Any]:
    """Normalise the connections structure to match the expected schema format.

    Ensures that connections is a dict mapping source node names to
    ``{"main": [[{"node": ..., "type": "main", "index": 0}]]}`` format.
    Drops connections that reference non-existent node names.

    Args:
        data: Parsed workflow dict.

    Returns:
        The workflow dict with connections normalised.
    """
    connections = data.get("connections", {})
    nodes = data.get("nodes", [])

    if not isinstance(connections, dict) or not isinstance(nodes, list):
        return data

    # Build set of known node names for reference validation
    node_names: set[str] = set()
    for node in nodes:
        if isinstance(node, dict) and isinstance(node.get("name"), str):
            node_names.add(node["name"])

    normalised: dict[str, Any] = {}
    for source_name, output_map in connections.items():
        if not isinstance(source_name, str) or source_name not in node_names:
            logger.debug(
                "Dropping connection from unknown source node '%s'.", source_name
            )
            continue

        if not isinstance(output_map, dict):
            continue

        normalised_output: dict[str, Any] = {}
        for output_type, groups in output_map.items():
            if not isinstance(groups, list):
                continue
            normalised_groups = []
            for group in groups:
                if not isinstance(group, list):
                    normalised_groups.append([])
                    continue
                normalised_group = []
                for conn in group:
                    if not isinstance(conn, dict):
                        continue
                    target = conn.get("node")
                    if not isinstance(target, str) or target not in node_names:
                        logger.debug(
                            "Dropping connection to unknown target node '%s'.", target
                        )
                        continue
                    normalised_conn = {
                        "node": target,
                        "type": conn.get("type", "main"),
                        "index": conn.get("index", 0),
                    }
                    normalised_group.append(normalised_conn)
                normalised_groups.append(normalised_group)
            normalised_output[output_type] = normalised_groups

        if normalised_output:
            normalised[source_name] = normalised_output

    return {**data, "connections": normalised}


def normalise_workflow_dict(data: dict[str, Any]) -> dict[str, Any]:
    """Apply all normalisation steps to a raw parsed workflow dict.

    Normalisation is applied before Pydantic schema validation to maximise
    the chance that valid-but-imperfect LLM output is accepted.

    The following normalisations are applied in order:

    1. :func:`_ensure_required_fields` — add missing top-level fields.
    2. :func:`_normalise_node_ids` — generate missing/duplicate node IDs.
    3. :func:`_normalise_node_positions` — fix invalid position arrays.
    4. :func:`_normalise_type_versions` — default typeVersion to 1.
    5. :func:`_normalise_parameters` — default parameters to ``{}``.
    6. :func:`_normalise_connections` — clean up connections structure.

    Args:
        data: Raw dict parsed from the LLM's JSON output.

    Returns:
        A normalised dict ready for :func:`~n8n_forge.schema.validate_workflow_dict`.
    """
    data = _ensure_required_fields(data)
    data = _normalise_node_ids(data)
    data = _normalise_node_positions(data)
    data = _normalise_type_versions(data)
    data = _normalise_parameters(data)
    data = _normalise_connections(data)
    return data


# ---------------------------------------------------------------------------
# Public high-level API
# ---------------------------------------------------------------------------


def parse_workflow_response(
    raw_response: str,
    strict: bool = False,
) -> WorkflowSchema:
    """Parse and validate an LLM response into a :class:`WorkflowSchema`.

    This is the primary public function of this module. It orchestrates the
    full pipeline: extraction → cleaning → parsing → normalisation → validation.

    Args:
        raw_response: The raw string returned by the OpenAI API. May contain
            prose, fenced code blocks, and/or bare JSON.
        strict: If ``True``, skip normalisation and apply schema validation
            directly to the raw parsed dict. This is stricter and will raise
            :class:`WorkflowValidationError` for any schema deviation.
            Defaults to ``False`` (normalisation is applied).

    Returns:
        A validated :class:`~n8n_forge.schema.WorkflowSchema` instance.

    Raises:
        JSONExtractionError: If no JSON object can be found in the response.
        JSONDecodeError: If the extracted text is not valid JSON.
        ParserError: If the JSON is not a top-level object (dict).
        WorkflowValidationError: If the parsed dict fails schema validation.

    Example::

        raw = '''
        ```json
        {
          "name": "Daily Report",
          "nodes": [{"id": "1", "name": "Schedule Trigger",
                     "type": "n8n-nodes-base.scheduleTrigger",
                     "typeVersion": 1, "position": [250, 300], "parameters": {}}],
          "connections": {},
          "active": false
        }
        ```
        '''
        workflow = parse_workflow_response(raw)
        assert workflow.name == "Daily Report"
    """
    logger.debug(
        "Parsing LLM response (%d chars, strict=%s).", len(raw_response), strict
    )

    # Step 1: Extract JSON string from the response
    json_string = extract_json_string(raw_response)

    # Step 2: Clean common LLM artefacts
    cleaned = clean_json_string(json_string)

    # Step 3: Parse into a Python dict
    parsed_dict = parse_json_string(cleaned, raw_response=raw_response)

    # Step 4: Normalise (unless strict mode)
    if not strict:
        parsed_dict = normalise_workflow_dict(parsed_dict)

    # Step 5: Validate against the schema
    # WorkflowValidationError is intentionally not caught here — let it propagate
    workflow = validate_workflow_dict(parsed_dict)

    logger.debug(
        "Successfully parsed workflow '%s' with %d node(s) and %d connection(s).",
        workflow.name,
        len(workflow.nodes),
        workflow.get_connection_count(),
    )

    return workflow


def workflow_to_json_string(workflow: WorkflowSchema, indent: int = 2) -> str:
    """Serialise a validated :class:`WorkflowSchema` to a JSON string.

    Uses :meth:`~n8n_forge.schema.WorkflowSchema.model_dump_n8n` to produce
    the n8n-compatible dict representation before serialising to JSON.

    Args:
        workflow: A validated :class:`~n8n_forge.schema.WorkflowSchema` instance.
        indent: Number of spaces used for JSON indentation.
            Defaults to ``2``.

    Returns:
        A pretty-printed JSON string representing the workflow.

    Example::

        workflow = parse_workflow_response(raw_llm_response)
        json_str = workflow_to_json_string(workflow)
        with open("workflow.json", "w") as f:
            f.write(json_str)
    """
    return json.dumps(workflow.model_dump_n8n(), indent=indent, ensure_ascii=False)
