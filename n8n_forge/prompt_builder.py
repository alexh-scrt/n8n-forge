"""Prompt builder module for n8n Forge.

This module constructs the structured system and user prompts sent to the
OpenAI LLM. It uses a Jinja2 template for the system prompt and injects
relevant n8n node catalog context based on the user's description to
improve accuracy and reduce hallucination of invalid node types.

Example usage::

    from n8n_forge.prompt_builder import build_messages

    messages = build_messages(
        description="Check competitor prices weekly and add to Google Sheets",
        existing_workflow_json=None,
    )
    # messages is a list of {"role": ..., "content": ...} dicts
    # ready to pass directly to the OpenAI chat completions API.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, TemplateNotFound

from n8n_forge.node_catalog import (
    NodeCatalogEntry,
    catalog_to_prompt_text,
    get_nodes_by_category,
    get_trigger_nodes,
    search_nodes,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Directory containing Jinja2 templates.
_TEMPLATES_DIR = Path(__file__).parent / "templates"

#: Name of the system prompt Jinja2 template file.
_SYSTEM_PROMPT_TEMPLATE = "system_prompt.j2"

#: Maximum number of nodes to include in the prompt catalog section.
_MAX_CATALOG_NODES = 25

#: Maximum number of search-result nodes to inject for a given description.
_MAX_SEARCH_NODES = 10


# ---------------------------------------------------------------------------
# Jinja2 environment
# ---------------------------------------------------------------------------

def _get_jinja_env() -> Environment:
    """Create and return a configured Jinja2 Environment.

    Returns:
        A :class:`jinja2.Environment` instance configured to load templates
        from the ``n8n_forge/templates/`` directory.

    Raises:
        RuntimeError: If the templates directory does not exist.
    """
    if not _TEMPLATES_DIR.is_dir():
        raise RuntimeError(
            f"Templates directory not found: {_TEMPLATES_DIR}. "
            "Ensure the package is installed correctly."
        )
    return Environment(
        loader=FileSystemLoader(str(_TEMPLATES_DIR)),
        autoescape=False,  # We handle escaping manually where needed
        trim_blocks=True,
        lstrip_blocks=True,
        keep_trailing_newline=True,
    )


# ---------------------------------------------------------------------------
# Node selection helpers
# ---------------------------------------------------------------------------

def _select_relevant_nodes(description: str) -> list[NodeCatalogEntry]:
    """Select node catalog entries most relevant to the given description.

    Uses keyword search against the description to find relevant nodes.
    Always includes all trigger nodes to ensure the LLM picks an appropriate
    trigger. Deduplicates and caps results at ``_MAX_CATALOG_NODES``.

    Args:
        description: Plain-English automation description from the user.

    Returns:
        Deduplicated list of :class:`~n8n_forge.node_catalog.NodeCatalogEntry`
        objects most relevant to the description, with triggers first.
    """
    seen: set[str] = set()
    selected: list[NodeCatalogEntry] = []

    # Always include triggers so the LLM can pick the right one
    for entry in get_trigger_nodes():
        if entry.type_name not in seen:
            seen.add(entry.type_name)
            selected.append(entry)

    # Search for nodes relevant to the description
    search_results = search_nodes(description)
    for entry in search_results[:_MAX_SEARCH_NODES]:
        if entry.type_name not in seen:
            seen.add(entry.type_name)
            selected.append(entry)

    # Fill remaining slots with core utility nodes
    core_nodes = get_nodes_by_category("Core")
    for entry in core_nodes:
        if len(selected) >= _MAX_CATALOG_NODES:
            break
        if entry.type_name not in seen:
            seen.add(entry.type_name)
            selected.append(entry)

    return selected[:_MAX_CATALOG_NODES]


# ---------------------------------------------------------------------------
# Prompt rendering
# ---------------------------------------------------------------------------

def render_system_prompt(
    node_catalog_text: str,
    example_workflow_json: str | None = None,
) -> str:
    """Render the system prompt using the Jinja2 template.

    Args:
        node_catalog_text: Formatted text listing available n8n nodes,
            as produced by
            :func:`~n8n_forge.node_catalog.catalog_to_prompt_text`.
        example_workflow_json: Optional JSON string of a minimal example
            workflow to include in the prompt for few-shot grounding.
            If ``None``, no example is included.

    Returns:
        Rendered system prompt string ready to send to the LLM.

    Raises:
        RuntimeError: If the system prompt template cannot be found.
    """
    env = _get_jinja_env()
    try:
        template = env.get_template(_SYSTEM_PROMPT_TEMPLATE)
    except TemplateNotFound as exc:
        raise RuntimeError(
            f"System prompt template '{_SYSTEM_PROMPT_TEMPLATE}' not found "
            f"in {_TEMPLATES_DIR}"
        ) from exc

    return template.render(
        node_catalog_text=node_catalog_text,
        example_workflow_json=example_workflow_json,
    )


def build_user_message(description: str, existing_workflow_json: str | None = None) -> str:
    """Build the user-role message content for the LLM request.

    For a fresh generation the message simply states the description.
    For a refinement pass (when ``existing_workflow_json`` is provided)
    the message includes the existing workflow JSON so the LLM can
    modify it rather than generating from scratch.

    Args:
        description: Plain-English automation description or refinement
            instruction from the user.
        existing_workflow_json: JSON string of a previously generated
            workflow to be refined. Pass ``None`` for initial generation.

    Returns:
        A string to use as the ``content`` field of the user message.
    """
    if existing_workflow_json is None:
        return (
            f"Generate a complete n8n workflow JSON for the following automation:\n\n"
            f"{description.strip()}"
        )

    return (
        f"Here is an existing n8n workflow JSON:\n\n"
        f"```json\n{existing_workflow_json.strip()}\n```\n\n"
        f"Please refine the workflow according to the following instructions:\n\n"
        f"{description.strip()}\n\n"
        f"Return the complete updated workflow JSON."
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_messages(
    description: str,
    existing_workflow_json: str | None = None,
    model_context_hint: str | None = None,
) -> list[dict[str, str]]:
    """Build the full list of chat messages for the OpenAI API call.

    Constructs a system message (with node catalog context) and a user
    message (with the automation description or refinement instruction)
    in the format expected by the OpenAI Chat Completions API.

    Args:
        description: Plain-English description of the desired automation,
            or a refinement instruction when ``existing_workflow_json``
            is provided.
        existing_workflow_json: JSON string of a previously generated
            workflow. When provided, the prompt instructs the LLM to
            refine the existing workflow rather than generate a new one.
            Defaults to ``None`` (fresh generation).
        model_context_hint: Optional string hint about the intended context
            or domain (e.g. ``"e-commerce"``). When provided, additional
            targeted node search is performed to improve relevance.
            Defaults to ``None``.

    Returns:
        A list of message dicts in the format::

            [
                {"role": "system", "content": "..."},
                {"role": "user",   "content": "..."},
            ]

    Raises:
        ValueError: If ``description`` is empty or whitespace-only.
        RuntimeError: If the Jinja2 template cannot be rendered.

    Example::

        messages = build_messages(
            description="Send a Slack message every Monday with a sales summary"
        )
        # Pass messages to openai.chat.completions.create(messages=messages)
    """
    if not description or not description.strip():
        raise ValueError("Automation description must not be empty.")

    # --- Select relevant nodes -------------------------------------------
    combined_query = description
    if model_context_hint:
        combined_query = f"{description} {model_context_hint}"

    relevant_nodes = _select_relevant_nodes(combined_query)
    node_catalog_text = catalog_to_prompt_text(
        entries=relevant_nodes,
        max_nodes=_MAX_CATALOG_NODES,
    )

    # --- Build the minimal example workflow for few-shot grounding ----------
    example_json = _get_example_workflow_json()

    # --- Render system prompt -----------------------------------------------
    system_content = render_system_prompt(
        node_catalog_text=node_catalog_text,
        example_workflow_json=example_json,
    )

    # --- Build user message -------------------------------------------------
    user_content = build_user_message(
        description=description,
        existing_workflow_json=existing_workflow_json,
    )

    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]


def _get_example_workflow_json() -> str:
    """Return a minimal example n8n workflow JSON string for few-shot prompting.

    The example illustrates the expected output structure: a ScheduleTrigger
    connected to an HTTP Request node, with correct connections format.

    Returns:
        A compact JSON string representing a valid minimal n8n workflow.
    """
    example: dict[str, Any] = {
        "name": "Example: Fetch Data Every Hour",
        "nodes": [
            {
                "id": "uuid-1111-aaaa",
                "name": "Schedule Trigger",
                "type": "n8n-nodes-base.scheduleTrigger",
                "typeVersion": 1,
                "position": [250, 300],
                "parameters": {
                    "rule": {
                        "interval": [{"field": "hours", "hoursInterval": 1}]
                    }
                },
            },
            {
                "id": "uuid-2222-bbbb",
                "name": "HTTP Request",
                "type": "n8n-nodes-base.httpRequest",
                "typeVersion": 4,
                "position": [500, 300],
                "parameters": {
                    "method": "GET",
                    "url": "https://api.example.com/data",
                    "responseFormat": "json",
                },
            },
        ],
        "connections": {
            "Schedule Trigger": {
                "main": [
                    [
                        {"node": "HTTP Request", "type": "main", "index": 0}
                    ]
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
    }
    return json.dumps(example, indent=2)
