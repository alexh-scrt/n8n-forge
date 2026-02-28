"""Static catalog of common n8n node types used to ground LLM prompt generation.

This module provides a curated dictionary of n8n nodes, their type identifiers,
descriptions, common parameters, and usage hints. The catalog is used by the
prompt builder to inject accurate node information into LLM prompts, reducing
hallucination of invalid node types and improving the quality of generated workflows.

Example usage::

    from n8n_forge.node_catalog import NODE_CATALOG, get_node_by_type, search_nodes

    # Look up a specific node
    node_info = get_node_by_type("n8n-nodes-base.httpRequest")
    print(node_info.description)

    # Find nodes related to scheduling
    results = search_nodes("schedule")
    for r in results:
        print(r.display_name, r.type_name)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class NodeParameter:
    """Describes a single configurable parameter on an n8n node.

    Attributes:
        name: The parameter key name as used in n8n's ``parameters`` dict.
        description: Human-readable description of what this parameter does.
        required: Whether this parameter must be provided for the node to function.
        default: The default value used when the parameter is not specified.
        param_type: The data type of the parameter value.
    """

    name: str
    description: str
    required: bool = False
    default: Any = None
    param_type: str = "string"


@dataclass(frozen=True)
class NodeCatalogEntry:
    """A single entry in the n8n node catalog.

    Attributes:
        type_name: Full n8n type identifier (e.g. ``n8n-nodes-base.httpRequest``).
        display_name: Human-readable name shown in n8n's UI.
        description: What this node does.
        category: Functional category (e.g. ``"Trigger"``, ``"Data Transformation"``).
        type_version: Default type version to use.
        common_parameters: List of commonly configured parameters.
        use_cases: Example automation scenarios where this node is useful.
        credentials_required: Credential types this node typically requires.
        is_trigger: Whether this node starts a workflow.
    """

    type_name: str
    display_name: str
    description: str
    category: str
    type_version: int = 1
    common_parameters: tuple[NodeParameter, ...] = field(default_factory=tuple)
    use_cases: tuple[str, ...] = field(default_factory=tuple)
    credentials_required: tuple[str, ...] = field(default_factory=tuple)
    is_trigger: bool = False

    def to_prompt_text(self) -> str:
        """Format this catalog entry as concise text suitable for LLM prompt injection.

        Returns:
            A multi-line string summarising the node type, description,
            category, and common use cases.
        """
        lines = [
            f"Node: {self.display_name}",
            f"  Type: {self.type_name}",
            f"  Category: {self.category}",
            f"  Description: {self.description}",
        ]
        if self.use_cases:
            lines.append("  Use cases: " + "; ".join(self.use_cases))
        if self.credentials_required:
            lines.append(
                "  Requires credentials: " + ", ".join(self.credentials_required)
            )
        if self.common_parameters:
            param_names = ", ".join(p.name for p in self.common_parameters)
            lines.append(f"  Key parameters: {param_names}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Node catalog data
# ---------------------------------------------------------------------------

NODE_CATALOG: dict[str, NodeCatalogEntry] = {
    # ------------------------------------------------------------------
    # Triggers
    # ------------------------------------------------------------------
    "n8n-nodes-base.scheduleTrigger": NodeCatalogEntry(
        type_name="n8n-nodes-base.scheduleTrigger",
        display_name="Schedule Trigger",
        description=(
            "Starts the workflow on a recurring schedule. Supports cron expressions, "
            "interval-based scheduling (every N minutes/hours/days/weeks), and specific "
            "time-of-day triggers. Use this for any periodic automation."
        ),
        category="Trigger",
        type_version=1,
        is_trigger=True,
        common_parameters=(
            NodeParameter(
                name="rule",
                description="Schedule rule defining when to trigger",
                required=True,
                param_type="object",
            ),
        ),
        use_cases=(
            "Run a workflow every Monday at 9am",
            "Fetch data every 15 minutes",
            "Send weekly reports",
            "Daily database cleanup",
        ),
    ),
    "n8n-nodes-base.webhook": NodeCatalogEntry(
        type_name="n8n-nodes-base.webhook",
        display_name="Webhook",
        description=(
            "Starts the workflow when an HTTP request is received at a unique URL. "
            "Supports GET, POST, PUT, PATCH, DELETE methods. The incoming request data "
            "is available to downstream nodes. Ideal for integrating with external "
            "services that send webhooks (e.g., Stripe, GitHub, Shopify)."
        ),
        category="Trigger",
        type_version=1,
        is_trigger=True,
        common_parameters=(
            NodeParameter(
                name="httpMethod",
                description="HTTP method to listen for (GET, POST, etc.)",
                required=True,
                default="POST",
            ),
            NodeParameter(
                name="path",
                description="URL path segment for the webhook endpoint",
                required=True,
                param_type="string",
            ),
            NodeParameter(
                name="responseMode",
                description="When to send the webhook response",
                default="onReceived",
            ),
        ),
        use_cases=(
            "Receive Stripe payment events",
            "Process GitHub pull request notifications",
            "Accept form submissions",
            "Listen for Shopify order events",
        ),
    ),
    "n8n-nodes-base.emailReadImap": NodeCatalogEntry(
        type_name="n8n-nodes-base.emailReadImap",
        display_name="Email Trigger (IMAP)",
        description=(
            "Triggers the workflow when a new email arrives in an IMAP mailbox. "
            "Polls the mailbox on a configurable interval and emits one item per "
            "new email, including subject, body, sender, and attachments."
        ),
        category="Trigger",
        type_version=2,
        is_trigger=True,
        credentials_required=("imap",),
        common_parameters=(
            NodeParameter(
                name="mailbox",
                description="Mailbox folder to monitor",
                default="INBOX",
            ),
            NodeParameter(
                name="postProcessAction",
                description="Action to take after reading (mark as read, delete, etc.)",
                default="read",
            ),
        ),
        use_cases=(
            "Process incoming support emails",
            "Parse order confirmation emails",
            "Monitor a shared inbox and route emails",
        ),
    ),
    "n8n-nodes-base.manualTrigger": NodeCatalogEntry(
        type_name="n8n-nodes-base.manualTrigger",
        display_name="Manual Trigger",
        description=(
            "Starts the workflow manually from the n8n UI by clicking 'Execute'. "
            "Useful for testing workflows or on-demand one-off executions."
        ),
        category="Trigger",
        type_version=1,
        is_trigger=True,
        use_cases=(
            "Test a workflow before scheduling",
            "Run a one-off data migration",
        ),
    ),
    "n8n-nodes-base.formTrigger": NodeCatalogEntry(
        type_name="n8n-nodes-base.formTrigger",
        display_name="n8n Form Trigger",
        description=(
            "Generates a hosted web form that, when submitted, starts the workflow. "
            "No external form service required. Form fields are configurable directly "
            "in n8n and the submitted data is passed to downstream nodes."
        ),
        category="Trigger",
        type_version=2,
        is_trigger=True,
        common_parameters=(
            NodeParameter(
                name="formTitle",
                description="Title displayed at the top of the form",
                required=True,
            ),
            NodeParameter(
                name="formFields",
                description="List of form field definitions",
                required=True,
                param_type="array",
            ),
        ),
        use_cases=(
            "Internal request intake forms",
            "Customer feedback collection",
            "Event registration",
        ),
    ),
    # ------------------------------------------------------------------
    # Core / Utility
    # ------------------------------------------------------------------
    "n8n-nodes-base.httpRequest": NodeCatalogEntry(
        type_name="n8n-nodes-base.httpRequest",
        display_name="HTTP Request",
        description=(
            "Makes HTTP requests to any REST API or URL. Supports all HTTP methods, "
            "custom headers, authentication (Bearer token, Basic Auth, OAuth2), "
            "query parameters, JSON/form bodies, and pagination. Use this to "
            "integrate with any service that exposes an HTTP API."
        ),
        category="Core",
        type_version=4,
        common_parameters=(
            NodeParameter(
                name="method",
                description="HTTP method (GET, POST, PUT, PATCH, DELETE)",
                required=True,
                default="GET",
            ),
            NodeParameter(
                name="url",
                description="URL to send the request to",
                required=True,
            ),
            NodeParameter(
                name="authentication",
                description="Authentication method",
                default="none",
            ),
            NodeParameter(
                name="sendHeaders",
                description="Whether to send custom headers",
                default=False,
                param_type="boolean",
            ),
            NodeParameter(
                name="sendBody",
                description="Whether to send a request body",
                default=False,
                param_type="boolean",
            ),
            NodeParameter(
                name="responseFormat",
                description="Expected response format (json, text, binary)",
                default="json",
            ),
        ),
        use_cases=(
            "Fetch data from a REST API",
            "Post data to a webhook",
            "Call an internal microservice",
            "Scrape publicly accessible JSON endpoints",
        ),
    ),
    "n8n-nodes-base.code": NodeCatalogEntry(
        type_name="n8n-nodes-base.code",
        display_name="Code",
        description=(
            "Executes custom JavaScript or Python code within the workflow. "
            "Input items are available as the ``$input.all()`` array. "
            "The node must return an array of items. Ideal for data transformation, "
            "filtering, or logic that no pre-built node covers."
        ),
        category="Core",
        type_version=2,
        common_parameters=(
            NodeParameter(
                name="language",
                description="Programming language (javaScript or python)",
                default="javaScript",
            ),
            NodeParameter(
                name="jsCode",
                description="JavaScript code to execute",
                param_type="string",
            ),
        ),
        use_cases=(
            "Transform or reshape data between nodes",
            "Apply custom business logic",
            "Filter items based on complex conditions",
            "Merge or split data arrays",
        ),
    ),
    "n8n-nodes-base.set": NodeCatalogEntry(
        type_name="n8n-nodes-base.set",
        display_name="Edit Fields (Set)",
        description=(
            "Adds, edits, or removes fields on each item passing through. "
            "Supports fixed values, expressions referencing other fields, "
            "and dot-notation for nested objects. Use to shape data before "
            "passing it to the next node."
        ),
        category="Core",
        type_version=3,
        common_parameters=(
            NodeParameter(
                name="mode",
                description="Operation mode (manual, expression, raw)",
                default="manual",
            ),
            NodeParameter(
                name="assignments",
                description="Field assignments to apply",
                required=True,
                param_type="array",
            ),
        ),
        use_cases=(
            "Rename or restructure fields",
            "Add computed fields using expressions",
            "Remove unwanted fields",
            "Prepare data for API calls",
        ),
    ),
    "n8n-nodes-base.if": NodeCatalogEntry(
        type_name="n8n-nodes-base.if",
        display_name="IF",
        description=(
            "Routes items to one of two outputs (true/false) based on one or more "
            "conditions. Conditions can compare strings, numbers, booleans, dates, "
            "and arrays. Multiple conditions can be combined with AND/OR logic."
        ),
        category="Core",
        type_version=2,
        common_parameters=(
            NodeParameter(
                name="conditions",
                description="Conditions to evaluate",
                required=True,
                param_type="object",
            ),
            NodeParameter(
                name="combineOperation",
                description="How to combine multiple conditions (all/any)",
                default="all",
            ),
        ),
        use_cases=(
            "Route items based on a field value",
            "Skip processing for certain conditions",
            "Branch workflow logic",
        ),
    ),
    "n8n-nodes-base.switch": NodeCatalogEntry(
        type_name="n8n-nodes-base.switch",
        display_name="Switch",
        description=(
            "Routes items to one of many outputs based on a value match, "
            "similar to a switch/case statement. Supports up to 10 output branches."
        ),
        category="Core",
        type_version=3,
        common_parameters=(
            NodeParameter(
                name="mode",
                description="Match mode (rules or expression)",
                default="rules",
            ),
            NodeParameter(
                name="rules",
                description="List of match rules and their output index",
                required=True,
                param_type="array",
            ),
        ),
        use_cases=(
            "Route by category or status",
            "Handle multiple event types from a webhook",
            "Multi-tenant workflow routing",
        ),
    ),
    "n8n-nodes-base.merge": NodeCatalogEntry(
        type_name="n8n-nodes-base.merge",
        display_name="Merge",
        description=(
            "Merges items from two or more input branches into a single output. "
            "Supports append, combine by position, combine by field, and "
            "SQL-style join operations (inner, left, outer)."
        ),
        category="Core",
        type_version=3,
        common_parameters=(
            NodeParameter(
                name="mode",
                description="Merge mode (append, combine, chooseBranch, etc.)",
                default="append",
            ),
        ),
        use_cases=(
            "Combine results from two API calls",
            "Merge enriched data back into a main flow",
            "Join lookup data with main records",
        ),
    ),
    "n8n-nodes-base.splitInBatches": NodeCatalogEntry(
        type_name="n8n-nodes-base.splitInBatches",
        display_name="Loop Over Items",
        description=(
            "Splits a list of items into smaller batches and loops through them. "
            "Use to process large datasets without hitting API rate limits, or "
            "to iterate through items one at a time."
        ),
        category="Core",
        type_version=3,
        common_parameters=(
            NodeParameter(
                name="batchSize",
                description="Number of items per batch",
                required=True,
                default=10,
                param_type="number",
            ),
        ),
        use_cases=(
            "Process large lists in chunks",
            "Iterate over rows from a spreadsheet",
            "Send bulk messages in batches",
        ),
    ),
    "n8n-nodes-base.wait": NodeCatalogEntry(
        type_name="n8n-nodes-base.wait",
        display_name="Wait",
        description=(
            "Pauses the workflow execution for a specified duration or until "
            "a webhook is called. Useful for rate limiting, waiting for async "
            "processes, or building human-in-the-loop approval steps."
        ),
        category="Core",
        type_version=1,
        common_parameters=(
            NodeParameter(
                name="resume",
                description="Resume condition (timeInterval, webhook, etc.)",
                default="timeInterval",
            ),
            NodeParameter(
                name="amount",
                description="Amount of time to wait",
                default=1,
                param_type="number",
            ),
            NodeParameter(
                name="unit",
                description="Time unit (seconds, minutes, hours, days)",
                default="hours",
            ),
        ),
        use_cases=(
            "Wait between API calls to avoid rate limits",
            "Pause for human approval",
            "Delay follow-up emails",
        ),
    ),
    "n8n-nodes-base.noOp": NodeCatalogEntry(
        type_name="n8n-nodes-base.noOp",
        display_name="No Operation",
        description=(
            "Passes items through unchanged. Used as a placeholder, merge point, "
            "or end-of-branch marker."
        ),
        category="Core",
        type_version=1,
        use_cases=("Placeholder node", "Branch merge point"),
    ),
    "n8n-nodes-base.stickyNote": NodeCatalogEntry(
        type_name="n8n-nodes-base.stickyNote",
        display_name="Sticky Note",
        description=(
            "Adds a text note to the canvas for documentation. Does not execute "
            "any logic and is not connected to other nodes."
        ),
        category="Utility",
        type_version=1,
        common_parameters=(
            NodeParameter(
                name="content",
                description="Markdown text content of the note",
                default="",
            ),
        ),
        use_cases=("Document workflow steps", "Add team notes"),
    ),
    # ------------------------------------------------------------------
    # Communication
    # ------------------------------------------------------------------
    "n8n-nodes-base.slack": NodeCatalogEntry(
        type_name="n8n-nodes-base.slack",
        display_name="Slack",
        description=(
            "Interacts with the Slack API. Supports sending messages to channels "
            "or users, creating channels, uploading files, managing reactions, "
            "and more. Requires a Slack OAuth2 or bot token credential."
        ),
        category="Communication",
        type_version=2,
        credentials_required=("slackApi",),
        common_parameters=(
            NodeParameter(
                name="resource",
                description="Resource type (message, channel, user, etc.)",
                default="message",
            ),
            NodeParameter(
                name="operation",
                description="Operation to perform (post, update, delete, etc.)",
                default="post",
            ),
            NodeParameter(
                name="channel",
                description="Channel name or ID to send the message to",
                required=True,
            ),
            NodeParameter(
                name="text",
                description="Message text content",
            ),
        ),
        use_cases=(
            "Send daily digest messages",
            "Alert on errors or anomalies",
            "Post reports to a Slack channel",
            "Notify team of new leads or orders",
        ),
    ),
    "n8n-nodes-base.sendEmail": NodeCatalogEntry(
        type_name="n8n-nodes-base.sendEmail",
        display_name="Send Email",
        description=(
            "Sends emails via SMTP. Supports HTML and plain text bodies, "
            "attachments, CC/BCC, and reply-to fields. Configure with "
            "your SMTP server credentials."
        ),
        category="Communication",
        type_version=2,
        credentials_required=("smtp",),
        common_parameters=(
            NodeParameter(name="fromEmail", description="Sender email address", required=True),
            NodeParameter(name="toEmail", description="Recipient email address", required=True),
            NodeParameter(name="subject", description="Email subject line", required=True),
            NodeParameter(name="emailType", description="Body format (html or text)", default="html"),
            NodeParameter(name="message", description="Email body content", required=True),
        ),
        use_cases=(
            "Send automated email reports",
            "Email confirmation on form submission",
            "Alert stakeholders of workflow results",
            "Send scheduled newsletters",
        ),
    ),
    "n8n-nodes-base.gmail": NodeCatalogEntry(
        type_name="n8n-nodes-base.gmail",
        display_name="Gmail",
        description=(
            "Reads and sends Gmail emails using the Gmail API. Supports "
            "creating drafts, sending emails with attachments, searching "
            "messages, and managing labels. Requires Gmail OAuth2 credentials."
        ),
        category="Communication",
        type_version=2,
        credentials_required=("gmailOAuth2",),
        common_parameters=(
            NodeParameter(name="resource", description="Resource (message, label, draft)", default="message"),
            NodeParameter(name="operation", description="Operation (send, get, getAll, etc.)", default="send"),
            NodeParameter(name="sendTo", description="Recipient email address"),
            NodeParameter(name="subject", description="Email subject line"),
            NodeParameter(name="message", description="Email body content"),
        ),
        use_cases=(
            "Send emails from a Gmail account",
            "Read and process incoming Gmail messages",
            "Search Gmail for specific messages",
        ),
    ),
    "n8n-nodes-base.microsoftTeams": NodeCatalogEntry(
        type_name="n8n-nodes-base.microsoftTeams",
        display_name="Microsoft Teams",
        description=(
            "Sends messages and manages channels in Microsoft Teams. "
            "Requires Microsoft OAuth2 credentials."
        ),
        category="Communication",
        type_version=2,
        credentials_required=("microsoftTeamsOAuth2Api",),
        common_parameters=(
            NodeParameter(name="resource", description="Resource (channel, chatMessage)", default="channel"),
            NodeParameter(name="operation", description="Operation to perform", default="create"),
        ),
        use_cases=(
            "Post workflow results to a Teams channel",
            "Send alerts to the operations team",
        ),
    ),
    "n8n-nodes-base.telegram": NodeCatalogEntry(
        type_name="n8n-nodes-base.telegram",
        display_name="Telegram",
        description=(
            "Sends messages and manages chats via the Telegram Bot API. "
            "Requires a Telegram Bot token."
        ),
        category="Communication",
        type_version=1,
        credentials_required=("telegramApi",),
        common_parameters=(
            NodeParameter(name="resource", description="Resource (message, callback)", default="message"),
            NodeParameter(name="operation", description="Operation (sendMessage, etc.)", default="sendMessage"),
            NodeParameter(name="chatId", description="Target chat ID", required=True),
            NodeParameter(name="text", description="Message text", required=True),
        ),
        use_cases=(
            "Send Telegram notifications",
            "Alert on monitoring thresholds",
            "Build a Telegram bot workflow",
        ),
    ),
    # ------------------------------------------------------------------
    # Productivity / Storage
    # ------------------------------------------------------------------
    "n8n-nodes-base.googleSheets": NodeCatalogEntry(
        type_name="n8n-nodes-base.googleSheets",
        display_name="Google Sheets",
        description=(
            "Reads and writes data in Google Sheets. Supports appending rows, "
            "updating cells, reading ranges, and clearing data. "
            "Requires Google Sheets OAuth2 credentials."
        ),
        category="Productivity",
        type_version=4,
        credentials_required=("googleSheetsOAuth2Api",),
        common_parameters=(
            NodeParameter(name="resource", description="Resource (spreadsheet, sheet)", default="sheet"),
            NodeParameter(name="operation", description="Operation (append, read, update, clear)", default="append"),
            NodeParameter(name="documentId", description="Google Spreadsheet ID", required=True),
            NodeParameter(name="sheetName", description="Sheet tab name", required=True),
            NodeParameter(name="columns", description="Column mapping configuration", param_type="object"),
        ),
        use_cases=(
            "Log results to a Google Sheet",
            "Read a list of items to process",
            "Update prices or inventory from an API",
            "Append scraped data to a tracker",
        ),
    ),
    "n8n-nodes-base.airtable": NodeCatalogEntry(
        type_name="n8n-nodes-base.airtable",
        display_name="Airtable",
        description=(
            "Reads, creates, updates, and deletes records in Airtable bases. "
            "Requires an Airtable personal access token or OAuth2 credential."
        ),
        category="Productivity",
        type_version=2,
        credentials_required=("airtableTokenApi",),
        common_parameters=(
            NodeParameter(name="operation", description="Operation (list, get, create, update, delete)", default="list"),
            NodeParameter(name="base", description="Airtable base ID", required=True),
            NodeParameter(name="table", description="Table name or ID", required=True),
        ),
        use_cases=(
            "Sync data to an Airtable database",
            "Read a list of contacts to email",
            "Log form submissions to a table",
        ),
    ),
    "n8n-nodes-base.googleDrive": NodeCatalogEntry(
        type_name="n8n-nodes-base.googleDrive",
        display_name="Google Drive",
        description=(
            "Manages files and folders in Google Drive. Supports uploading, "
            "downloading, moving, copying, and deleting files. "
            "Requires Google Drive OAuth2 credentials."
        ),
        category="Productivity",
        type_version=3,
        credentials_required=("googleDriveOAuth2Api",),
        common_parameters=(
            NodeParameter(name="resource", description="Resource (file, folder)", default="file"),
            NodeParameter(name="operation", description="Operation (upload, download, list, etc.)", default="upload"),
        ),
        use_cases=(
            "Save generated reports to Drive",
            "Read CSV files from Drive",
            "Archive files automatically",
        ),
    ),
    "n8n-nodes-base.notion": NodeCatalogEntry(
        type_name="n8n-nodes-base.notion",
        display_name="Notion",
        description=(
            "Reads and writes Notion pages, databases, and blocks. "
            "Supports creating pages, querying databases, updating properties, "
            "and appending blocks. Requires a Notion internal integration token."
        ),
        category="Productivity",
        type_version=2,
        credentials_required=("notionApi",),
        common_parameters=(
            NodeParameter(name="resource", description="Resource (page, database, block)", default="page"),
            NodeParameter(name="operation", description="Operation (get, getAll, create, update, archive)", default="get"),
        ),
        use_cases=(
            "Log tasks to a Notion database",
            "Create Notion pages from form submissions",
            "Sync CRM data with a Notion table",
        ),
    ),
    "n8n-nodes-base.microsoftExcel": NodeCatalogEntry(
        type_name="n8n-nodes-base.microsoftExcel",
        display_name="Microsoft Excel 365",
        description=(
            "Reads and writes Microsoft Excel 365 workbooks stored on OneDrive. "
            "Supports appending rows, reading ranges, and table operations. "
            "Requires Microsoft OAuth2 credentials."
        ),
        category="Productivity",
        type_version=2,
        credentials_required=("microsoftExcelOAuth2Api",),
        common_parameters=(
            NodeParameter(name="resource", description="Resource (table, worksheet)", default="table"),
            NodeParameter(name="operation", description="Operation (addRow, getAll, etc.)", default="addRow"),
        ),
        use_cases=(
            "Append results to an Excel report",
            "Read product data from a spreadsheet",
        ),
    ),
    # ------------------------------------------------------------------
    # CRM / Sales
    # ------------------------------------------------------------------
    "n8n-nodes-base.hubspot": NodeCatalogEntry(
        type_name="n8n-nodes-base.hubspot",
        display_name="HubSpot",
        description=(
            "Manages HubSpot CRM records including contacts, companies, deals, "
            "and tickets. Supports create, read, update, delete, and search. "
            "Requires HubSpot API credentials."
        ),
        category="CRM",
        type_version=2,
        credentials_required=("hubspotApi",),
        common_parameters=(
            NodeParameter(name="resource", description="Resource type (contact, deal, company, ticket)", default="contact"),
            NodeParameter(name="operation", description="Operation (create, get, update, delete, search)", default="create"),
        ),
        use_cases=(
            "Create CRM contacts from form submissions",
            "Update deal stages automatically",
            "Sync leads from a landing page",
        ),
    ),
    "n8n-nodes-base.salesforce": NodeCatalogEntry(
        type_name="n8n-nodes-base.salesforce",
        display_name="Salesforce",
        description=(
            "Manages Salesforce records: leads, contacts, accounts, opportunities, "
            "cases, and custom objects. Requires Salesforce OAuth2 credentials."
        ),
        category="CRM",
        type_version=1,
        credentials_required=("salesforceOAuth2Api",),
        common_parameters=(
            NodeParameter(name="resource", description="Salesforce object type", default="lead"),
            NodeParameter(name="operation", description="Operation (create, get, update, delete, query)", default="create"),
        ),
        use_cases=(
            "Create Salesforce leads from web forms",
            "Update opportunity stages from deal data",
            "Sync contacts between Salesforce and a spreadsheet",
        ),
    ),
    # ------------------------------------------------------------------
    # Databases
    # ------------------------------------------------------------------
    "n8n-nodes-base.postgres": NodeCatalogEntry(
        type_name="n8n-nodes-base.postgres",
        display_name="Postgres",
        description=(
            "Executes SQL queries against a PostgreSQL database. Supports "
            "SELECT, INSERT, UPDATE, DELETE, and stored procedure calls. "
            "Requires PostgreSQL connection credentials."
        ),
        category="Database",
        type_version=2,
        credentials_required=("postgres",),
        common_parameters=(
            NodeParameter(name="operation", description="Operation (executeQuery, insert, update, delete, select)", default="select"),
            NodeParameter(name="query", description="SQL query to execute", param_type="string"),
            NodeParameter(name="table", description="Target table name"),
        ),
        use_cases=(
            "Read records from a Postgres database",
            "Insert processed data into a table",
            "Update records based on workflow logic",
        ),
    ),
    "n8n-nodes-base.mysql": NodeCatalogEntry(
        type_name="n8n-nodes-base.mysql",
        display_name="MySQL",
        description=(
            "Executes SQL queries against a MySQL or MariaDB database. "
            "Requires MySQL connection credentials."
        ),
        category="Database",
        type_version=2,
        credentials_required=("mySql",),
        common_parameters=(
            NodeParameter(name="operation", description="Operation (executeQuery, insert, update, delete, select)", default="select"),
            NodeParameter(name="query", description="SQL query to execute", param_type="string"),
        ),
        use_cases=(
            "Read data from a MySQL database",
            "Log workflow results to a table",
        ),
    ),
    "n8n-nodes-base.redis": NodeCatalogEntry(
        type_name="n8n-nodes-base.redis",
        display_name="Redis",
        description=(
            "Interacts with a Redis key-value store. Supports GET, SET, DELETE, "
            "INCR, EXPIRE, and pub/sub operations. Useful for caching, rate "
            "limiting, or shared state between workflow executions."
        ),
        category="Database",
        type_version=1,
        credentials_required=("redis",),
        common_parameters=(
            NodeParameter(name="operation", description="Redis command (get, set, delete, incr, etc.)", default="get"),
            NodeParameter(name="key", description="Redis key", required=True),
        ),
        use_cases=(
            "Cache API responses",
            "Implement rate limiting",
            "Track counters across executions",
        ),
    ),
    # ------------------------------------------------------------------
    # AI / LLM
    # ------------------------------------------------------------------
    "@n8n/n8n-nodes-langchain.lmChatOpenAi": NodeCatalogEntry(
        type_name="@n8n/n8n-nodes-langchain.lmChatOpenAi",
        display_name="OpenAI Chat Model",
        description=(
            "Connects to OpenAI's chat completion API (GPT-4o, GPT-4, GPT-3.5-turbo). "
            "Used as the language model sub-node in AI agent and chain workflows. "
            "Requires an OpenAI API key credential."
        ),
        category="AI",
        type_version=1,
        credentials_required=("openAiApi",),
        common_parameters=(
            NodeParameter(name="model", description="OpenAI model to use", default="gpt-4o-mini"),
            NodeParameter(name="temperature", description="Sampling temperature (0-2)", default=0.7, param_type="number"),
        ),
        use_cases=(
            "Power an AI agent with GPT",
            "Summarise text using GPT",
            "Classify or extract information from text",
        ),
    ),
    "@n8n/n8n-nodes-langchain.agent": NodeCatalogEntry(
        type_name="@n8n/n8n-nodes-langchain.agent",
        display_name="AI Agent",
        description=(
            "An autonomous AI agent that uses a language model and tools to "
            "complete tasks described in natural language. The agent can call "
            "tools iteratively until it reaches a final answer. Pair with "
            "an OpenAI Chat Model and tool sub-nodes."
        ),
        category="AI",
        type_version=1,
        common_parameters=(
            NodeParameter(name="text", description="Task description or user message", required=True),
            NodeParameter(name="systemMessage", description="System prompt for the agent"),
        ),
        use_cases=(
            "Build a customer support chatbot",
            "Automate research tasks",
            "Process natural language instructions",
        ),
    ),
    "@n8n/n8n-nodes-langchain.chainSummarization": NodeCatalogEntry(
        type_name="@n8n/n8n-nodes-langchain.chainSummarization",
        display_name="Summarization Chain",
        description=(
            "Summarizes long documents or large amounts of text using an LLM. "
            "Supports map-reduce and refine strategies for handling text that "
            "exceeds the model's context window."
        ),
        category="AI",
        type_version=2,
        common_parameters=(
            NodeParameter(name="mode", description="Summarization mode (map_reduce or refine)", default="map_reduce"),
        ),
        use_cases=(
            "Summarise article collections",
            "Condense long reports",
            "Generate executive summaries",
        ),
    ),
    # ------------------------------------------------------------------
    # DevOps / Developer Tools
    # ------------------------------------------------------------------
    "n8n-nodes-base.github": NodeCatalogEntry(
        type_name="n8n-nodes-base.github",
        display_name="GitHub",
        description=(
            "Interacts with GitHub repositories, issues, pull requests, releases, "
            "and more via the GitHub REST API. Requires a GitHub Personal Access "
            "Token or OAuth2 credential."
        ),
        category="Developer Tools",
        type_version=1,
        credentials_required=("githubApi",),
        common_parameters=(
            NodeParameter(name="resource", description="Resource (issue, pullRequest, release, file, etc.)", default="issue"),
            NodeParameter(name="operation", description="Operation to perform", default="create"),
            NodeParameter(name="owner", description="Repository owner username", required=True),
            NodeParameter(name="repository", description="Repository name", required=True),
        ),
        use_cases=(
            "Create GitHub issues from alerts",
            "List open pull requests",
            "Automate release notes",
        ),
    ),
    "n8n-nodes-base.jira": NodeCatalogEntry(
        type_name="n8n-nodes-base.jira",
        display_name="Jira Software",
        description=(
            "Creates and manages Jira issues, projects, and sprints. "
            "Requires Jira API credentials (cloud or server)."
        ),
        category="Developer Tools",
        type_version=1,
        credentials_required=("jiraSoftwareCloudApi",),
        common_parameters=(
            NodeParameter(name="resource", description="Resource (issue, project, user)", default="issue"),
            NodeParameter(name="operation", description="Operation (create, get, update, etc.)", default="create"),
            NodeParameter(name="projectKey", description="Jira project key", required=True),
        ),
        use_cases=(
            "Create Jira tickets from support emails",
            "Update issue status from a webhook",
            "Sync GitHub issues to Jira",
        ),
    ),
    # ------------------------------------------------------------------
    # E-commerce
    # ------------------------------------------------------------------
    "n8n-nodes-base.shopify": NodeCatalogEntry(
        type_name="n8n-nodes-base.shopify",
        display_name="Shopify",
        description=(
            "Reads and manages Shopify store data including orders, products, "
            "customers, and inventory. Requires Shopify API credentials."
        ),
        category="E-commerce",
        type_version=1,
        credentials_required=("shopifyApi",),
        common_parameters=(
            NodeParameter(name="resource", description="Resource (order, product, customer)", default="order"),
            NodeParameter(name="operation", description="Operation (get, getAll, create, update)", default="getAll"),
        ),
        use_cases=(
            "Sync new Shopify orders to a spreadsheet",
            "Notify team of new orders via Slack",
            "Update inventory levels",
        ),
    ),
    # ------------------------------------------------------------------
    # Data Transformation
    # ------------------------------------------------------------------
    "n8n-nodes-base.xml": NodeCatalogEntry(
        type_name="n8n-nodes-base.xml",
        display_name="XML",
        description=(
            "Converts between XML and JSON formats. Useful for processing "
            "SOAP API responses or RSS/Atom feeds."
        ),
        category="Data Transformation",
        type_version=1,
        common_parameters=(
            NodeParameter(name="mode", description="Conversion mode (xmlToJson or jsonToXml)", default="xmlToJson"),
            NodeParameter(name="dataPropertyName", description="Field containing the XML/JSON data", default="data"),
        ),
        use_cases=(
            "Parse RSS feed XML",
            "Convert SOAP responses to JSON",
        ),
    ),
    "n8n-nodes-base.rssFeedRead": NodeCatalogEntry(
        type_name="n8n-nodes-base.rssFeedRead",
        display_name="RSS Feed Read",
        description=(
            "Fetches and parses an RSS or Atom feed, returning one item per "
            "feed entry. Ideal for monitoring news sources, blogs, or any "
            "service that publishes an RSS feed."
        ),
        category="Data Transformation",
        type_version=1,
        common_parameters=(
            NodeParameter(name="url", description="URL of the RSS/Atom feed", required=True),
        ),
        use_cases=(
            "Monitor Hacker News RSS for new posts",
            "Track competitor blog updates",
            "Aggregate news from multiple sources",
        ),
    ),
    "n8n-nodes-base.htmlExtract": NodeCatalogEntry(
        type_name="n8n-nodes-base.htmlExtract",
        display_name="HTML Extract",
        description=(
            "Extracts data from HTML using CSS selectors. Use after an HTTP "
            "Request node to scrape data from web pages."
        ),
        category="Data Transformation",
        type_version=1,
        common_parameters=(
            NodeParameter(name="dataPropertyName", description="Field containing HTML content", default="data"),
            NodeParameter(name="extractionValues", description="CSS selector extraction definitions", param_type="array"),
        ),
        use_cases=(
            "Scrape product prices from a website",
            "Extract article text from HTML pages",
        ),
    ),
    "n8n-nodes-base.markdown": NodeCatalogEntry(
        type_name="n8n-nodes-base.markdown",
        display_name="Markdown",
        description=(
            "Converts between Markdown and HTML. Useful for formatting "
            "content before sending emails or posting to services that "
            "accept HTML or Markdown."
        ),
        category="Data Transformation",
        type_version=1,
        common_parameters=(
            NodeParameter(name="mode", description="Conversion mode (markdownToHtml or htmlToMarkdown)", default="markdownToHtml"),
            NodeParameter(name="markdown", description="Markdown content to convert"),
        ),
        use_cases=(
            "Convert LLM output to HTML for emails",
            "Format content for a CMS",
        ),
    ),
}


# ---------------------------------------------------------------------------
# Lookup helpers
# ---------------------------------------------------------------------------


def get_node_by_type(type_name: str) -> NodeCatalogEntry | None:
    """Look up a node catalog entry by its exact type identifier.

    Args:
        type_name: The full n8n node type string,
            e.g. ``"n8n-nodes-base.httpRequest"``.

    Returns:
        The matching :class:`NodeCatalogEntry`, or ``None`` if not found.
    """
    return NODE_CATALOG.get(type_name)


def search_nodes(query: str) -> list[NodeCatalogEntry]:
    """Search the catalog for nodes matching a query string.

    The search is case-insensitive and matches against the node's
    display name, description, category, type name, and use cases.

    Args:
        query: Free-text search query.

    Returns:
        List of :class:`NodeCatalogEntry` objects that match the query,
        ordered by relevance (display name match first, then description).
    """
    q = query.lower().strip()
    if not q:
        return list(NODE_CATALOG.values())

    name_matches: list[NodeCatalogEntry] = []
    other_matches: list[NodeCatalogEntry] = []

    for entry in NODE_CATALOG.values():
        in_name = q in entry.display_name.lower() or q in entry.type_name.lower()
        in_desc = q in entry.description.lower()
        in_category = q in entry.category.lower()
        in_use_cases = any(q in uc.lower() for uc in entry.use_cases)

        if in_name:
            name_matches.append(entry)
        elif in_desc or in_category or in_use_cases:
            other_matches.append(entry)

    return name_matches + other_matches


def get_nodes_by_category(category: str) -> list[NodeCatalogEntry]:
    """Return all nodes in a given category.

    Args:
        category: Category name to filter by (case-insensitive),
            e.g. ``"Trigger"``, ``"Communication"``.

    Returns:
        List of :class:`NodeCatalogEntry` objects in that category.
    """
    cat = category.lower().strip()
    return [
        entry for entry in NODE_CATALOG.values() if entry.category.lower() == cat
    ]


def get_trigger_nodes() -> list[NodeCatalogEntry]:
    """Return all trigger nodes from the catalog.

    Returns:
        List of :class:`NodeCatalogEntry` instances where ``is_trigger`` is True.
    """
    return [entry for entry in NODE_CATALOG.values() if entry.is_trigger]


def get_all_categories() -> list[str]:
    """Return a deduplicated, sorted list of all categories in the catalog.

    Returns:
        Sorted list of unique category name strings.
    """
    return sorted({entry.category for entry in NODE_CATALOG.values()})


def catalog_to_prompt_text(
    entries: list[NodeCatalogEntry] | None = None,
    max_nodes: int = 30,
) -> str:
    """Render catalog entries as formatted text for LLM prompt injection.

    Args:
        entries: Specific entries to render. If ``None``, all catalog entries
            are used.
        max_nodes: Maximum number of nodes to include in the output to avoid
            exceeding prompt token limits.

    Returns:
        A multi-line string with one section per catalog entry, separated
        by blank lines.
    """
    source = entries if entries is not None else list(NODE_CATALOG.values())
    selected = source[:max_nodes]
    return "\n\n".join(entry.to_prompt_text() for entry in selected)
