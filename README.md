# n8n Forge

**n8n Forge** converts plain-English automation descriptions into ready-to-import [n8n](https://n8n.io) workflow JSON files using OpenAI GPT.

Describe your automation in plain English, and n8n Forge generates a fully-structured n8n workflow with appropriate nodes, connections, and scheduling — no technical n8n knowledge required.

---

## Features

- **Natural language → n8n workflow** via a single CLI command
- **Grounded prompt engineering** using a curated n8n node catalog for accurate node types and connections
- **Automatic JSON extraction & validation** ensures output is always importable
- **Rich terminal summary** showing generated nodes, triggers, and connections
- **Iterative refinement** via the `refine` subcommand for follow-up adjustments

---

## Requirements

- Python 3.10 or later
- An [OpenAI API key](https://platform.openai.com/api-keys)

---

## Installation

### From source

```bash
git clone https://github.com/example/n8n-forge.git
cd n8n-forge
pip install -e .
```

### From PyPI (when published)

```bash
pip install n8n-forge
```

---

## Setup

Set your OpenAI API key as an environment variable:

```bash
export OPENAI_API_KEY="sk-..."
```

Or pass it directly using the `--api-key` flag on any command.

---

## Usage

### Basic usage

```bash
n8n-forge generate "Check competitor prices weekly and add results to Google Sheets"
```

This prints the generated workflow JSON to stdout.

### Save to a file

```bash
n8n-forge generate "Send a Slack message every Monday morning with a sales summary" --output workflow.json
```

### Specify OpenAI model

```bash
n8n-forge generate "Parse incoming emails and create Trello cards" --model gpt-4o
```

### Refine an existing workflow

```bash
# First, generate a workflow and save it
n8n-forge generate "Send daily Slack digests" --output workflow.json

# Then refine it with additional instructions (overwrites the file in place)
n8n-forge refine workflow.json "Also send the digest to email subscribers"

# Or save the refined version to a new file
n8n-forge refine workflow.json "Add error handling" --output workflow_v2.json
```

### View all options

```bash
n8n-forge --help
n8n-forge generate --help
n8n-forge refine --help
```

---

## Command Reference

### `n8n-forge generate`

```
Usage: n8n-forge generate [OPTIONS] DESCRIPTION

  Generate an n8n workflow from a plain-English DESCRIPTION.

Options:
  -o, --output PATH       Save the workflow JSON to this file path.
  -m, --model MODEL       OpenAI model to use.  [default: gpt-4o-mini]
  --api-key KEY           OpenAI API key (overrides OPENAI_API_KEY env var).
  --temperature FLOAT     Sampling temperature (0.0–2.0).  [default: 0.2]
  --max-tokens INT        Maximum tokens to generate.  [default: 4096]
  --version               Show the version and exit.
  --help                  Show this message and exit.
```

### `n8n-forge refine`

```
Usage: n8n-forge refine [OPTIONS] WORKFLOW_FILE INSTRUCTIONS

  Refine an existing n8n workflow JSON file using INSTRUCTIONS.

Options:
  -o, --output PATH       Save the refined workflow to this file.
                          Defaults to overwriting WORKFLOW_FILE.
  -m, --model MODEL       OpenAI model to use.  [default: gpt-4o-mini]
  --api-key KEY           OpenAI API key (overrides OPENAI_API_KEY env var).
  --temperature FLOAT     Sampling temperature (0.0–2.0).  [default: 0.2]
  --max-tokens INT        Maximum tokens to generate.  [default: 4096]
  --help                  Show this message and exit.
```

---

## Examples

### Hacker News to Slack Digest

**Input:**
```bash
n8n-forge generate "Every Monday at 9am, fetch the top 5 posts from Hacker News and post a summary to Slack" --output hn_digest.json
```

**Terminal output:**
```
✨ n8n Forge  v0.1.0

  → Building prompt…
  → Generating workflow with gpt-4o-mini…
  → Parsing and validating workflow JSON…

✓ Workflow generated: Hacker News to Slack Digest

╭────────────────────────────────────────────────────────────╮
│ Nodes (4)                                                  │
├──────────────────────┬──────────────────────────┬─────────┤
│ Name                 │ Type                     │ Role    │
├──────────────────────┼──────────────────────────┼─────────┤
│ Schedule Trigger     │ n8n-nodes-base.schedule… │ 📡 Trig │
│ HTTP Request         │ n8n-nodes-base.httpRequ… │ 🔧 Acti │
│ Code                 │ n8n-nodes-base.code      │ 🔧 Acti │
│ Slack                │ n8n-nodes-base.slack     │ 🔧 Acti │
╰──────────────────────┴──────────────────────────┴─────────╯

🔗 Connections: 3
⚡ Trigger(s): Schedule Trigger

💾 Saved to: hn_digest.json
```

The generated `hn_digest.json` is a valid n8n workflow you can import directly.

---

### Competitor Price Tracker

```bash
n8n-forge generate "Check competitor prices every day at 8am by scraping their website, compare with our prices stored in Google Sheets, and send a Slack alert if any competitor is more than 10% cheaper" --output price_tracker.json
```

---

### Email-to-Jira Ticket

```bash
n8n-forge generate "When a support email arrives in the inbox, create a Jira ticket with the subject as the title and the body as the description, then reply to the sender with the ticket number"
```

---

### Refine an existing workflow

```bash
# Generate initial workflow
n8n-forge generate "Send a daily sales report to Slack" --output sales_report.json

# Add email delivery
n8n-forge refine sales_report.json "Also send the same report by email to sales@company.com"

# Add error handling
n8n-forge refine sales_report.json "Add error handling so the team is notified on Slack if anything fails" --output sales_report_v2.json
```

---

## How It Works

1. **Prompt building** — Your description is combined with a curated n8n node catalog into a structured system + user prompt. Relevant nodes are selected based on keywords in your description to keep the prompt focused and accurate.

2. **LLM call** — The prompt is sent to OpenAI GPT, which returns a workflow JSON wrapped in a fenced code block.

3. **Parsing & validation** — The JSON is extracted from the response, cleaned of common LLM artefacts (trailing commas, JS comments), normalised (missing IDs/positions filled in), and validated against n8n's workflow schema.

4. **Output** — A rich terminal summary is displayed showing all nodes, connections, and triggers. The workflow is optionally saved to a file.

---

## Supported n8n Nodes

n8n Forge's built-in node catalog includes the most commonly used n8n nodes:

| Category | Nodes |
|---|---|
| **Triggers** | Schedule Trigger, Webhook, Email Trigger (IMAP), Manual Trigger, n8n Form Trigger |
| **Core** | HTTP Request, Code, Edit Fields (Set), IF, Switch, Merge, Loop Over Items, Wait, No Operation |
| **Communication** | Slack, Send Email, Gmail, Microsoft Teams, Telegram |
| **Productivity** | Google Sheets, Airtable, Google Drive, Notion, Microsoft Excel 365 |
| **CRM** | HubSpot, Salesforce |
| **Databases** | Postgres, MySQL, Redis |
| **AI / LLM** | OpenAI Chat Model, AI Agent, Summarization Chain |
| **Developer Tools** | GitHub, Jira Software |
| **E-commerce** | Shopify |
| **Data Transformation** | XML, RSS Feed Read, HTML Extract, Markdown |

For integrations not in the catalog, n8n Forge will use the **HTTP Request** node to call the service's API directly.

---

## Importing into n8n

Once you have a generated workflow JSON file:

1. Open your n8n instance
2. Go to **Workflows** in the left sidebar
3. Click **Add workflow** → **Import from file**
4. Select your generated `.json` file
5. Click **Import**
6. Configure your credentials for any nodes that require them
7. Activate the workflow

---

## Tips for Better Results

- **Be specific about timing**: Instead of "regularly", say "every Monday at 9am" or "every 15 minutes".
- **Name the services**: Say "Google Sheets" not just "a spreadsheet", or "Slack" not just "messaging".
- **Describe the data flow**: Mention what data moves between steps, e.g. "fetch orders from Shopify and add each one as a row in Google Sheets".
- **Use a capable model**: For complex workflows with many nodes, try `--model gpt-4o` for better accuracy.
- **Refine iteratively**: Start simple and use the `refine` command to add steps one at a time.
- **Low temperature**: The default temperature of `0.2` produces more consistent JSON output. Avoid raising it above `0.5`.

---

## Development

### Setup

```bash
git clone https://github.com/example/n8n-forge.git
cd n8n-forge
pip install -e .
```

### Running tests

```bash
pip install pytest
pytest
```

All tests use mocked OpenAI responses — no API key is needed to run the test suite.

### Project structure

```
n8n_forge/
├── __init__.py          # Package version
├── cli.py               # Click CLI entry point (generate & refine commands)
├── generator.py         # OpenAI API communication
├── parser.py            # JSON extraction, cleaning, and schema validation
├── prompt_builder.py    # Prompt construction with Jinja2 templates
├── node_catalog.py      # Static catalog of n8n node types
├── schema.py            # Pydantic models for n8n workflow structure
└── templates/
    └── system_prompt.j2 # Jinja2 system prompt template
tests/
├── test_parser.py
├── test_prompt_builder.py
└── test_generator.py
```

---

## Configuration

| Environment Variable | Description |
|---|---|
| `OPENAI_API_KEY` | Your OpenAI API key. Required unless `--api-key` is passed. |

---

## Troubleshooting

**`AuthenticationError: OpenAI API key is required`**
> Set `export OPENAI_API_KEY="sk-..."` or pass `--api-key sk-...`.

**`JSONExtractionError: No JSON object found in the LLM response`**
> The model didn't return valid JSON. Try again, or use a more capable model with `--model gpt-4o`.

**`WorkflowValidationError: Workflow JSON failed schema validation`**
> The generated workflow had structural issues. Try again or use `--model gpt-4o` for better output quality.

**`RateLimitError: OpenAI rate limit exceeded`**
> Wait a moment and try again, or use a different API key.

**Generated workflow has wrong node types**
> Try being more specific in your description, e.g. name the exact service ("Google Sheets" instead of "spreadsheet"). You can also use the `refine` command to correct specific nodes.

---

## License

MIT License — see [LICENSE](LICENSE) for details.
