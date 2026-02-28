# n8n Forge 🔨

**Turn plain English into ready-to-import n8n workflows — no JSON wrangling required.**

n8n Forge is a CLI tool that converts natural language automation descriptions into valid, importable [n8n](https://n8n.io) workflow JSON files using OpenAI GPT. Just describe what you want to automate, and n8n Forge generates a fully-structured workflow with the right nodes, connections, and scheduling. No need to understand n8n's internal data model.

---

## Quick Start

**Install:**

```bash
pip install n8n-forge
```

**Set your OpenAI API key:**

```bash
export OPENAI_API_KEY=sk-...
```

**Generate your first workflow:**

```bash
n8n-forge generate "Check competitor prices weekly and add results to Google Sheets" --output competitor_prices.json
```

Then import `competitor_prices.json` into n8n via **Workflows → Import from file**. Done.

---

## Features

- **Natural language → n8n workflow** — describe your automation in plain English and get a complete, importable workflow JSON in seconds
- **Grounded prompt engineering** — a curated n8n node catalog keeps the LLM accurate on real node types, parameters, and connection patterns
- **Automatic JSON extraction & validation** — the output is always structurally valid and safe to import, with Pydantic-backed schema checks
- **Rich terminal summary** — see a formatted breakdown of generated nodes, triggers, and connections before committing to a file
- **Iterative refinement** — use the `refine` subcommand to tweak an existing workflow with a follow-up description

---

## Usage Examples

### Generate a new workflow

```bash
# Print to terminal only
n8n-forge generate "Send a Slack message every Monday at 9am with a team standup reminder"

# Save to a file
n8n-forge generate "When a new row is added to Google Sheets, send a welcome email via Gmail" \
  --output onboarding_email.json

# Use a specific model
n8n-forge generate "Scrape HackerNews top stories and post to Slack daily" \
  --model gpt-4o \
  --output hackernews_digest.json
```

**Example terminal output:**

```
✔ Workflow generated: HackerNews Daily Digest

┌─────────────────────────────────────────────────────────┐
│ Nodes (4)                                               │
├──────────────────────┬──────────────────────────────────┤
│ Schedule Trigger     │ n8n-nodes-base.scheduleTrigger   │
│ HTTP Request         │ n8n-nodes-base.httpRequest        │
│ Code                 │ n8n-nodes-base.code              │
│ Slack                │ n8n-nodes-base.slack             │
└──────────────────────┴──────────────────────────────────┘

Connections: Schedule Trigger → HTTP Request → Code → Slack
Saved to: hackernews_digest.json
```

### Refine an existing workflow

```bash
# Load a generated workflow and adjust it with a follow-up description
n8n-forge refine hackernews_digest.json \
  "Also filter out posts with fewer than 100 upvotes before posting to Slack" \
  --output hackernews_digest_v2.json
```

### CLI reference

```
Usage: n8n-forge [OPTIONS] COMMAND [ARGS]...

Commands:
  generate  Convert a plain-English description into an n8n workflow JSON.
  refine    Modify an existing workflow JSON based on new instructions.

Options for `generate`:
  DESCRIPTION       Plain-English automation description  [required]
  --output, -o      Path to write the workflow JSON file
  --model, -m       OpenAI model to use  [default: gpt-4o-mini]
  --temperature, -t Sampling temperature  [default: 0.2]

Options for `refine`:
  WORKFLOW_FILE     Path to an existing workflow JSON file  [required]
  INSTRUCTIONS      Follow-up instructions for modifying the workflow  [required]
  --output, -o      Path to write the updated workflow JSON file
  --model, -m       OpenAI model to use  [default: gpt-4o-mini]
```

---

## Project Structure

```
n8n-forge/
├── pyproject.toml                  # Project metadata, dependencies, CLI entry point
├── README.md
├── n8n_forge/
│   ├── __init__.py                 # Package version
│   ├── cli.py                      # Click CLI: generate & refine subcommands
│   ├── generator.py                # OpenAI API communication
│   ├── parser.py                   # JSON extraction, cleaning, and validation
│   ├── prompt_builder.py           # System & user prompt construction
│   ├── node_catalog.py             # Curated n8n node type catalog
│   ├── schema.py                   # Pydantic models for n8n workflow structure
│   └── templates/
│       └── system_prompt.j2        # Jinja2 system prompt template
└── tests/
    ├── test_parser.py
    ├── test_prompt_builder.py
    ├── test_generator.py
    ├── test_schema.py
    └── test_node_catalog.py
```

---

## Configuration

n8n Forge is configured via environment variables and CLI flags.

### Environment Variables

| Variable | Required | Description |
|---|---|---|
| `OPENAI_API_KEY` | ✅ Yes | Your OpenAI API key |
| `OPENAI_BASE_URL` | No | Custom API base URL (e.g. for Azure OpenAI or local proxies) |

### CLI Defaults

| Option | Default | Description |
|---|---|---|
| `--model` | `gpt-4o-mini` | OpenAI model used for generation |
| `--temperature` | `0.2` | Sampling temperature (lower = more deterministic) |
| `--output` | *(stdout)* | File path for the generated workflow JSON |

**Tip:** For the most accurate workflows, use `gpt-4o` or `gpt-4o-mini`. GPT-3.5-class models may generate structurally invalid node configurations.

---

## Requirements

- Python 3.10+
- An [OpenAI API key](https://platform.openai.com/api-keys)
- An [n8n](https://n8n.io) instance (self-hosted or cloud) to import the generated workflows

---

## Running Tests

```bash
pip install -e '.[dev]'
pytest tests/
```

All OpenAI calls are mocked — no API key is needed to run the test suite.

---

## License

MIT — see [LICENSE](LICENSE) for details.

---

*Built with [Jitter](https://github.com/jitter-ai) - an AI agent that ships code daily.*
