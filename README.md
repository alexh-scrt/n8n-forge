# n8n Forge

**n8n Forge** converts plain-English automation descriptions into ready-to-import [n8n](https://n8n.io) workflow JSON files using OpenAI GPT.

Describe your automation in plain English, and n8n Forge generates a fully-structured n8n workflow with appropriate nodes, connections, and scheduling — no technical n8n knowledge required.

---

## Features

- **Natural language → n8n workflow** via a single CLI command
- **Grounded prompt engineering** using a curated n8n node catalog for accurate node types and connections
- **Automatic JSON extraction & validation** ensures output is always importable
- **Rich terminal summary** showing generated nodes, triggers, and connections
- **Iterative refinement** via `--refine` flag for follow-up adjustments

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

Or pass it directly using the `--api-key` flag.

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
n8n-forge generate "Send daily Slack digests" --output workflow.json
n8n-forge refine workflow.json "Also send the digest to email subscribers"
```

### Full options

```bash
n8n-forge generate --help
```

```
Usage: n8n-forge generate [OPTIONS] DESCRIPTION

  Generate an n8n workflow from a plain-English description.

Options:
  -o, --output PATH       Save the workflow JSON to this file.
  -m, --model TEXT        OpenAI model to use. [default: gpt-4o-mini]
  --api-key TEXT          OpenAI API key (overrides OPENAI_API_KEY env var).
  --temperature FLOAT     Sampling temperature for the LLM. [default: 0.2]
  --help                  Show this message and exit.
```

---

## Example

**Input:**
```bash
n8n-forge generate "Every Monday at 9am, fetch the top 5 posts from Hacker News and post a summary to Slack" --output hn_digest.json
```

**Terminal output:**
```
✨ n8n Forge — Workflow Generated

📋 Workflow: Hacker News to Slack Digest

┌─────────────────────────────────────────────────────────┐
│ Nodes (4)                                               │
├──────────────────────┬──────────────────────────────────┤
│ Name                 │ Type                             │
├──────────────────────┼──────────────────────────────────┤
│ Schedule Trigger     │ n8n-nodes-base.scheduleTrigger   │
│ HTTP Request         │ n8n-nodes-base.httpRequest       │
│ Code                 │ n8n-nodes-base.code              │
│ Slack                │ n8n-nodes-base.slack             │
└──────────────────────┴──────────────────────────────────┘

🔗 Connections: 3
💾 Saved to: hn_digest.json
```

**Generated `hn_digest.json`** is a valid n8n workflow you can import directly via:
- n8n UI → **Workflows** → **Import from file**

---

## How It Works

1. **Prompt building** — Your description is combined with n8n node catalog context into a structured system + user prompt.
2. **LLM call** — The prompt is sent to OpenAI GPT, which returns a workflow JSON.
3. **Parsing & validation** — The JSON is extracted from the response, cleaned, and validated against n8n's workflow schema.
4. **Output** — A rich summary is displayed and the workflow is optionally saved to a file.

---

## Development

```bash
pip install -e ".[dev]"
pytest
```

---

## License

MIT License — see [LICENSE](LICENSE) for details.
