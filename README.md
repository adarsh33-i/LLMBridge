# LLMBridge

A local protocol bridge that routes **Anthropic Messages API** requests to **OpenAI-compatible** model endpoints (Ollama, vLLM, NVIDIA, OpenRouter, etc.). Define model aliases, then use them in Claude Code.

---

## Quick Start

**1. Install** with [uv](https://docs.astral.sh/uv/) — it bootstraps Python and keeps the tool isolated:

macOS / Linux:

```bash
brew install uv    # or: curl -LsSf https://astral.sh/uv/install.sh | sh
uv tool install git+https://github.com/adarsh33-i/LLMBridge
```

Windows (PowerShell):

```powershell
winget install Git.Git        # installing from GitHub needs git
winget install astral-sh.uv
uv tool install git+https://github.com/adarsh33-i/LLMBridge
```

Upgrade any time with `uv tool upgrade llmbridge`. Working from a clone instead? `pip install -e .` works the same on both platforms (Python 3.11+).

**2. Configure and start:**

```bash
llmbridge setup        # guided config (Ollama, OpenAI, OpenRouter, NVIDIA presets)
llmbridge doctor       # verify config, keys, port
llmbridge serve        # start server on http://127.0.0.1:8787
```

The wizard asks these questions:

| Prompt | What to enter |
|--------|---------------|
| Provider preset | `nvidia`, `ollama`, `openai` or `openrouter` |
| Model alias | A short name **you invent** — this is what clients request. No dots, no slashes (see [Troubleshooting](#troubleshooting)) |
| Upstream base URL | Press Enter to accept the preset default |
| Upstream model name | The provider's **real model ID**, e.g. `nvidia/nemotron-3.5-super-vl-120b-a12b` |
| API key environment variable | Only asked for presets without a built-in key (Ollama). Blank = no key. NVIDIA/OpenAI/OpenRouter use `NVIDIA_API_KEY`/`OPENAI_API_KEY`/`OPENROUTER_API_KEY` automatically |

Prefer no prompts? Pass it all in one line:
`llmbridge setup --preset nvidia --alias claude-sonnet-4-5 --model nvidia/nemotron-3.5-super-vl-120b-a12b` (add `--force` to replace an existing file).

If Claude Code should use a route automatically, name the alias exactly what Claude Code requests: `claude-sonnet-4-5`, `claude-opus-4-5`, or `claude-haiku-4-5`. Any other name must be passed explicitly with `claude --model <alias>`.

**3. Point Claude Code at the bridge**

macOS / Linux (bash or zsh):

```bash
export ANTHROPIC_BASE_URL="http://127.0.0.1:8787"
export ANTHROPIC_AUTH_TOKEN="local"      # or your LLMBRIDGE_API_KEY
claude --model claude-sonnet-4-5         # any alias defined in your config
```

Windows (PowerShell):

```powershell
$env:ANTHROPIC_BASE_URL  = "http://127.0.0.1:8787"
$env:ANTHROPIC_AUTH_TOKEN = "local"
claude --model claude-sonnet-4-5
```

To persist on Windows across new terminals: `setx ANTHROPIC_BASE_URL "http://127.0.0.1:8787"` (and the same for `ANTHROPIC_AUTH_TOKEN`).

**Claude Code desktop app** doesn't read shell exports either. In the desktop app: Help → Troubleshooting → Enable Developer Mode, restart the app, then go to Developer → Developer Settings → Configure Third-Party Inference and set:

- Base URL: `http://127.0.0.1:8787`
- Token: `local` (or your `LLMBRIDGE_API_KEY`)

**VS Code extension / CLI** (without shell exports): add the same values to `~/.claude/settings.json` instead:

```json
{
  "env": {
    "ANTHROPIC_BASE_URL": "http://127.0.0.1:8787",
    "ANTHROPIC_AUTH_TOKEN": "local"
  }
}
```

For a per-project setup, put the same `"env"` block in that project's `.claude/settings.json`.

Or skip the exports entirely: `llmbridge claude` launches Claude Code wired to the bridge (add `--print-env` to just see the values).

---

## Configuration

`llmbridge.toml` (searched in order: `--config` flag, `$LLMBRIDGE_CONFIG`, `./llmbridge.toml`, `~/.config/llmbridge/llmbridge.toml` — honouring `$XDG_CONFIG_HOME`):

```toml
# What `llmbridge setup` writes (nvidia preset), with a local safety net added.
# One [models.<alias>] block per route.

[models.claude-sonnet-4-5]                 # Claude Code requests this exact name
base_url = "https://integrate.api.nvidia.com/v1"
model = "nvidia/nemotron-3.5-super-vl-120b-a12b"   # real upstream model ID
api_key_env = "NVIDIA_API_KEY"
reasoning_effort = "high"
fallbacks = ["claude-haiku-4-5", "claude-local"]   # tried in order on failure

[models.claude-haiku-4-5]
base_url = "https://integrate.api.nvidia.com/v1"
model = "nvidia/nemotron-3.5-lightning-30b-a3b"
api_key_env = "NVIDIA_API_KEY"
reasoning_effort = "high"
fallbacks = ["claude-local"]

[models.claude-local]                      # no key needed — keeps working when
base_url = "http://127.0.0.1:11434/v1"     # NVIDIA is down or the key dies
model = "qwen3:8b"                         # any model you have pulled in Ollama

[models."nemotron-3.5-super"]              # dots in an alias need a quoted key
base_url = "https://integrate.api.nvidia.com/v1"
model = "nvidia/nemotron-3.5-super-vl-120b-a12b"
api_key_env = "NVIDIA_API_KEY"
fallbacks = ["claude-local"]
```

Use these aliases in Claude Code: `claude --model nemotron-3.5-super`, or just let
Claude Code's Sonnet/Opus/Haiku picks land on the matching `claude-*` blocks above.

**Key fields per route:**
| Field | Purpose |
|-------|---------|
| `base_url` | Upstream OpenAI-compatible endpoint |
| `model` | Upstream model ID (optional; defaults to alias) |
| `api_key_env` | Env var holding the API key |
| `reasoning_effort` | `low` \| `medium` \| `high` — enables thinking |
| `supports_images` / `supports_tools` | Reject unsupported capabilities early |
| `fallbacks` | Aliases to try on failure |
| `cache_control` | `"strip"` (default) or `"passthrough"` |

Legacy single-endpoint vars (`LLMBRIDGE_UPSTREAM_*`) still supported.

**How the config file is picked — three rules that explain most confusion:**

1. The file depends on **where you start `llmbridge serve`**, not where you ran `setup`. A `./llmbridge.toml` in the serve folder wins over `~/.config/llmbridge/llmbridge.toml`. Run `llmbridge models` from the same folder you serve from to see the live route set. `setup` follows the same order when choosing where to **write**: an existing `./llmbridge.toml` wins, otherwise it writes to the user-level file.
2. The config is read **once at startup**. After every edit, restart `llmbridge serve` — a running server never sees your changes.
3. `fallbacks` may only list aliases **defined in the same file**. A fallback to a missing alias stops the server from starting.

A model alias with dots needs a quoted key, because TOML reads dots as nesting:

```toml
[models."nemotron-3.5-super"]   # correct
# [models.nemotron-3.5-super]  # wrong — parses as a route named "nemotron-3"
```

**Environment variables that tune the bridge** (all optional):

| Variable | Default | Purpose |
|----------|---------|---------|
| `LLMBRIDGE_HOST` | `127.0.0.1` | Bind address. Non-loopback requires `LLMBRIDGE_API_KEY` |
| `LLMBRIDGE_PORT` | `8787` | Listen port (also settable with `--port`) |
| `LLMBRIDGE_CONFIG` | unset | Config file path; overrides the search order above |
| `LLMBRIDGE_BASE_URL` | `http://127.0.0.1:8787` | Which bridge URL the read-only CLI commands (`check`, `generations`, `status`, `fork`, …) talk to |
| `LLMBRIDGE_CLIENT_TIMEOUT_SECONDS` | `300` | How long one client request may stream |
| `LLMBRIDGE_UPSTREAM_TIMEOUT_SECONDS` | `180` | Timeout for each upstream call |
| `LLMBRIDGE_ROUTE_ATTEMPTS` | `1` | Tries on one route before falling through to the next fallback (max 10) |
| `LLMBRIDGE_ROUTE_RETRY_BACKOFF` | `0.5` | Base seconds between attempts; doubles each try |
| `LLMBRIDGE_DOCUMENT_ROOTS` | unset | Directories allowed as local `file:` document sources (`os.pathsep`-separated). Unset disables local file documents |
| `LLMBRIDGE_ANTHROPIC_FILES_API_KEY` | unset | Enables document sources that reference the Anthropic Files API |

Feature-specific vars (`LLMBRIDGE_SELF_HEAL`, `LLMBRIDGE_QUEUE_*`, `LLMBRIDGE_TRACK_*`, `LLMBRIDGE_LEDGER_*`, `LLMBRIDGE_ENABLE_OCR`) are documented with their features below.

---

## Troubleshooting

Real errors and what they mean:

| Error | Cause | Fix |
|-------|-------|-----|
| `Unknown model 'claude-opus-4-5'. Configured model aliases: ...` | Claude Code requests exact model names; the live config has no alias with that name | Add that alias to the config file the server reads, restart serve. The alias list in the error tells you exactly what is live |
| `Route 'x' falls back to unknown model 'y'` (server refuses to start) | `fallbacks` references an alias not defined in the same file | Define the fallback alias in the same file, or remove it from `fallbacks` |
| `Route 'nemotron-3' needs a base_url` | Alias contains dots — TOML split the table name | Quote the key: `[models."nemotron-3.5-super"]` |
| `error: ...llmbridge.toml already exists (use --force to overwrite)` | `setup` refuses to clobber an existing file | Re-run `llmbridge setup --force` — it **replaces the whole file**, so re-add any other routes afterwards |
| `Upstream returned HTTP 403` → client sees `502`, work saved as `gen_...` | The provider rejected your API key (expired or revoked). The bridge behaved correctly: retried, saved the interrupted work, told you | Confirm with `curl` against the provider using the same key; generate a fresh key; add a local Ollama route to `fallbacks` so requests survive a dead key |
| Request fails with `404` from Ollama | The `model` ID is not pulled locally | `ollama list` to check, `ollama pull <model>` to fix |

---

## CLI Commands

| Command | Purpose |
|---------|---------|
| `llmbridge` / `serve` | Start server |
| `llmbridge setup` | Guided config writer |
| `llmbridge doctor` | Diagnose env, config, keys, port |
| `llmbridge check` | Validate config |
| `llmbridge models` | List aliases, endpoints, fallbacks |
| `llmbridge claude` | Launch Claude Code against bridge |
| `llmbridge init` | Write starter config |
| `llmbridge generations` | List recorded generations |
| `llmbridge status` | Summarize ledger, flag recoverable work |
| `llmbridge fork <id> --model <alias>` | Continue a generation on another model |
| `llmbridge tree <id>` | Render a fork lineage (ancestors + descendants) |
| `llmbridge diff <idA> <idB>` | Compare two generations' outputs side by side |
| `llmbridge replay <id>\|--all` | Re-send recorded requests; flag output drift |

---

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Readiness check |
| `GET` | `/v1/models` | Configured aliases |
| `POST` | `/v1/messages` | Anthropic-compatible completion |
| `POST` | `/v1/messages/count_tokens` | Approximate input-token estimate |
| `GET` | `/v1/interrupted` | Saved interrupted work (404 if none) |
| `POST` | `/v1/interrupted/resume` | Resume interrupted work on a model |
| `DELETE` | `/v1/interrupted` | Clear saved interrupted work |
| `GET` | `/v1/generations` | List durable generations |
| `GET` | `/v1/generations/{id}` | Full generation record |
| `POST` | `/v1/generations/{id}/fork` | Fork/continue generation on another model |
| `GET` | `/v1/generations/{id}/tree` | Fork lineage: ancestors + nested subtree |
| `DELETE` | `/v1/generations/{id}` | Remove generation from ledger |
| `GET` | `/v1/queue` | List generations parked on quota (queue mode) |
| `DELETE` | `/v1/queue/{id}` | Cancel a queued generation |

Every `/v1/messages` response includes `X-LLMBridge-Generation` header with the generation ID.

---

## Durable Generations & Forking

Each request is recorded in an append-only ledger (`.llmbridge/generations/`):

- **Recorded**: prompt, system, tools, assistant text, event history
- **Fork**: Continue from any point on a **different model** — full conversation history rebuilt
- **Lineage**: Parent/child relationships preserved via `fingerprint`

```bash
# 1. Request interrupted at 40%
curl ... /v1/messages -d '{"model":"cheap","max_tokens":2000,"messages":[...]}'
# -> X-LLMBridge-Generation: gen_1a2b3c4d5e6f

# 2. Inspect what was produced
curl ... /v1/generations/gen_1a2b3c4d5e6f

# 3. Fork onto a stronger model
curl ... /v1/generations/gen_1a2b3c4d5e6f/fork -d '{"model":"strong","max_tokens":2000}'
# or: llmbridge fork gen_1a2b3c4d5e6f --model strong
```

Ledger: atomic writes, file-lock serialized, per-file `0600` permissions. Disable with `LLMBRIDGE_TRACK_GENERATIONS=0`. Only the newest records are kept — 10 by default (`MAX_TRACKED_INTERRUPTED` in `llmbridge/ledger.py`, override with `LLMBRIDGE_LEDGER_MAX_RECORDS`); older ones are pruned as new generations arrive.

### Lineage, diffs, and replay

Forks record their parent, so every generation knows where it grew from:

```bash
llmbridge tree gen_1a2b3c4d5e6f          # ASCII lineage: ancestors + children
curl ... /v1/generations/gen_1a2b3c4d5e6f/tree   # same, as JSON

llmbridge diff gen_1a2b3c4d5e6f gen_9f8e7d6c5b4a   # outputs side by side + similarity

# Regression suite: re-send recorded requests and compare against saved output
llmbridge replay gen_1a2b3c4d5e6f        # one generation
llmbridge replay --all --limit 20 --threshold 0.8   # newest 20 completed; exit 1 on drift
```

`replay` sends the exact recorded request back to the configured route and scores the new output against the saved one with `difflib` similarity. Use it after switching models, editing prompts, or upgrading upstreams to catch behavioral drift.

---

## Self-Healing Streams

When a streaming response breaks mid-generation (proxy hiccup, upstream crash, malformed SSE), the bridge does not give the client a truncated answer. If partial text was already sent **and** a fallback route is configured, it:

1. Opens the next fallback with the partial output appended as an assistant turn
2. Stitches the continuation into the **same SSE stream** — the client sees one unbroken response and one `message_start`
3. Records a `heal` event in the ledger (from/to model + how many characters were resumed)

The final generation record contains the full stitched output. Disable with:

```bash
LLMBRIDGE_SELF_HEAL=0
```

Tool-use blocks are not stitched — healing applies to text output only; without partial text or a fallback the normal interrupted-work flow applies.

---

## Quota-Aware Queue

Opt in to park requests on upstream quota/rate-limit errors instead of failing them:

```bash
LLMBRIDGE_QUEUE_ON_QUOTA=1        # opt-in (default off)
LLMBRIDGE_QUEUE_RETRY_SECONDS=5   # base delay when upstream sends no Retry-After
LLMBRIDGE_QUEUE_MAX_ATTEMPTS=5    # give up after this many dispatch attempts
```

On a `402`/`429` the bridge:

1. Parks the generation in the ledger (status `queued`, attempt + retry time recorded)
2. Replies `429` with `Retry-After`, a `generation_id`, and `"queued": true`
3. A background dispatcher retries the recorded request until the upstream recovers — surviving restarts, because the queue **is** the ledger
4. The client polls `GET /v1/generations/{id}` for the completed output, or cancels with `DELETE /v1/queue/{id}`

```bash
curl ... /v1/queue         # what is parked, next retry time, attempt count
curl -X DELETE ... /v1/queue/gen_1a2b3c4d5e6f   # cancel
```

---

## Partial Work Tracking (Auto)

The bridge automatically saves interrupted work when upstream fails (quota, rate limit, timeout, stream error). Interrupted work is not a separate file — it is stored as an `interrupted` generation record in the same ledger (`.llmbridge/generations/`), so it can be inspected and forked like any other generation:

- **Saved**: prompt, partial output, progress %, error context, replayable request
- **Auto-cleared**: when same prompt succeeds
- **Inspect**: `GET /v1/interrupted` (returns the newest interrupted record)
- **Resume**: `POST /v1/interrupted/resume` — sends partial output as assistant turn
- **Retention**: among the newest 10 ledger records; resuming creates a new generation record

```bash
# Config
LLMBRIDGE_TRACK_INTERRUPTED=1   # enabled by default
LLMBRIDGE_LEDGER_PATH=.llmbridge/generations
LLMBRIDGE_LEDGER_MAX_RECORDS=10 # records retained in the ledger
```

---

## Compatibility & Limitations

| Feature | Support |
|---------|---------|
| Messages | System, user/assistant text, images (URL/base64), documents |
| Documents | Text, base64, public HTTP(S) URLs, local files, Anthropic Files API |
| Tools | Definitions, choices, results, upstream tool calls → function calling |
| Streaming | OpenAI SSE → Anthropic events (`ping`, `input_tokens`, `content_filter`→`end_turn`) |
| Token counting | Route-configured tiktoken or char-based fallback (estimate only) |
| Thinking | Only on routes with `reasoning_effort`; `budget_tokens` not preserved |
| Caching | Off by default (`cache_control="strip"`); `"passthrough"` forwards breakpoints |

**Not supported**: Full Anthropic API parity, beta/newer features, preserved citations/layout/formatting in documents. OCR optional (PyMuPDF + Tesseract, 10 pages / 90s limit).

---

## Security Notes

- Server binds `127.0.0.1:8787` by default (no auth on loopback)
- Non-loopback bind requires `LLMBRIDGE_API_KEY` (≥32 chars)
- Use TLS reverse proxy for network access
- Ledger records contain plain-text prompts (`0600` perms)
- Anthropic Files API key grants file retrieval access to bridge clients

---

## Optional Extras

```bash
# OCR for scanned PDFs (Tesseract required)
pip install -e ".[ocr]"
export LLMBRIDGE_ENABLE_OCR=1

# Tokenizer-based input estimates
pip install -e ".[token-count]"
```

---

## Tests

```bash
python -m unittest discover -s tests -v
```

---

## License

Apache 2.0 — see [LICENSE](LICENSE).