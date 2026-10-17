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

**3. Point Claude Code at the bridge**

macOS / Linux (bash or zsh):

```bash
export ANTHROPIC_BASE_URL="http://127.0.0.1:8787"
export ANTHROPIC_AUTH_TOKEN="local"      # or your LLMBRIDGE_API_KEY
claude --model nemotron-local             # use a configured alias
```

Windows (PowerShell):

```powershell
$env:ANTHROPIC_BASE_URL  = "http://127.0.0.1:8787"
$env:ANTHROPIC_AUTH_TOKEN = "local"
claude --model nemotron-local
```

To persist on Windows across new terminals: `setx ANTHROPIC_BASE_URL "http://127.0.0.1:8787"` (and the same for `ANTHROPIC_AUTH_TOKEN`).

---

## Configuration

`llmbridge.toml` (searched: `--config`, `./llmbridge.toml`, `~/.config/llmbridge/llmbridge.toml`):

```toml
[models.nemotron-local]
base_url = "http://127.0.0.1:11434/v1"   # Ollama
model = "nemotron"
reasoning_effort = "high"                 # enables thinking

[models.claude-sonnet-4-5]
base_url = "https://integrate.api.nvidia.com/v1"
model = "nvidia/nemotron-3-ultra-550b-a55b"
api_key_env = "NVIDIA_API_KEY"
reasoning_effort = "high"
fallbacks = ["claude-haiku-4-5"]
```

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