# LLMBridge

LLMBridge is a local protocol bridge for routing Claude Code requests to
multiple OpenAI-compatible model endpoints. Define an alias for each local or
hosted model, then use that alias in Claude Code. It accepts Anthropic Messages
API requests and translates them to OpenAI Chat Completions requests.

The bridge supports model aliases and per-model endpoints, text, image
URL/base64 content, plain-text documents, text-based PDFs, DOCX, XLSX, PPTX,
HTML and RTF documents, tool definitions, tool results, non-streaming replies,
and basic server-sent-event streaming.
It is an API compatibility layer; it does not modify Claude Desktop's built-in
model. Claude Desktop integrations would need a separate integration, such as
an MCP server.

## Run

Requires Python 3.11 or newer.

Install the project and its dependencies:

```sh
python -m pip install -e .
```

Copy `llmbridge.toml.example` to `llmbridge.toml`, edit the model aliases and
endpoints, and set any required upstream API keys in environment variables:

```sh
cp llmbridge.toml.example llmbridge.toml
export NVIDIA_API_KEY="your-provider-key"
python -m llmbridge
```

Each `[models.<alias>]` entry has a `base_url` and may specify an upstream
`model`, an `api_key_env` environment-variable name, and (for providers that
support it) a `reasoning_effort` of `low`, `medium`, or `high`. You can declare
`supports_images` or `supports_tools` as `true` or `false` to reject requests
using a capability known to be unsupported by the endpoint. Set
`tokenizer_encoding` to a supported tiktoken encoding (`cl100k_base`,
`o200k_base`, `p50k_base`, or `r50k_base`) to use tokenizer-based input
estimates for that alias. The alias is what Claude Code sends; `model` is the
ID understood by the upstream. Routes can target local Ollama, vLLM, or
compatible hosted services.

Add `fallbacks = ["other-alias", "..."]` to a route to try other configured
aliases when the primary route fails (see *Fallbacks* below). Set
`cache_control = "passthrough"` on a route to forward Anthropic `cache_control`
breakpoints to an upstream that understands them instead of stripping them
(the default is `"strip"`, which warns and removes them).

Anthropic requests that enable thinking are accepted only on a route configured
with `reasoning_effort`; the configured level is sent upstream. This does not
preserve Anthropic's `budget_tokens` value.

To resolve Anthropic Files API document IDs, set
`LLMBRIDGE_ANTHROPIC_FILES_API_KEY` in the LLMBridge terminal. Anyone able to
submit requests to a bridge using this key may be able to retrieve files
accessible to it.

Optional scanned-PDF OCR requires Tesseract on `PATH` and the PyMuPDF extra:

```sh
python -m pip install -e ".[ocr]"
export LLMBRIDGE_ENABLE_OCR=1
```

OCR is limited to 10 scanned pages and 90 seconds per document. To use
tokenizer-based counts, install the optional tokenizer package:

```sh
python -m pip install -e ".[token-count]"
```

Without `llmbridge.toml`, the legacy single-endpoint environment variables
`LLMBRIDGE_UPSTREAM_BASE_URL`, `LLMBRIDGE_UPSTREAM_MODEL`, and
`LLMBRIDGE_UPSTREAM_API_KEY` remain supported.

The server binds to `127.0.0.1:8787` by default. For local use, incoming API
requests are unauthenticated unless `LLMBRIDGE_API_KEY` is set. When configured,
clients must send the same secret as a bearer token or in the `x-api-key`
header. Health checks remain public. Binding to a non-loopback address requires
an `LLMBRIDGE_API_KEY` of at least 32 characters. Use TLS or a trusted
TLS-terminating reverse proxy for network access; the built-in server does not
provide TLS.

Point Claude Code at LLMBridge and select a configured alias:

```sh
export ANTHROPIC_BASE_URL="http://127.0.0.1:8787"
export ANTHROPIC_AUTH_TOKEN="local"
claude --model nemotron-local
```

If `LLMBRIDGE_API_KEY` is configured, set `ANTHROPIC_AUTH_TOKEN` to the same
secret before launching Claude Code. `GET /v1/models` returns configured
aliases.

## Command-line interface

The `llmbridge` command starts the server when run with no subcommand. It also
offers:

```sh
llmbridge check  --config llmbridge.toml   # validate the routing config
llmbridge models --config llmbridge.toml   # list aliases, endpoints, fallbacks
llmbridge init   --path llmbridge.toml     # write a starter config (--force to overwrite)
llmbridge serve  --host 127.0.0.1 --port 8787 --config llmbridge.toml
```

`python -m llmbridge` behaves the same as `llmbridge serve`.

### Fallbacks and retries

A route may list `fallbacks = ["alias-a", "alias-b"]`. When the primary route
fails, LLMBridge tries each fallback alias in order before reporting an error,
so a quota-exhausted or unavailable model can be transparently replaced. For
streaming requests a fallback is only used before the first byte is sent to the
client; a failure after streaming has started is recorded as interrupted work
instead.

Each route can also be retried before moving on:

- `LLMBRIDGE_ROUTE_ATTEMPTS` — attempts per route (default `1`, max `10`).
- `LLMBRIDGE_ROUTE_RETRY_BACKOFF` — base seconds between attempts, doubled each
  retry (default `0.5`; set to `0` to disable the delay).

## Endpoints

- `GET /health` — readiness check.
- `GET` or `HEAD /api/hello` — gateway connection check.
- `GET /v1/models` — configured model aliases.
- `POST /v1/messages` — Anthropic Messages API-compatible request.
- `POST /v1/messages/count_tokens` — approximate input-token estimate; it is
  not the upstream model's exact tokenizer count.
- `GET /v1/interrupted` — the interrupted work saved for resumption (`404` when
  nothing is saved). Requires client authentication when configured.
- `POST /v1/interrupted/resume` — re-issue the saved interrupted work to a model
  and return a normal Messages response. Pass `{"model": "alias"}` to choose the
  model (defaults to the saved one) and an optional `max_tokens`. Any partial
  output is sent as an assistant turn so the upstream continues from where it
  stopped. On success the saved work is cleared. Requires client authentication
  when configured.
- `DELETE /v1/interrupted` — clear the saved interrupted work. Requires client
  authentication when configured.

## Compatibility and limitations

LLMBridge translates a subset of Anthropic Messages API requests into
OpenAI-compatible Chat Completions calls. This is not full Anthropic API parity.

| Area | Current behavior |
| --- | --- |
| Messages | System text, user/assistant text, and supported content blocks are translated. |
| Images | URL and base64 image blocks are translated to OpenAI image content; actual support depends on the upstream provider and model. |
| Documents | Text, base64, public HTTP(S) URL sources, local file references, and Anthropic Files API IDs are converted to text. Remote downloads are limited to 10 MiB over standard ports. File IDs require an Anthropic API key; local paths require configured allowlisted roots. |
| Tools | Tool definitions, tool choices, tool results, and upstream tool calls are translated to/from function-calling format. Provider/model behavior may differ. |
| Streaming | OpenAI-style SSE chunks are translated into Anthropic-style events, including an initial `ping`, an `input_tokens` count when the upstream reports one, and a `content_filter` finish reason mapped to `end_turn`. Truly unknown upstream finish reasons are rejected; this is not full Anthropic streaming parity. |
| Token counting | `/v1/messages/count_tokens` returns an estimate using a route-configured tiktoken encoding or a character-based fallback. It may differ from the upstream model's exact count. |
| Other request fields | Only fields with explicit translations are sent upstream. Unrecognized Anthropic fields are not forwarded. |
| Unknown content blocks | Unsupported block types are rejected. Assistant `thinking`, `redacted_thinking`, and `signature` blocks are explicitly rejected because generic chat-completions routes cannot represent them. Citation details and tool-result errors are rendered as readable text. |

Routes may set `supports_images = false` or `supports_tools = false` to reject
requests using capabilities known to be unsupported. LLMBridge does not
automatically discover route capabilities.

Local file references are disabled unless one or more trusted directories are
configured in `LLMBRIDGE_DOCUMENT_ROOTS`, separated by the platform path
separator (`:` on macOS/Linux, `;` on Windows). Resolved paths must remain
within those roots, including after symlink resolution. URL documents must use
HTTP(S) on the standard ports (80/443), resolve only to public IP addresses, and
are revalidated and pinned to the validated addresses at each redirect. Avoid
allowing untrusted clients to submit document references.

**Note:** Prompt caching is off by default: Anthropic `cache_control` metadata is
removed with a warning unless a route sets `cache_control = "passthrough"`, in
which case the breakpoints are forwarded to the upstream (useful for
Anthropic-aware gateways, not standard OpenAI endpoints). Remaining
compatibility work includes provider-specific beta and newer API features,
broader content-block support, and richer document fidelity. Full Anthropic API
parity is not provided. OCR is optional and requires PyMuPDF and Tesseract;
document conversion to text does not preserve full layout, embedded images,
formatting, or structured citations.

## Tests

```sh
python -m unittest discover -s tests -v
```

## Partial Work Tracking

LLMBridge includes a **partial work tracking** feature that allows you to resume AI generation work when quotas expire or errors occur, without losing your place.

### Automatic Server Tracking

The bridge tracks every `/v1/messages` request automatically:

1. When a request arrives, the bridge records the prompt (the last user message) and model.
2. If the upstream call stops because of a quota, rate-limit, transient server, connection, timeout, or stream error, the bridge **saves the interrupted work** — including any partial assistant output streamed so far and an estimated progress percentage (generated tokens vs. the request's `max_tokens`).
3. When a later request with the **same prompt** completes successfully, the saved work is cleared automatically.
4. You can inspect the saved work with `GET /v1/interrupted`, and clear it with `DELETE /v1/interrupted`.

Configuration (environment variables):

- `LLMBRIDGE_TRACK_INTERRUPTED` — set to `0`, `false`, `no`, or `off` to disable tracking (enabled by default).
- `LLMBRIDGE_PARTIAL_WORK_PATH` — where to store the state file (defaults to `.partial_work.json` in the server's working directory).

**Privacy:** the state file contains the prompt text in plain text. It is created with private (`0600`) permissions, but disable tracking or point `LLMBRIDGE_PARTIAL_WORK_PATH` at a suitable location if prompts are sensitive.

### Library Usage

The underlying `WorkTracker` class can also be used directly:

1. **Start tracking** — Call `track_generation(prompt, model_name)` at the beginning of a generation run
2. **During generation** — Call `tracker.update_progress(percent, partial_output=...)` periodically (no disk writes)
3. **When generation stops** (quota error, rate limit, etc.) — Call `tracker.save_interrupted()`
   - This saves the interrupted prompt + progress percentage + error context + partial output to `.partial_work.json`
   - No intermediate progress steps are stored — just the prompt being processed, the text generated so far, and optional message history if you passed it to `update_progress()`
   - The state stays naturally small, so no arbitrary size caps are needed
4. **Resume later** — Call `tracker.load_state()` to retrieve the saved interrupted work
   - The original prompt, the partial output generated before the stop, and the progress percentage are preserved
   - Resume from where you left off, not from scratch

### Usage Example

```python
from llmbridge.work_tracker import track_generation

# Start tracking a generation session
tracker = track_generation(
    prompt="Analyze this Python code for security issues and suggest fixes:",
    model_name="claude-3-opus",
)

# During generation — update progress (no disk writes)
tracker.update_progress(progress_percent=75.0, partial_output="...text generated so far...")

# Generation stops (quota exhausted) — save interrupted state
tracker.save_interrupted()

# Later, resume with a different model
loaded = tracker.load_state()
if loaded:
    print(f"Resume from {loaded['progress_percent']:.1f}%")
    print(f"Original prompt: {loaded['prompt'][:50]}...")
    print(f"Partial output: {loaded['partial_output'][:50]}...")
    # Continue generation with a new model using the saved prompt + partial output
```

### State Lifecycle

| Action | Behavior |
|--------|----------|
| `track_generation()` | Starts tracking, no disk write |
| `update_progress()` | Updates internal state, no disk write |
| `save_interrupted()` | **Saves to `.partial_work.json`** (one write) |
| `load_state()` | Loads if the file exists (no expiry) |
| `clear_state()` | Manual cleanup when done |

### File

- **`.partial_work.json`** — Created in the current working directory when `save_interrupted()` is called (or at a custom path passed to `WorkTracker(storage_path=...)` or `track_generation(..., storage_path=...)`)
- **`.partial_work.<namespace>.json`** — Used when you pass `namespace=` to keep independent generations in separate slots. Namespaces must start with an alphanumeric character and contain only letters, digits, `.`, `_`, or `-` (path separators are rejected to prevent directory traversal)
- **Typical size**: 500 bytes - a few KB (the interrupted prompt, partial output, and metadata)
- **No size cap needed**: The state stores one prompt, the partial output of a single run, and optional message history, so it stays small naturally
- **Atomic writes**: State is written to a temp file and moved into place, so an interrupted write cannot corrupt the saved state
- **Thread- and process-safe**: Reads, writes, and deletions are serialized with an advisory file lock, so concurrent trackers sharing a path cannot interleave or corrupt state. Lock files live in the system temp directory (not the project), and in-memory session state is guarded so a single `WorkTracker` can be shared across threads
- **Logging, not stdout**: Status messages go to the `llmbridge.work_tracker` logger, so they never interleave with server/protocol output

### Resume Behavior

| Scenario | Result |
|----------|--------|
| State exists | ✅ Resume from saved prompt + partial output + progress |
| No state file exists | 🆕 Start from scratch with new prompt |

### Why This Design

- **No arbitrary KB caps**: The state stores only one prompt plus the partial output generated so far, so it's naturally small
- **No intermediate bloat**: We don't save on every progress update
- **Clear semantics**: `save_interrupted()` explicitly marks "this generation stopped"
- **No expiry**: Saved state persists until you resume or clear it, so you can pick it back up the next day
- **Concurrency-safe**: An advisory file lock serializes access, and `namespace=`/`storage_path=` let independent generations use separate slots
- **Model-agnostic**: Works the same way when switching between Claude, OpenAI, Gemini, etc.

## License

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE).
