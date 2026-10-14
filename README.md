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

## Endpoints

- `GET /health` — readiness check.
- `GET` or `HEAD /api/hello` — gateway connection check.
- `GET /v1/models` — configured model aliases.
- `POST /v1/messages` — Anthropic Messages API-compatible request.
- `POST /v1/messages/count_tokens` — approximate input-token estimate; it is
  not the upstream model's exact tokenizer count.

## Compatibility and limitations

LLMBridge translates a subset of Anthropic Messages API requests into
OpenAI-compatible Chat Completions calls. This is not full Anthropic API parity.

| Area | Current behavior |
| --- | --- |
| Messages | System text, user/assistant text, and supported content blocks are translated. |
| Images | URL and base64 image blocks are translated to OpenAI image content; actual support depends on the upstream provider and model. |
| Documents | Text, base64, public HTTP(S) URL sources, local file references, and Anthropic Files API IDs are converted to text. Remote downloads are limited to 10 MiB. File IDs require an Anthropic API key; local paths require configured allowlisted roots. |
| Tools | Tool definitions, tool choices, tool results, and upstream tool calls are translated to/from function-calling format. Provider/model behavior may differ. |
| Streaming | Basic OpenAI-style SSE chunks are translated into Anthropic-style events. Unknown upstream finish reasons are rejected; this is not full Anthropic streaming parity. |
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
HTTP(S), resolve only to public IP addresses, and are revalidated and pinned
to the validated address at each redirect. Avoid allowing untrusted clients
to submit document references.

**Note:** Prompt caching is not provided: Anthropic `cache_control` metadata is
removed with a warning, and requests continue without caching. Remaining
compatibility work includes provider-specific beta and newer API features,
broader content-block support, and richer document fidelity. Full Anthropic API
parity is not provided. OCR is optional and requires PyMuPDF and Tesseract;
document conversion to text does not preserve full layout, embedded images,
formatting, or structured citations.

## Tests

```sh
python -m unittest discover -s tests -v
```
