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

Copy `llmbridge.toml.example` to `llmbridge.toml`, edit the model aliases and
endpoints, and set any required upstream API keys in environment variables:

```sh
cp llmbridge.toml.example llmbridge.toml
export HOSTED_MODEL_API_KEY="..."
python -m llmbridge
```

Each `[models.<alias>]` entry has a `base_url` and may specify an upstream
`model` and an `api_key_env` name. Add one entry for each model; aliases can
target different models and providers. The alias is what Claude Code sends;
`model` is the ID understood by the upstream. Routes may point to local Ollama,
vLLM, or compatible hosted services. Hosted free-tier availability, quotas,
model support, and API compatibility depend on the provider and can change;
LLMBridge does not make a paid provider free.

Install the project and its PDF text-extraction dependency with
`python -m pip install -e .`.

Without `llmbridge.toml`, the legacy single-endpoint environment variables
`LLMBRIDGE_UPSTREAM_BASE_URL`, `LLMBRIDGE_UPSTREAM_MODEL`, and
`LLMBRIDGE_UPSTREAM_API_KEY` remain supported.

The server binds to `127.0.0.1:8787` by default. To use a different port, set
`LLMBRIDGE_PORT`. For local use, incoming API requests are unauthenticated
unless `LLMBRIDGE_API_KEY` is set. When configured, clients must send the same
secret as `Authorization: Bearer <key>` or in the `x-api-key` header. Health
checks remain public.
Binding to a non-loopback address requires an `LLMBRIDGE_API_KEY` of at least
32 characters; the server refuses to start without one. Generate a key with:

```sh
export LLMBRIDGE_API_KEY="$(python3 -c 'import secrets; print(secrets.token_hex(32))')"
```

Use TLS or a trusted TLS-terminating reverse proxy for network access; the
built-in server does not provide TLS.

Point Claude Code at LLMBridge and select a configured alias:

```sh
export ANTHROPIC_BASE_URL="http://127.0.0.1:8787"
export ANTHROPIC_AUTH_TOKEN="local"
claude --model nemotron-local
```

If `LLMBRIDGE_API_KEY` is configured, set `ANTHROPIC_AUTH_TOKEN` to the same
secret before launching Claude Code.

Set `ANTHROPIC_MODEL` to the desired alias instead if that is how your Claude
Code setup selects a model. `GET /v1/models` returns configured aliases.

## Endpoints

- `GET /health` — readiness check.
- `GET` or `HEAD /api/hello` — gateway connection check.
- `GET /v1/models` — configured model aliases.
- `POST /v1/messages` — Anthropic Messages API-compatible request.
- `POST /v1/messages/count_tokens` — approximate input-token estimate for
  compatibility; it is not the upstream model's exact tokenizer count.

Other Anthropic-specific features not described above are not implemented yet.
Document blocks with plain-text or UTF-8 text sources are sent as text. Base64
PDFs are converted to extracted page text; DOCX, XLSX, and PPTX are converted
to their readable text. HTML markup and RTF formatting are stripped. Scanned
PDFs need OCR; document layout, embedded images, spreadsheet formatting, and
citations are not preserved. Office archive extraction is capped at 25 MiB of
expanded content. Legacy binary Office formats, URL documents,
Anthropic file IDs, and other unlisted binary formats cannot be passed through
generic OpenAI-compatible chat routes. Convert unsupported files to text or a
supported format before sending. Model behavior and tool calling still depend
on the capabilities of the selected upstream model.

## Tests

```sh
python -m unittest discover -s tests -v
```
