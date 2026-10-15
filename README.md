# LLMBridge

A local protocol bridge that routes **Anthropic Messages API** requests to **OpenAI-compatible** model endpoints (Ollama, vLLM, NVIDIA, OpenRouter, etc.). Define model aliases, then use them in Claude Code.

---

## Quick Start

```bash
pip install -e .
llmbridge setup        # guided config (Ollama, OpenAI, OpenRouter, NVIDIA presets)
llmbridge doctor       # verify config, keys, port
llmbridge serve        # start server on http://127.0.0.1:8787
```

### Point Claude Code at the bridge

```bash
export ANTHROPIC_BASE_URL="http://127.0.0.1:8787"
export ANTHROPIC_AUTH_TOKEN="local"      # or your LLMBRIDGE_API_KEY
claude --model nemotron-local             # use a configured alias
```

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
| `DELETE` | `/v1/generations/{id}` | Remove generation from ledger |

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

Ledger: atomic writes, file-lock serialized, per-file `0600` permissions. Disable with `LLMBRIDGE_TRACK_GENERATIONS=0`.

---

## Partial Work Tracking (Auto)

The bridge automatically saves interrupted work when upstream fails (quota, rate limit, timeout, stream error):

- **Saved**: prompt, partial output, progress %, error context
- **Auto-cleared**: when same prompt succeeds
- **Inspect**: `GET /v1/interrupted`
- **Resume**: `POST /v1/interrupted/resume` — sends partial output as assistant turn

```bash
# Config
LLMBRIDGE_TRACK_INTERRUPTED=1   # enabled by default
LLMBRIDGE_PARTIAL_WORK_PATH=.partial_work.json
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
- Ledger & partial work files contain plain-text prompts (`0600` perms)
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