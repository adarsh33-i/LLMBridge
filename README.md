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

## Setup

LLMBridge requires Python 3.11 or newer and is designed to run on macOS,
Windows, and Linux. The commands below use a virtual environment to isolate
the installation. Run them from a terminal in the directory where you want
the project.

### 1. Install prerequisites

Install Python 3.11+ and Git for your OS. On Windows, enable the Python
installer's **Add Python to PATH** option. Verify that Python is available:

```text
python --version
```

On macOS or Linux, use `python3 --version` if `python` is not the Python 3
command. Continue only if the version is 3.11 or newer.

### 2. Get the source

```sh
git clone https://github.com/adarsh33-i/LLMBridge.git
cd LLMBridge
```

The same commands work in Windows PowerShell:

```powershell
git clone https://github.com/adarsh33-i/LLMBridge.git
Set-Location LLMBridge
```

### 3. Create and activate a virtual environment

**macOS (zsh/bash):**

```sh
python3 -m venv .venv
source .venv/bin/activate
```

**Linux (bash/zsh):**

```sh
python3 -m venv .venv
source .venv/bin/activate
```

**Windows (PowerShell):**

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python --version
```

Confirm the activated environment reports Python 3.11 or newer.

If PowerShell blocks activation, allow it for the current terminal session
only, then retry:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

Alternatively, skip activation and use the environment's Python directly:
`.venv\Scripts\python.exe` on Windows or `.venv/bin/python` on macOS/Linux.

### 4. Install LLMBridge

With the virtual environment active on any OS:

```text
python -m pip install --upgrade pip
python -m pip install -e .
```

### 5. Configure model routes

Copy the example config:

**macOS/Linux:**

```sh
cp llmbridge.toml.example llmbridge.toml
```

**Windows PowerShell:**

```powershell
Copy-Item llmbridge.toml.example llmbridge.toml
```

Edit `llmbridge.toml`. Replace example/placeholder routes with your actual
provider URL and model ID, and remove routes you will not use. Each
`[models.<alias>]` entry has a `base_url` and may specify an upstream `model`
and an `api_key_env` environment-variable name.
The alias is what Claude Code sends; `model` is the ID understood by the
upstream. Routes can target local Ollama, vLLM, or compatible hosted services.
The sample includes example NVIDIA routes; select a model available to your
account. Provider quotas, model availability, and API compatibility depend on
the provider.

Set each required upstream key in the terminal where you will run LLMBridge.
For example, if the selected route uses `api_key_env = "NVIDIA_API_KEY"`:

**macOS/Linux:**

```sh
export NVIDIA_API_KEY="your-provider-key"
```

**Windows PowerShell:**

```powershell
$env:NVIDIA_API_KEY = "your-provider-key"
```

These commands set a key only for the current terminal session. Keep provider
keys private; use your OS's environment-variable manager if you need a key to
persist between sessions.

### 6. Start and verify LLMBridge

From the project directory, with the virtual environment active:

```text
python -m llmbridge
```

By default, the server listens on `http://127.0.0.1:8787`. Keep this terminal
open while using the bridge. Stop the server with `Ctrl+C`.

Open a second terminal to check the local server:

**macOS/Linux:**

```sh
curl http://127.0.0.1:8787/health
curl http://127.0.0.1:8787/v1/models
```

**Windows PowerShell:**

```powershell
Invoke-RestMethod http://127.0.0.1:8787/health
Invoke-RestMethod http://127.0.0.1:8787/v1/models
```

The model list shows configured aliases. A successful health check confirms
the local process is responding; it does not verify upstream credentials or
model availability.

### 7. Connect Claude Code

Set the Claude Code base URL in the terminal where you launch Claude Code.
Replace `nemotron-local` below with a configured alias:

**macOS/Linux:**

```sh
export ANTHROPIC_BASE_URL="http://127.0.0.1:8787"
export ANTHROPIC_AUTH_TOKEN="local"
claude --model nemotron-local
```

**Windows PowerShell:**

```powershell
$env:ANTHROPIC_BASE_URL = "http://127.0.0.1:8787"
$env:ANTHROPIC_AUTH_TOKEN = "local"
claude --model nemotron-local
```

If `LLMBRIDGE_API_KEY` is enabled below, set `ANTHROPIC_AUTH_TOKEN` to the
same key instead of `local`. Alternatively, set `ANTHROPIC_MODEL` to an alias
if that is how your Claude Code installation selects models.

## Server security

The server binds to `127.0.0.1:8787` by default. Local API requests are
unauthenticated unless `LLMBRIDGE_API_KEY` is configured. When configured,
clients must send this key using the standard Bearer authorization scheme or
the `x-api-key` header. Health checks remain public.

Binding to a non-loopback address requires an `LLMBRIDGE_API_KEY` of at least
32 characters; the server refuses to start without one. To generate a key:

**macOS/Linux:**

```sh
export LLMBRIDGE_API_KEY="$(python3 -c 'import secrets; print(secrets.token_hex(32))')"
```

**Windows PowerShell:**

```powershell
$env:LLMBRIDGE_API_KEY = (python -c "import secrets; print(secrets.token_hex(32))")
```

Use TLS or a trusted TLS-terminating reverse proxy for network access; the
built-in server does not provide TLS. Set `LLMBRIDGE_HOST` and optionally
`LLMBRIDGE_PORT` before starting to change the listener. Use the shell's
environment-variable syntax shown above.

## Advanced configuration

Without `llmbridge.toml`, the legacy single-endpoint environment variables
`LLMBRIDGE_UPSTREAM_BASE_URL`, `LLMBRIDGE_UPSTREAM_MODEL`, and
`LLMBRIDGE_UPSTREAM_API_KEY` remain supported. Set
`LLMBRIDGE_UPSTREAM_BASE_URL` to the compatible API base URL and optionally set
`LLMBRIDGE_UPSTREAM_MODEL` to the provider's model ID. Set
`LLMBRIDGE_UPSTREAM_API_KEY` to the actual upstream key value; it is read
directly as the secret in this legacy single-route mode.

Hosted free-tier availability, quotas, model support, and API compatibility
depend on the provider and can change; LLMBridge does not make a paid provider
free.

## Endpoints

- `GET /health` — readiness check.
- `GET` or `HEAD /api/hello` — gateway connection check.
- `GET /v1/models` — configured model aliases.
- `POST /v1/messages` — Anthropic Messages API-compatible request.
- `POST /v1/messages/count_tokens` — approximate input-token estimate; it is
  not the upstream model's exact tokenizer count.

Other Anthropic-specific features not described above are not implemented yet.
Document blocks with plain-text or UTF-8 text sources are sent as text. Base64
PDFs are converted to extracted page text; DOCX, XLSX, and PPTX are converted
to readable text. HTML markup and RTF formatting are stripped. Scanned PDFs
need OCR; document layout, embedded images, spreadsheet formatting, and
citations are not preserved. Office archive extraction is capped at 25 MiB of
expanded content. Legacy binary Office formats, URL documents, Anthropic file
IDs, and other unlisted binary formats cannot be passed through generic
OpenAI-compatible chat routes. Convert unsupported files to text or a
supported format before sending. Model behavior and tool calling depend on
the selected upstream model's capabilities.

## Tests

With the project environment active:

```text
python -m unittest discover -s tests -v
```

The project is designed for macOS, Windows, and Linux. The test suite has been
verified on macOS; Windows and Linux verification is still needed.
