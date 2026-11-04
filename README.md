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

Or skip the exports entirely: `llmbridge claude` launches Claude Code wired to the bridge (add `--print-env` to just see the values). It ignores any `ANTHROPIC_BASE_URL` already in your shell — a stray export pointing at another gateway is exactly what it exists to override — and targets `LLMBRIDGE_BASE_URL`, `--base-url`, or `--host`/`--port` instead. It passes the URL and token as a CLI settings layer as well as environment variables: Claude Code lets a settings-pinned env outrank inherited exports, which is how a launch can look correct and still send traffic elsewhere. The CLI layer wins over every settings file.

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
| `tokenizer_encoding` | tiktoken encoding for `count_tokens` (`cl100k_base`, `o200k_base`, `p50k_base`, `r50k_base`); needs the `token-count` extra |
| `fallbacks` | Aliases to try on failure |
| `cache_control` | `"strip"` (default) or `"passthrough"` |

Legacy single-endpoint vars (`LLMBRIDGE_UPSTREAM_*`) still supported.

Client `thinking` blocks follow the route: `enabled` maps to `reasoning_effort` when the route sets one and is dropped otherwise; `disabled` stays off. A thinking type the bridge does not recognise — newer clients already send them — is treated as enabled rather than rejected, so a client-version bump cannot turn every request into a 400.

A Claude Code turn carries roughly 28.6k input tokens of context, so a local route spends most of its wall time in prefill: measured at 1.7–19.7 tok/s on `qwen3:8b` against 38.8–60.4 tok/s on a hosted route. Keep a local alias as a fallback for resilience, but point long agent sessions at a hosted route.

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
| `LLMBRIDGE_BASE_URL` | `http://127.0.0.1:8787` | Which bridge URL the commands that talk to a running server (`fork`, `claude`) use. `status`, `stats`, `generations`, `tree`, `diff`, `audit`, `check` and `models` read the ledger and config directly and need no server |
| `LLMBRIDGE_CLIENT_TIMEOUT_SECONDS` | `300` | How long one client request may stream |
| `LLMBRIDGE_UPSTREAM_TIMEOUT_SECONDS` | `180` | Timeout for each upstream call |
| `LLMBRIDGE_ROUTE_ATTEMPTS` | `1` | Tries on one route before falling through to the next fallback (max 10) |
| `LLMBRIDGE_ROUTE_RETRY_BACKOFF` | `0.5` | Base seconds between attempts; doubles each try |
| `LLMBRIDGE_DOCUMENT_ROOTS` | unset | Directories allowed as local `file:` document sources (`os.pathsep`-separated). Unset disables local file documents |
| `LLMBRIDGE_ANTHROPIC_FILES_API_KEY` | unset | Enables document sources that reference the Anthropic Files API |

Feature-specific vars (`LLMBRIDGE_SELF_HEAL`, `LLMBRIDGE_HEAL_TOOL_CALLS`, `LLMBRIDGE_HEAL_HINT`, `LLMBRIDGE_SEAM_EVENTS`, `LLMBRIDGE_ADAPTIVE_FALLBACKS`, `LLMBRIDGE_SHADOW_AFTER_SECONDS`, `LLMBRIDGE_AUTO_RESUME`, `LLMBRIDGE_HANDOFF_*`, `LLMBRIDGE_QUEUE_*`, `LLMBRIDGE_COALESCE_REQUESTS`, `LLMBRIDGE_TRACK_*`, `LLMBRIDGE_LEDGER_*`, `LLMBRIDGE_ENABLE_OCR`) are documented with their features below.

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
| `502 ... Upstream returned an empty answer (reasoning alone is not text output)` | A reasoning model spent the whole `max_tokens` budget thinking and emitted no text — common with `qwen3` at small budgets | Raise `max_tokens`, or continue the saved generation with `llmbridge fork <id> --model <alias>`; an empty answer is never passed off as success |

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
| `llmbridge stats` | Status counts, tokens, p50/p95 wall time, heal verdicts (`--json`) |
| `llmbridge fork <id> --model <alias>` | Continue a generation on another model |
| `llmbridge handoff <id> [--to <url> --model <alias>]` | Print a signed token another bridge can resume, or send it there directly |
| `llmbridge tree <id>` | Render a fork lineage (ancestors + descendants) |
| `llmbridge diff <idA> <idB>` | Compare two generations' outputs side by side |
| `llmbridge replay <id>\|--all` | Re-send recorded requests; flag output drift |
| `llmbridge audit` | Consolidate heal seam audits into a digest that outlives records |
| `llmbridge dream <id> --at N` | Replay a saved answer as if its stream had died mid-way |
| `llmbridge failover [id…]` | Replay suspect heals across fallbacks and rank them |

---

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Readiness check |
| `GET` | `/v1/models` | Configured aliases |
| `POST` | `/v1/messages` | Anthropic-compatible completion |
| `POST` | `/v1/messages/count_tokens` | Approximate input-token estimate |
| `GET` | `/v1/interrupted` | Newest interrupted record + `recoverable_count`, `other_recoverable` (404 if none) |
| `GET` | `/v1/stats` | Ledger stats: statuses, tokens, wall time, heal verdicts and digest rate |
| `POST` | `/v1/interrupted/resume` | Resume interrupted work on a model; marks the source `resumed`. Accepts `{"handoff_token": ...}` from another bridge |
| `DELETE` | `/v1/interrupted` | Delete the newest record, returns `remaining_recoverable` |
| `GET` | `/v1/generations` | List durable generations |
| `GET` | `/v1/generations/{id}` | Full generation record |
| `POST` | `/v1/generations/{id}/fork` | Fork/continue on another model; settles the source if it was interrupted |
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
- **Prefill-safe**: when the conversation already ends in an assistant turn (a prefill), fork, heal, and resume merge the continuation into that turn instead of adding a second one — Anthropic rejects adjacent assistant turns

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

Ledger: atomic writes, file-lock serialized, per-file `0600` permissions. Disable with `LLMBRIDGE_TRACK_GENERATIONS=0`. Only the newest records are kept — 10 by default (`MAX_TRACKED_INTERRUPTED` in `llmbridge/ledger.py`, override with `LLMBRIDGE_LEDGER_MAX_RECORDS`); older ones are pruned as new generations arrive. Two things are exempt from that window, each with its own budget: a record whose newest heal audit is `suspect` (up to 25, `LLMBRIDGE_AUDIT_MAX_SUSPECT`) and work that is still unfinished — `interrupted` or `queued` (up to 10, `LLMBRIDGE_LEDGER_MAX_RECOVERABLE`). Set either to `0` to restore the plain window. Without those exemptions the bridge forgets an invisible heal, or stops offering work it could still finish, for a reason that has nothing to do with memory policy. See [What the audit remembers](#what-the-audit-remembers).

Work in flight when the process exits is settled, not forgotten: a graceful stop marks any `running` record `interrupted`, and a sweep at startup — repeated every minute while serving — settles records left behind by a hard kill once they are older than twice the upstream timeout (five minutes minimum). Either way the request surfaces as recoverable work instead of claiming to still be running forever.

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

`replay` sends the exact recorded request back to the configured route and scores the new output against the saved one with `difflib` similarity. Use it after switching models, editing prompts, or upgrading upstreams to catch behavioral drift. It compares real text only: a record with no saved output is `SKIP`ped (two empty answers are trivially identical), and a replay that comes back empty is a `FAIL`, not a silent pass.

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

Without partial text or a fallback the normal interrupted-work flow applies. A stream that dies inside a tool call is healed too — see [Healing inside a tool call](#healing-inside-a-tool-call).

### Healing inside a tool call

A gateway that fails over at the request restarts the turn, so an agent mid tool call gets a different call or an error. LLMBridge finishes the same turn.

While tool-call healing is on (the default, `LLMBRIDGE_HEAL_TOOL_CALLS=1`), a `tool_use` block is delivered to the client **whole, at the moment it completes**, instead of argument fragment by fragment. Clients act on a tool call only once its JSON is complete, so nothing observable changes on a healthy stream, and a model that dies mid-call has then shown the client nothing it must retract. When the stream breaks:

1. Tool calls that completed before the break are released to the client under their original ids.
2. The fallback receives the text so far as an assistant prefill plus a system addendum naming the completed calls (do not repeat), the tool that was cut off, and the argument prefix it had written. It is asked to finish that call with the same tool.
3. The fallback's tool blocks are re-indexed above every index the client has seen and forwarded with the fallback's own ids. The bridge closes every open block and ends the message with the fallback's own stop reason, so a turn that ends in a tool call still says `tool_use`.
4. The ledger gets a `heal` with `reason: tool_call` and which calls were cut and issued, and a `heal_seam` with `tool_call: true` and `tool_finished`. A fallback that called a different tool than the one cut off, or none, is `suspect`, and the digest carries `tool_finished` so `stats` can report it.

Set `LLMBRIDGE_HEAL_TOOL_CALLS=0` to restore the previous behaviour: a break inside a tool call is interrupted and saved, never stitched.

### Fallbacks ranked by evidence

The config order of `fallbacks` is a guess made before any heal happened. The digest knows better: it has a scored seam for every join each fallback ever made onto this primary.

```bash
LLMBRIDGE_ADAPTIVE_FALLBACKS=1      # off by default
LLMBRIDGE_ADAPTIVE_MIN_SAMPLES=3    # seams a fallback needs before its rate counts
```

When on, a heal orders the remaining candidates by their measured suspect rate for this primary model, lowest first. A fallback with fewer than the minimum samples keeps its config position relative to the others and sorts after every trusted one, so the chain only moves where the evidence can justify it. The order the heal chose from is written into the `heal` event (`order`: model, config position, samples, suspect rate, trusted), so every reordering is auditable. `llmbridge failover` is the offline version of the same ranking; this is the online one. The digest is the source, so pruned records still count.

### Speculative shadow stream

A provider that accepts the connection and then says nothing for ten seconds is a stall the normal timeout cannot tell from a slow answer. Opt in to race it:

```bash
LLMBRIDGE_SHADOW_AFTER_SECONDS=3    # 0 / unset = off
```

The primary is opened at once. If no visible token has arrived after the delay, the first fallback is opened too and whichever yields a token first carries the stream; the other is closed. The client sees one `message_start` and one model's answer. The loser's partial output is never shown, and a shadow winner that later breaks still heals onto the next candidate like any stream. Each race writes a `shadow` event (winner, loser, reason, how long the bridge waited, and `loser_sent`: whether the losing request was sent to its provider, and so billed) whether or not the fallback was actually opened, so the delay can be judged against how often it triggered and cost a second call. A primary that dies after the shadow was opened hands the stream to the shadow at once (`reason: primary_failed`) rather than waiting for the shadow's first token. Every race that opens the shadow spends a second upstream call on purpose; that is the trade the delay sets.


### Seam audit

A stitched answer has a joint, and the client cannot see it: one model's text ends
mid-sentence and another model's text carries on inside the same stream. The bridge
scores that joint and writes a `heal_seam` event with the seam position, how many
characters the continuation added, and how much of it repeats what the first model
already wrote.

A seam is flagged `suspect` when either half genuinely looks wrong:

| flag | what it catches |
|---|---|
| `duplicated` | the continuation re-emits text that started where the first model stopped (12+ characters shared, and the repeat has to be anchored at the joint — a phrase reused mid-continuation is ordinary language) |
| `restarted` | the continuation opens with a fresh preamble such as "Sure, here's…" — it ignored the appended assistant turn and began a new answer. Markdown, quotes, list markers, and leading numbers are stripped before the check, so `**Sure,**` and `> Of course!` are caught too |

```bash
llmbridge status
# Healed streams audited: 2 (1 suspect)
#   gen_x1a2b3: nemotron -> qwen at char 480, 31 chars repeated (duplicated)
#   flag only: review with `llmbridge replay <id>`
# Digest: 2 seam(s) (1 suspect, 50%)
```

The two counts answer different questions: `Healed streams audited` is the working set
(records still on disk), `Digest` is every seam ever consolidated, including ones whose
records the retention window has since pruned. `llmbridge stats` reports both side by side.

`llmbridge stats` also prints status counts, recorded output tokens, and p50/p95 wall time
over completed generations; `--json` emits the same numbers (also served at `GET /v1/stats`).

```bash
llmbridge stats
# Ledger: .llmbridge/generations (12 records)
# Statuses: completed=10, interrupted=2
# Recoverable: 2
# Output tokens: 4120 over 10 record(s)
# Wall time: p50 2.4s, p95 9.8s, max 12.1s (n=10)
# Heals: 2 healed, 2 audited (1 suspect, 50% suspect); duplicated 1, restarted 0
# Digest: 2 seam(s) (1 suspect, 50%)
```

This is an **audit, not a gate**. A suspect seam is still delivered — the whole point of
self-healing is that the client never sees the repair, and refusing to stitch would send
a truncated answer instead. What the flag buys you is a denominator: `llmbridge replay
<id>` re-sends the recorded request and scores the new output against the saved stitched
one, so a suspect join can be checked rather than argued about.

Two known gaps, stated plainly. The audit is written when the continuation finishes, so
a client that disconnects mid-continuation leaves a `heal` with no `heal_seam`. And
`overlap` compares only the 160 characters either side of the joint — a fallback that
repeats something said much earlier in the answer is not caught.

A heal whose fallback contributed no text is recorded as a `heal` but not audited: there
is no joint to compare, and scoring it would count a pass for a join that never happened.

### What the audit remembers

Ten records is the whole ledger by default, so an audit is otherwise erased by ordinary
traffic: twenty heals with five suspect joins would report as zero suspects once the
window moved on. Forgetting is now a stated policy instead of a storage side effect.

A record whose newest seam audit is `suspect` is kept beyond the retention window, up to
`LLMBRIDGE_AUDIT_MAX_SUSPECT` of them (default 25, newest first). Set it to `0` to get the
plain window back. The exemption applies only to the automatic window — giving `prune()` an
explicit count is a hard cut and keeps nothing. Clean seams still age out after ten
requests, which is what the next command is for.

Unfinished work gets the same treatment under its own budget (`LLMBRIDGE_LEDGER_MAX_RECOVERABLE`,
see [Partial Work Tracking](#partial-work-tracking-auto)): a `resumed` record is no longer
exempt, because it is no longer work the bridge could still do.

### Consolidating audits

`llmbridge status` reads records, and records die. `llmbridge audit` copies every seam
audit into `<ledger>/heal_audit.jsonl`: one line per audit, keyed by generation id and
event sequence, so running it twice adds nothing.

```bash
llmbridge audit
# Digest: .llmbridge/generations/heal_audit.jsonl
# Seam audits on disk: 5 (0 newly consolidated)
# Full history: 20 audit(s), 5 suspect (25%)
#   gen_x1a2b3: nemotron -> qwen at char 480 (duplicated)
#   audits consolidate only while their record survives: run after traffic
```

The digest keeps the denominator after the records behind it are gone, so the suspect rate
is a rate over history and not over the last ten requests. Consolidation also runs by
itself at the two moments when records are about to disappear: before the retention prune
deletes a record that carries seam audits, and on graceful shutdown (`SIGINT`/`SIGTERM`
through `serve`, or `LLMBridgeServer.stop()`). A clean seam is copied out a moment before
its record is dropped, so ordinary traffic can no longer erase the denominator. A hard
kill still loses whatever was on disk and not yet copied — run `llmbridge audit` on a
schedule if you want that gap closed. An empty run creates no file: a prune that only
deletes records with no audits consolidates nothing. `llmbridge status` says
`not yet consolidated` when it can see audits the digest has not captured.

### Dreaming a break

Real bad seams are rare, and you cannot ask a stream to break on cue. `llmbridge dream`
invent one: it cuts a completed generation's saved answer at a character, sends that cut
text as the assistant turn a fallback would have been given, and scores the continuation
that comes back.

```bash
llmbridge dream gen_x1a2b3 --at 480 --model qwen     # cut at the midpoint if --at is omitted
# Dreaming gen_x1a2b3 on qwen: cut at char 480 of 1120
#   continuation 264 chars vs recorded remainder 640 chars, similarity 0.18
#   seam: 0 chars repeated (0.00)
#   clean join, but the continuation diverges from what the first model said next
```

Two numbers, because they answer different questions. `seam` runs the same scorer the
server uses, so it says whether that join would have been flagged. `similarity` compares
the continuation against what the original model actually went on to say — the silent-fix
case, where the joint looks clean and the answer nevertheless turns somewhere else.
`--json` emits both plus the cut position. Dreams are read-only: nothing is written to the
ledger, and no heal is recorded for a stream that never broke.

### Ranking fallbacks on the seams you already have

`dream` tests one model at one cut. `llmbridge failover` does the same thing in bulk: it
takes the newest recorded suspect heals, cuts each at its recorded seam, and replays the
continuation on every fallback configured for that record's model.

```bash
llmbridge failover                       # newest 5 suspect heals x their fallbacks
llmbridge failover --limit 10 --threshold 0.7
llmbridge failover gen_x1a2b3 gen_x4c5d6 --models qwen,nemotron
llmbridge failover --dry-run             # print the sampling plan, send nothing

# MODEL           RUNS  FLAGGED  FAITHFUL  MEAN SIM  WORST
# qwen            5     1        3         0.74      gen_x4c5d6 (0.21)
# nemotron        5     3        1         0.48      gen_x1a2b3 (0.09)
#
# Best fallback on these seams: qwen (mean similarity 0.74, 1 flagged join(s))
```

Two columns, two questions. `FLAGGED` is the server's own seam scorer: how often that
fallback would have produced a join the audit would flag. `FAITHFUL` counts continuations
that stayed close to what the interrupted model actually said next (`--threshold`, default
`0.8`), which catches a clean-looking join that changed the answer. Ranking sorts by fewest
flagged joins, then by mean similarity. `--json` emits the plan, the ranking, and every
individual run.

Read the numbers for what they are. Each run spends real upstream tokens — the plan line
states the call count before anything is sent, and `--dry-run` stops there. A handful of
seams is a sample, not a rate: the base rate of bad heals is unknown, so five flagged joins
from five samples says "this fallback joins badly on these five", not "20% of heals fail".
No ledger record is written, so a ranking never pollutes the history it was measured on.

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

A dispatch retries transient failures up to the attempt budget: quota (`402`/`429`), any
`5xx`, timeouts, connection errors, and replies the bridge cannot translate. A client-side
error such as `400` fails the record immediately — retrying a malformed request only burns
the budget.

The dispatcher is per process. Two bridge processes sharing one ledger can both pick up the
same due record; there is no lease, so each would send its own upstream request. Run one
dispatcher per ledger.

---

## Duplicate Request Coalescing

Agents retry. A client that times out and re-sends the identical non-streaming request
otherwise pays for the same answer twice, and writes two generations for it. Coalescing
lets the duplicate attach to the request already in flight and receive its rendered
response instead.

```bash
LLMBRIDGE_COALESCE_REQUESTS=deterministic   # off (default) | deterministic | always
```

| Mode | Joins when |
|------|-----------|
| `off` (default) | Never |
| `deterministic` | The body pins `"temperature": 0` explicitly — the same request should produce the same answer |
| `always` | Any byte-identical non-streaming request is in flight (`1`/`true`/`yes`/`on` also work) |

Identity is the request's content fingerprint, the same one the ledger uses: model,
messages, system, tools and every parameter. A follower never opens a generation record;
it answers with the leader's exact status and payload plus `X-LLMBridge-Coalesced: 1` and
the leader's `X-LLMBridge-Generation`, so it can still inspect what it received. Followers
wait up to `LLMBRIDGE_CLIENT_TIMEOUT_SECONDS`, and a follower whose leader never answered
gets the leader's error rather than a second upstream call.

The default is `off` on purpose. At `temperature > 0` two identical requests are two
legitimate samples, and collapsing them would silently reduce variety. `deterministic`
only activates when the request itself says it wants no variety. Streaming responses are
never coalesced: the chunks belong to one connection, and replaying them to a second
client would fake a stream that never happened.

---

## Partial Work Tracking (Auto)

The bridge automatically saves interrupted work when upstream fails (quota, rate limit, timeout, stream error). Interrupted work is not a separate file — it is stored as an `interrupted` generation record in the same ledger (`.llmbridge/generations/`), so it can be inspected and forked like any other generation:

- **Saved**: prompt, partial output, progress %, error context, replayable request
- **Matched**: by the request's content fingerprint, not by prompt text. Re-sending the request that died settles that interruption; a different request that happens to share the same prompt never does.
- **Offered**: `GET /v1/interrupted` returns the newest interrupted record, plus `recoverable_count` and `other_recoverable` — the ids still waiting, each addressable at `/v1/generations/{id}/fork`. `llmbridge status` prints the same list.
- **Resumed**: `POST /v1/interrupted/resume` replays the work onto the model you name, with the partial output returned as an assistant turn. The interruption is kept and marked `resumed`, with a `resume` event naming the run that finished it. Deleting it would erase the only evidence that the answer needed a second try.
- **Forked**: forking an `interrupted` generation settles it the same way — but only when the fork completes. A failed fork leaves both runs recoverable.
- **Cleared**: `DELETE /v1/interrupted` deletes the newest record and answers with `remaining_recoverable`. This is the one place work is removed rather than marked.
- **Picked up automatically** (opt-in, `LLMBRIDGE_AUTO_RESUME=1`): a byte-identical re-send of the request that died continues from the saved text instead of starting over. See [Automatic resume on re-send](#automatic-resume-on-re-send).
- **Retention**: the newest 10 records, plus up to 10 recent `interrupted`/`queued` records and up to 25 suspect seam audits kept beyond that window.

```bash
# Config
LLMBRIDGE_TRACK_INTERRUPTED=1          # enabled by default
LLMBRIDGE_LEDGER_PATH=.llmbridge/generations
LLMBRIDGE_LEDGER_MAX_RECORDS=10        # the retention window
LLMBRIDGE_LEDGER_MAX_RECOVERABLE=10    # unfinished work kept beyond it (0 = plain window)
LLMBRIDGE_AUDIT_MAX_SUSPECT=25         # suspect seam audits kept beyond it
```

Without the recoverable exemption the bridge silently stops offering work it could still finish: after ten unrelated requests, the interrupted record ages out of the window. Both exemptions apply only to the automatic window — `GenerationLedger.prune(keep=N)` and `DELETE /v1/interrupted` are hard cuts that keep nothing.

One caveat on resuming: the stored partial output is **text only**. A stream that died inside a tool call keeps its text deltas and nothing else, so a resume rebuilds the conversation from the replayed request rather than from the interrupted call itself.

### Telling the fallback how to continue

```bash
LLMBRIDGE_HEAL_HINT=1    # off by default
```

By default a heal hands the fallback the partial text as a bare assistant turn and lets the
model work out that it should continue. With the hint on, the heal request also carries a
short system addendum: continue from exactly where the text stops, do not restate or
acknowledge the interruption. The hint goes upstream only; the client's stream is
unchanged. Every `heal` and `heal_seam` event records `hinted`, and the digest keeps it, so
`llmbridge stats` can answer the only question that matters about a prompt tweak: did it
change the restart rate for this model?

```bash
llmbridge stats
# ...
# Heal hint (restart rate, hinted vs not):
#   qwen3:8b: 0% over 12 vs 25% over 20 (delta -25%)
```

The line appears once a model has seams on both sides. Tool-call heals are excluded from
the comparison, since they always carry an addendum. The default is off so the digest keeps
accruing un-hinted samples; turn it on once you have a baseline, and leave it on if the
delta is negative.

### Telling the client how the join was scored

```bash
LLMBRIDGE_SEAM_EVENTS=1    # off by default
```

The seam audit is written for whoever reads the ledger later. An agent that just received
a healed answer cannot see it, so it cannot decide to re-ask. With this on, every healed
stream ends with one extra SSE event **after** `message_stop`:

```
event: llmbridge_seam
data: {"type":"llmbridge_seam","generation_id":"gen_...","from_model":"nemotron",
       "to_model":"qwen3:8b","seam_at":480,"audited":true,"suspect":false,
       "duplicated":false,"restarted":false,"overlap_chars":9,"overlap_ratio":0.056,
       "hinted":false}
```

A tool-call heal adds `tool_call`, `tool_finished` and marks the seam `suspect` when the
fallback did not finish the call it was asked to. It is placed after `message_stop` so a
strict Anthropic parser that stops reading there is unaffected; a client that opts in reads
one more event. A non-streaming auto-resume reports the same verdict as a header,
`X-LLMBridge-Seam: clean|suspect`. The event never changes the answer; it tells the client
the answer was stitched and how well.

### Moving interrupted work to another bridge

Two bridges that share a secret can hand work to each other without sharing a ledger.

```bash
export LLMBRIDGE_HANDOFF_SECRET="<16+ chars, same on both bridges>"   # or LLMBRIDGE_API_KEY
```

With a secret set, `GET /v1/interrupted` and `GET /v1/generations/{id}` (for an
`interrupted` record with saved text) include a `handoff_token`: the replayable request,
the partial output, the source model and the fingerprint, packed and signed with
HMAC-SHA256. Any bridge with the same secret accepts it:

```bash
# on bridge A
llmbridge handoff gen_1a2b3c4d5e6f                      # prints the token
# on bridge B (or from A, pointing at B)
llmbridge handoff gen_1a2b3c4d5e6f --to http://bridge-b:8787 --model <alias-on-B>
# the HTTP form
curl -X POST http://bridge-b:8787/v1/interrupted/resume \
     -d '{"handoff_token": "<token>", "model": "<alias-on-B>"}'
```

Bridge B rebuilds the conversation with the partial text as an assistant prefill, runs it
on the named route, and records the child under the original request's fingerprint with a
`handoff` event naming the source generation and model, so lineage is traceable across the
two ledgers. Nothing on B is touched besides that child. The source record on A stays
`interrupted` until something on A settles it: re-run `llmbridge status` there and
`DELETE /v1/interrupted` or fork it once B has finished, or let auto-resume on A pick it
up if the same request is ever re-sent there.

Tokens expire after `LLMBRIDGE_HANDOFF_TTL_SECONDS` (default 24 h), are rejected if
tampered with or over 2 MiB, and are only minted and accepted when a secret of at least 16
characters is configured. A token is a bearer credential for one piece of work: it carries
the prompt in clear, so send it over the same channel you would send the prompt.

### Automatic resume on re-send

By default a client that re-sends a request whose earlier run was interrupted pays for the
whole answer again: the new run starts from scratch, and the old record is merely marked
`resumed` once it completes. Opt in to have the bridge pick the work up instead:

```bash
LLMBRIDGE_AUTO_RESUME=1    # off by default
```

When a `POST /v1/messages` arrives whose **fingerprint** matches an `interrupted` record
that saved partial text, the bridge (after joining any identical request already in flight,
so two simultaneous re-sends share one resume when coalescing is on):

1. Rebuilds the request with the saved text appended as an assistant turn, exactly as
   `fork` does, and opens the route (primary, then its `fallbacks`) **before sending any
   byte** to the client. If no route answers, the ordinary request path runs and reports
   the failure the usual way; nothing is settled.
2. Answers the client with one whole response. A streaming client gets one
   `message_start`, the saved text as the first `content_block_delta`, then the
   continuation in the same stream. A non-streaming client gets the saved text and the
   continuation joined in the first text block, with `usage.output_tokens` summed.
3. Records the continuation as a child of the interrupted record (`fork` event, parent
   set), with a `heal` event whose `reason` is `auto_resume` and a scored `heal_seam`, so
   `llmbridge status`, `stats`, `tree` and `audit` all see it. The source is marked
   `resumed`. The reply carries `X-LLMBridge-Generation` (the child) and
   `X-LLMBridge-Resumed` (the source).

The match is the request's content fingerprint, the same identity coalescing and
settlement use: a byte-identical re-send from a client timeout retry, an agent loop, or a
second bridge sharing the ledger qualifies. A different request that shares the prompt
never does, and an interrupted record with no saved text (a `429` before the first token)
runs fresh. A Claude Code `continue` is a *different* request, with the partial text
already in its history, so it is routed normally rather than resumed.

The default is off on purpose. The bridge cannot tell a timeout retry from a client that
cancelled on purpose and wants a fresh answer; turn this on where retries are the common
case. The join is audited like any heal: a continuation that repeats or restarts the saved
text shows up as `suspect`, and `llmbridge replay <id>` re-checks it.

---

## Compatibility & Limitations

| Feature | Support |
|---------|---------|
| Messages | System, user/assistant text, images (URL/base64), documents |
| Documents | Text, base64, public HTTP(S) URLs, local files, Anthropic Files API |
| Tools | Definitions, choices, results, upstream tool calls → function calling |
| Self-healing | Text and tool-call streams; fallbacks optionally ranked by measured seam quality; optional first-token race against the first fallback |
| Streaming | OpenAI SSE → Anthropic events (`ping`, `input_tokens`, `content_filter`→`end_turn`) |
| Token counting | Route-configured tiktoken or char-based fallback (estimate only) |
| Thinking | Only on routes with `reasoning_effort`; `budget_tokens` not preserved |
| Caching | Off by default (`cache_control="strip"`); `"passthrough"` forwards breakpoints |
| Coalescing | Non-streaming only, opt-in; identity is the full request fingerprint |
| Auto-resume | Opt-in; a re-sent interrupted request continues from its saved text, streaming or not |

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

CI runs the same suite under pytest with coverage (the build fails below 85%):

```bash
pip install -e ".[test]"
python -m pytest --cov=llmbridge --cov-report=term-missing
```

---

## License

Apache 2.0 — see [LICENSE](LICENSE).