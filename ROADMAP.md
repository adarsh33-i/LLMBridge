# Novelty roadmap

What makes LLMBridge different is one idea: the bridge treats a broken answer as
work to be finished and measured, not as an error to be logged. Every item below
extends that idea. Items are ordered by novelty gained per line of code.

Status key: `have` = in the code today and committed; `have, uncommitted` = in the working tree,
not yet committed. Items under *To implement* are not started unless marked DONE.

## Already novel (document, do not rebuild)

| # | Capability | Status | Where |
|---|-----------|--------|-------|
| A | Failover inside the answer: partial text handed to the fallback as a prefill, continuation appended to the same stream, one `message_start` | have | `server._heal_stream`, `_relay_healed_stream` |
| B | Scored handoffs: every heal gets a `heal_seam` verdict (duplicated / restarted / suspect) | have | `server.score_seam`, `_audit_seam` |
| C | Digest that outlives the records: verdicts copied to `heal_audit.jsonl` before prune and on shutdown | have | `ledger.consolidate_audits` |
| D | Retention with stated exemptions: suspect seams and unfinished work survive the window under their own budgets | have | `ledger._prune_locked` |
| E | Rehearsed failovers: `dream` cuts a real answer and scores a model's continuation; `failover` ranks every fallback on recorded seams | have | `cli._dream`, `cli._failover` |
| F | Interrupted work as a first-class record: replayable request + partial text, forkable onto any model, source marked `resumed` not deleted | have | `ledger.interrupt`, `resume`, `fork_request` |
| G | Auto-resume on re-send: identical request continues from saved text, streaming or not, with lineage and a seam audit (opt-in) | have, uncommitted | `server._try_auto_resume`, `ledger.resumable` |
| H | Replay drift check: re-send a recorded request, score the new output against the saved one | have | `cli._replay` |
| I | Heal inside a tool call: completed calls released, cut call finished by the fallback under its own id, tool seam scored | have, uncommitted | `server._heal_tool_stream`, `_relay_healed_tool_stream` |
| J | Fallbacks ranked by evidence: heal candidates ordered by digest suspect rate per primary, order written to the heal event (opt-in) | have, uncommitted | `server.rank_heal_candidates`, `ledger.fallback_quality` |
| K | Speculative shadow stream: first fallback raced after N s without a first token, loser closed, `shadow` event recorded (opt-in) | have, uncommitted | `server._race_shadow_stream`, `_Racer` |
| L | Resume across gateways: signed handoff token on `GET /v1/interrupted` and `/v1/generations/{id}`, accepted by `POST /v1/interrupted/resume` on any bridge sharing the secret; `llmbridge handoff` | have, uncommitted | `server.mint_handoff_token`, `verify_handoff_token`, `cli._handoff` |
| M | Heal hint with measured effect: opt-in system addendum on heal requests, `hinted` on every seam, restart rate per model split by hinted in `stats` | have, uncommitted | `server._HEAL_HINT`, `ledger.hint_effect` |
| N | Seam verdict to the client: `llmbridge_seam` SSE event after `message_stop` (opt-in), `X-LLMBridge-Seam` header on non-stream resume | have, uncommitted | `server._emit_seam_event` |

Action for the chart and article: add a **Scored handoffs** row. It exists, it is
measured, and no competitor has it. Zero code.

## To implement

### 1. Resume across gateways (handoff token) — DONE (see row L)

`LLMBRIDGE_HANDOFF_SECRET` (or `LLMBRIDGE_API_KEY`, 16+ chars) enables it. With a
secret set, `GET /v1/interrupted` and `GET /v1/generations/{id}` carry a
`handoff_token` for interrupted work with saved text: replayable request,
partial output, source model, fingerprint, source generation id; HMAC-SHA256,
24 h TTL (`LLMBRIDGE_HANDOFF_TTL_SECONDS`), 2 MiB cap. `POST /v1/interrupted/resume`
on any bridge with the same secret accepts `{"handoff_token": ...}`, records the
child under the original fingerprint with a `handoff` event naming the remote
source, and runs it. `llmbridge handoff <id> [--to <url> --model <alias>]` prints
or sends the token.

- Chart row: *Work moves between machines* — competitors: no.
- Still open: the source on bridge A stays `interrupted` until settled there;
  B does not call back to A.

### 2. Fallback order learned from evidence — DONE (see row J)

Opt-in with `LLMBRIDGE_ADAPTIVE_FALLBACKS=1`. At heal time (`_rank_remaining`,
used by both the text and tool heal paths) the remaining candidates are ordered
by their digest suspect rate for this primary, lowest first. A candidate with
fewer than `LLMBRIDGE_ADAPTIVE_MIN_SAMPLES` (default 3) seams keeps its config
position and sorts after every trusted one. The order chosen is written into
the `heal` event (`order`: model, config position, samples, suspect rate,
trusted). `ledger.fallback_quality` reads the digest plus unconsolidated
records, cached 30 s and invalidated on every new seam audit.

- Chart row: *Fallbacks ranked by evidence* — competitors: static list.
- Verified 2026-11-04 by unit tests: a two-fallback chain heals onto the
  cleaner-scored route; stays in config order below the sample minimum and
  when the flag is off.
- Still open: `stats` does not yet summarise reorderings; the evidence is in
  each heal event.

### 3. Prefill-aware heal prompting — DONE (see row M)

`LLMBRIDGE_HEAL_HINT=1` adds a system addendum to text heals and auto-resumes
("continue from exactly where it stops, do not restate"). `hinted` is recorded
on the heal and seam events and kept in the digest; `ledger.hint_effect()`
reports restart rate per continuation model, hinted vs un-hinted, with the
delta, and `llmbridge stats` prints it once a model has both sides. Pre-hint
seams count as un-hinted. Tool-call heals are excluded (they always carry an
addendum).

- Still open: the README should state a measured delta once real traffic
  has produced one; today it shows the format only.

### 4. Heal inside a tool call — DONE (see row I)

On by default (`LLMBRIDGE_HEAL_TOOL_CALLS=1`). `tool_use` blocks are held until
their arguments parse and released whole on `content_block_stop`; a healthy
stream's visible content is unchanged. When the stream dies inside a call:
completed calls are released under their original ids; the fallback gets the
text as a prefill plus a system addendum naming the finished calls, the cut
tool, and its argument prefix; the fallback's blocks are re-indexed above every
index the client saw and carry the fallback's own ids; the message ends with
the fallback's stop reason. An empty argument string counts as *cut*, never
complete. The ledger gets a `heal` (`reason: tool_call`, which calls were cut
and issued) and a `heal_seam` with `tool_call`/`tool_finished`; a fallback that
called the wrong tool or none is `suspect`.

- Chart row: *Self-healing streams, including tool calls* — competitors: no.
- Verified live 2026-11-04: a mid-call death healed into one `tool_use` block
  (fallback id, `{"path": "/etc/hosts"}`), `stop_reason` `tool_use`, seam audit
  `tool_finished: true`. With `LLMBRIDGE_HEAL_TOOL_CALLS=0` the same death
  aborts with a typed error and an interrupted record.
- The article named this as a direction; this closes that loop.

### 5. Seam verdict delivered to the client — DONE (see row N)

`LLMBRIDGE_SEAM_EVENTS=1` appends one `llmbridge_seam` SSE event after
`message_stop` on every healed stream (text and tool-call), carrying the
generation id, both models, the seam position, and the audit verdict. Placed
after `message_stop` so strict parsers are unaffected. Non-stream auto-resume
reports `X-LLMBridge-Seam: clean|suspect` instead.

- Chart row: *Client told when a repair is suspect* — competitors: no.

### 6. Speculative shadow stream — DONE (see row K)

Opt-in with `LLMBRIDGE_SHADOW_AFTER_SECONDS=N`. The primary opens at once; if
no visible token arrives within N seconds the first fallback is opened too and
whichever yields a token first carries the stream. The loser is closed and its
output discarded (the client only ever sees one model's opening). A primary that
dies after the shadow opened hands the stream to the shadow immediately
(`reason: primary_failed`). A shadow winner that breaks later still heals onto
the next candidate. Every race writes a `shadow` event: winner, loser, reason,
waited_ms, `raced`, and `loser_sent` (the losing request reached its provider
and was billed), so the delay can be costed from the ledger.

- Chart row: *Slow first byte becomes a cost choice* — competitors: timeout only.
- Verified live 2026-11-04: a 3 s stall answered in 0.78 s with the delay at
  0.5; a fast primary never opened the shadow (`raced: false`).

### 7. Route-pair drift on real traffic

Run a sample of recorded requests against two routes in the background and
record the similarity distribution over time. Low agreement on a pair flags a
behavioural change before users notice. Extends H from one record to the fleet.

- Size: two days. Reuses `replay` scoring and the ledger as storage.

### 8. Budget guard that degrades instead of refusing

When a route's token spend crosses a threshold, move it down the fallback order
rather than returning an error. Requests still complete on cheaper models and
`stats` shows the shift.

- Size: two days. Spend is already recorded per generation.
- The article lists a budget guard as a direction; this is the non-blocking form.

### 9. Suspect seams as a test corpus

Export suspect-seam records as fixtures and run them through `dream` against
any new model before adding it as a fallback. The bridge generates its own
continuation benchmark from production traffic.

- Size: one day. Mostly CLI plumbing around existing `dream`.

### 10. Exact-fingerprint answer reuse

For `temperature: 0` requests, serve a later identical request from a completed
record within a freshness window, with a header saying so. Lives in the same
ledger as heals and forks and is auditable with `replay`.

- Size: one day. Extends coalescing (which today only joins in-flight requests).
- Lowest novelty of the list; semantic caches exist elsewhere.

## Not worth adding

A semantic cache or a cost dashboard. LiteLLM has both; adding them creates
chart rows where LLMBridge ties or loses and pulls the project away from its
one clear idea.

## Chart honesty fix

Change the LiteLLM cell under *Quota queue* from "limits only" to "retries and
cooldowns". The row still favours LLMBridge (the queue survives a restart
because it is the ledger), and readers who know LiteLLM will trust the rest of
the chart more.

## Suggested order

Done: 1, 2, 3, 4, 5, 6 (rows G and I–N). Remaining:

1. **9. Suspect seams as a test corpus** — build once the digest holds real suspect seams.
2. **7. Route-pair drift** — only if someone asks; overlaps with `replay`.
3. **8. Budget guard** and **10. exact-fingerprint cache** — not recommended; both
   invite comparison on LiteLLM's home ground.

The project's gap is now visibility, not features: add the *Scored handoffs* chart
row, fix the LiteLLM quota-cell wording, and show tool-call healing in the article.
