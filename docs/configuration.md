# Configuration reference

Every setting of [salome-agent](../README.md). To get started you only need three of them: copy
`config.example.json` to `config.json` and set `llm.base_url`, `llm.api_key` and `llm.model`.
The rest have working values in the example.

- [`llm`: the model endpoint](#llm--the-model-endpoint)
- [`policy`: how a run behaves](#policy--how-a-run-behaves)
- [Other keys](#other-keys)

If you use the salome-agent-gui plugin, its settings drawer edits this same format, saved in
`~/.config/salome/salome-agent.json`.

## The full file

```json
{
  "llm": {
    "base_url": "http://your-endpoint:port/v1",
    "api_key": "sk-...",
    "model": "auto",
    "ssl_cert_file": "",
    "temperature": 0,
    "timeout": 120,
    "json_retries": 2,
    "vision_fix_retries": 2,
    "loop_retries": 3,
    "text_tool_calls": true,
    "max_retries": 2,
    "reasoning_budget": 20000,
    "carry_reasoning": true,
    "extra_headers": {},
    "extra_body": {},
    "module_extra_body": {},
    "modules": {}
  },
  "salome": {
    "root": "/path/to/your/SALOME"
  },
  "policy": {
    "tool_iteration_cap": 12,
    "plan_detail": "balanced",
    "tool_selection": "hints",
    "verify_cadence": "final",
    "on_stage_failure": "abort"
  },
  "work_dir": "~/.local/share/salome-agent/runs"
}
```

Required keys: `llm.base_url` and `llm.model`. Missing either is a hard error. An unknown key in the `llm` block (a typo such as `extra-body`) is logged as a warning and ignored.

## `llm` — the model endpoint

Any OpenAI-compatible `/chat/completions` endpoint. Requests are streamed (SSE).

| Key | Default | Meaning |
|-----|---------|---------|
| `base_url` | — (required) | Endpoint base URL, up to and including `/v1`. |
| `model` | — (required) | Model name sent in every request. |
| `api_key` | `""` | Sent as `Authorization: Bearer <key>`. Leave empty for an endpoint without auth. |
| `ssl_cert_file` | `""` | Path to a CA bundle for an HTTPS endpoint with a private CA. `~` is expanded; a missing file fails at startup. Empty verifies against the system trust store. |
| `temperature` | `0` | Sampling temperature. |
| `timeout` | `120` | Seconds. Total wall-clock budget for one streamed LLM call; a generation still running at the limit (a runaway reasoning loop) is cut and not retried. The gap between two streamed chunks is capped separately at 60 s. `--timeout` overrides it. |
| `json_retries` | `2` | When the planner's answer does not parse as JSON, the answer and the exact parser error are fed back and the planner is re-asked this many times before the run fails. |
| `vision_fix_retries` | `2` | When the image-analysis agent's spec fails the mechanical checks (an outline that does not close, an arc end off its radius, a hole outside the outline, a dangling reference), the errors are fed back and it is re-asked this many times; errors still left go to the planner and geom as `conflicts`. |
| `max_retries` | `2` | Retries for transient failures — connection drop, truncated stream, HTTP 429 (honouring `Retry-After`), fast 502/503/504, a mid-stream error frame for a server-side failure (5xx/429 status, or type `server_error`/`api_error`) — with exponential backoff. Raise it for a flaky free-tier endpoint. |
| `reasoning_budget` | `20000` | Specialist (tool-calling) calls only: approximate reasoning tokens (characters / 4) a model may spend before its first tool call. Past it the call is cut and re-asked once with the tail of its own reasoning and a "continue from it and act" nudge; a second cut in a row ends the stage. If the cut reasoning ends in a loop, it is handled as a loop (see `loop_retries`). `0` turns it off. |
| `loop_retries` | `3` | Every call (planner, specialists, vision): when the streamed reasoning or answer starts repeating itself (the same sentences over and over, as small models do), the call is cut at once. A planner or specialist call is then re-asked with its text from before the loop and one repeated sentence, never the loop itself, up to this many cuts in a row; past that the stage ends (the plan fails). A vision call that loops ends the run. |
| `text_tool_calls` | `true` | Specialist (tool-calling) calls only: a reply with no tool call is read for calls the model wrote as text (JSON, `<tool key="value"/>`, `<call:tool key=value/>`) and those run. Calls after other text in the reply (such as a result the model made up) are not run and the model is told so; a call whose exact text appears in a message the model was given (request, tool result) is never run. An echo the model reformats is not caught. `false` turns reading text calls off. Either way, the same reply with no tool call 3 times in a row ends the stage. |
| `carry_reasoning` | `true` | Specialist (tool-calling) calls only: each turn's reasoning (`reasoning_content`) stays in the stage's history, so after a tool result (a `calc`, say) the model resumes its own thinking instead of starting over. Costs input tokens for the rest of the stage. Set `false` for a provider that rejects `reasoning_content` in requests. |
| `extra_headers` | `{}` | Extra HTTP headers sent verbatim on every request. In a string value `{run_id}` is replaced with the run's id — for gateways that want a stable per-conversation token, e.g. `{"x-opencode-session": "{run_id}"}`. |
| `extra_body` | `{}` | Extra top-level fields merged into every request's JSON body, next to `model`/`messages`/`temperature`. Use this, not `extra_headers`, for provider body params such as `{"reasoning_effort": "high"}`; whether the provider honours a field is up to the provider. |
| `module_extra_body` | `{}` | `{"planner" \| "shaper" \| "geom" \| "smesh": {...}}` — a body sent *instead of* `extra_body` (full replacement, not a merge) on that module's calls. Lets an expensive knob sit on the planner only, e.g. `{"planner": {"reasoning_effort": "high"}}`, while the many specialist calls run without it. |
| `modules` | `{}` | `{"vision" \| "planner" \| "shaper" \| "geom" \| "smesh": {...}}` — sends that module's calls to their own endpoint and/or model. Each entry may set `base_url`, `api_key`, `model`, `ssl_cert_file`, `temperature`, `timeout`, `extra_headers`, `extra_body`; a key left out keeps the shared `llm` value — except with a `base_url` of its own, which never inherits the shared `api_key`, `ssl_cert_file`, `extra_headers` or `extra_body` (another host gets no credentials meant for the main one). Its `extra_body`, when set, wins over `module_extra_body`. E.g. a stronger drawing reader for images only: `{"vision": {"base_url": "https://generativelanguage.googleapis.com/v1beta/openai", "api_key": "...", "model": "gemini-3-pro-preview"}}`. |

## `policy` — how a run behaves

The first two keys have no built-in default — keep them in your config.

| Key | Default | Meaning |
|-----|---------|---------|
| `tool_iteration_cap` | — | Maximum LLM turns (each turn may carry several tool calls) a tool-calling specialist gets for one stage. |
| `on_stage_failure` | — | `abort` \| `continue`. What `--batch` does when a stage fails: stop the run, or skip the stage and go on. Interactive runs ask instead. `--on-failure` overrides it. |
| `plan_detail` | `balanced` | `coarse` \| `balanced` \| `fine`. How finely the planner splits a request into stages — `coarse` favours fewer, bigger stages; `fine` favours splitting a part or assembly into more stages. |
| `tool_selection` | `hints` | `hints` \| `all`. `hints`: the SHAPER / GEOM specialist is offered a small per-stage tool subset (planner hint + a core set) and widens it on demand via `load_tools`. `all`: the module's full registry on every call. `--tools` overrides it. |
| `verify_cadence` | `final` | How hard the SHAPER / GEOM specialist is pushed to `measure`-check where its shapes landed — see below. |

`policy.verify_cadence`:
- `none` — no self-check.
- `final` (default) — read the `shape_info` that rides on every result for free; call `measure` explicitly only when something looks off and once on the result shape(s) before `finish_stage`. If a move or boolean is still unmeasured at `finish_stage`, the loop asks for a measure once before accepting.
- `strict` — measure after every move or boolean. If the model does three in a row without one, the loop forces a measure before it can continue.

## Other keys

| Key | Default | Meaning |
|-----|---------|---------|
| `work_dir` | `.` | Where each run's folder (`<run_id>/` with `result.py`, `report.md`, logs) is written. `~` is expanded. `--work-dir` overrides it. |
| `salome.root` | — | Path to your SALOME install. The agent itself does not use it (it talks to an already-running agent server); the live tests and `scripts/regen-toplevel-api.py` read it when `$SALOME_ROOT` is not set. |

The salome-agent-gui plugin stores its own settings (`gui.theme`, `gui.batch`, …) in a `gui` section of the same file; the agent ignores that section.

`config.json` is gitignored — keep your credentials there, not in `config.example.json`.
