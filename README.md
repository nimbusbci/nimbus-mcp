# nimbus-mcp

<!-- mcp-name: io.github.nimbusbci/nimbus-mcp -->

MCP server that lets AI agents (Claude Code, Cursor, Claude Desktop) build, validate,
run, and analyze Nimbus BCI pipelines — upload their own EEG data, persist pipelines
into studio projects, run multi-configuration experiment campaigns, watch live EEG
sessions, and (explicitly confirmed) start live streaming — through your **local**
Nimbus backend **or the hosted deployment, with one `nimbus-mcp login`**.

## Install

```bash
pip install nimbus-mcp   # or: uvx nimbus-mcp
```

(Also installable from the repo: `pip install -e nimbus-studio/mcp`.)

## Authentication

Three ways to give the server a credential — tried in this order at startup:

1. **`nimbus-mcp login` (recommended — hosted API, no token pasting).** A
   device-code login: the command prints a URL and an 8-character code, opens
   your browser, you approve in Nimbus Studio, and the minted token is stored
   at `~/.nimbus/credentials.json` (0600) and picked up automatically on every
   future start.

   ```bash
   nimbus-mcp login     # options: --api-url URL, --ttl-days 7..90, --name NAME
   nimbus-mcp status    # doctor: credential source, plan, quota, days-to-expiry, live probe
   nimbus-mcp logout    # remove the stored credential
   ```

   After a login, MCP client configs need no secret at all:

   ```json
   {
     "mcpServers": {
       "nimbus": {
         "command": "uvx",
         "args": ["nimbus-mcp"],
         "env": { "NIMBUS_API_URL": "https://nimbus-studio.fly.dev" }
       }
     }
   }
   ```

2. **Desktop app — zero config.** Just have the Nimbus Studio desktop app
   running: its local key file is auto-discovered (macOS `~/Library/Application
   Support/Nimbus Studio/mcp-key.json`, Linux `~/.config/Nimbus Studio/…`,
   Windows `%APPDATA%\Nimbus Studio\…`) and the server talks to the local
   backend at `http://127.0.0.1:8080`. Nothing to paste or configure.

3. **Environment variables (advanced / CI).** `NIMBUS_TOKEN` (a hosted API
   token `nimb_…` minted in Nimbus Studio → Account → API tokens) or
   `NIMBUS_TOKEN_FILE` (a 0600 JSON file `{"token": "…"}` — keeps the secret
   out of process env and MCP configs), or the local pair
   `NIMBUS_MCP_KEY` / `NIMBUS_MCP_KEY_FILE` (must match `MCP_LOCAL_KEY` on a
   local backend — see below). Explicit env always beats files on disk.

**No credential anywhere?** The server still starts — in **setup mode**. Every
tool call returns `{ok: false, setupRequired: true, message, options}` with the
three paths above, so your agent walks you through onboarding instead of the
server crashing. A hosted token rejected mid-session (expired or revoked)
returns the same shape, including how many days ago it expired and a
`nimbus-mcp login` first option.

### Checking who you are

```
whoami() → {userId, email, plan: {isPro, pioneerAccess},
            freeRuns: {monthlyLimit, remaining},
            token: {name, expiresAt, daysLeft} | null,   # hosted-token mode only
            source}                                      # store | env | token_file | …
```

Call `whoami()` from the agent to see the account, plan, this month's free-run
quota, and (in token mode) the token's days-to-expiry; `nimbus-mcp status` is
the terminal equivalent with a live backend probe.

### What a hosted token means

- **The token IS you.** Requests run under your account: executions appear in your
  studio history and **your plan's quotas and limits apply** — there is no separate
  agent allowance. When the free monthly quota is exhausted, run errors carry the
  upgrade link `https://studio.nimbusbci.com/pricing?reason=mcp-quota`.
- **CPU-only in v0.4.** Token-authenticated runs do not hydrate cloud GPUs.
- **Rotation.** Tokens live at most 90 days (30 by default). Plan changes are
  snapshotted at mint time — after an upgrade, re-login (or revoke and re-create
  the token) to pick up the new plan. An expired token surfaces as setup
  guidance with the day count, not a dead end.

## Requirements (local mode)

- A Nimbus backend running locally: the **desktop app**, or the dev server
  (`cd nimbus-studio/backend-py && python -m nimbus_backend.server.app`) with `DEBUG=1`.
- The backend started with `MCP_LOCAL_KEY=<some-secret>` (never set this on Fly — it is
  refused there).
- Desktop app users: open **Settings → MCP & Agents** — no manual key setup (the app
  creates the key, injects it into its backend, and hands you copy-ready configs).

## Configure the backend

Desktop/dev env (e.g. `backend-py/data/.env` or the dev shell):

```bash
MCP_LOCAL_KEY=choose-a-long-random-string
MCP_LOCAL_USER_ID=user_your_clerk_user_id
DEBUG=1   # dev server only; the desktop app qualifies automatically
```

`MCP_LOCAL_USER_ID` sets the principal the MCP key authenticates as. Set it to your
own Clerk user id (`user_…`) so everything the agent creates — projects, saved
pipelines, executions — appears in your studio UI as yours. Pick one owner and stick
with it: switching the id mid-life splits ownership of agent-created work across two
principals, and neither identity then sees the whole history.

Watchdog default: streaming sessions started through MCP are auto-stopped after
15 minutes with no one watching (every `stream_status` / `get_live_session` poll
resets the timer). Pass `idle_timeout_sec=0` to `start_stream` to disable it for a
session.

When enabling `MCP_LOCAL_KEY` on a machine connected to an untrusted network, also set
`HOST=127.0.0.1` on the backend. The `0.0.0.0` default (`settings.host`) applies to the
bare dev server (`python -m nimbus_backend.server.app`), so with it the key would
otherwise be accepted from the LAN; `backend-py/scripts/run_server.py` already defaults
to `127.0.0.1`, and the desktop app pins loopback itself.

## Run the server

```bash
cd nimbus-studio/mcp
python -m venv .venv && source .venv/bin/activate
pip install -e ".[test]"
NIMBUS_MCP_KEY=choose-a-long-random-string python -m nimbus_mcp
```

Env vars: `NIMBUS_API_URL` (default `http://127.0.0.1:8080`, or the store's
`api_url` after a login), `NIMBUS_TOKEN` / `NIMBUS_TOKEN_FILE` (hosted API
token — see **Authentication**), `NIMBUS_MCP_KEY` (must match `MCP_LOCAL_KEY`),
`NIMBUS_MCP_KEY_FILE` (path to a 0600 JSON file `{"key": "…"}` — the desktop
app's one-click MCP setup writes it; consulted only when `NIMBUS_MCP_KEY` is
unset), `NIMBUS_EXPORT_DIR` (default `~/nimbus-exports`). With none of the
token/key vars set, the login store and then the desktop key file are
auto-discovered; with nothing found, the server runs in setup mode (every tool
returns onboarding guidance).

## Hosted gateway (streamable HTTP)

`nimbus-mcp serve` runs the same 32 tools over streamable HTTP instead of
stdio — for remote MCP clients, registries (Smithery lists URL-based
servers), and browser-side clients:

```bash
nimbus-mcp serve --host 0.0.0.0 --port 8080   # env: NIMBUS_MCP_HOST/PORT/PATH
```

The gateway holds **no** credential itself: each request's Nimbus API token
arrives as a header, so one shared deployment serves many users as
themselves. Calls without a header get setup guidance (add the header, or
install locally via `uvx`).

```json
{
  "mcpServers": {
    "nimbus": {
      "type": "http",
      "url": "https://nimbus-mcp.fly.dev/mcp",
      "headers": { "X-Nimbus-Token": "nimb_…  (Account → API tokens)" }
    }
  }
}
```

(A gateway started with `NIMBUS_TOKEN` in the environment uses it as the
fallback for headerless calls — single-tenant self-hosting.)

## Claude Code

```bash
# --env flags go BEFORE the -- separator (everything after it is the literal
# server command, so the after-form would feed --env to python/uvx):
claude mcp add nimbus --env NIMBUS_MCP_KEY=choose-a-long-random-string \
  -- <path-to-mcp-venv>/bin/python -m nimbus_mcp
```

## Cursor / Claude Desktop (stdio)

```json
{
  "mcpServers": {
    "nimbus": {
      "command": "<path-to-mcp-venv>/bin/python",
      "args": ["-m", "nimbus_mcp"],
      "env": { "NIMBUS_MCP_KEY": "choose-a-long-random-string" }
    }
  }
}
```

## Tools (32)

Auth: `whoami` (account, plan, quota, token expiry)
Discovery: `list_nodes`, `get_node_schema`, `list_templates`, `get_template`, `list_datasets`, `get_leaderboard`
Data: `upload_data`
Inspect: `inspect_dataset`, `inspect_file` (EDA: channels, class balance, band powers, PSD)
Build: `validate_pipeline`, `validate_node_config`
Run: `run_pipeline` (non-blocking), `get_execution`, `list_executions`, `get_results`, `cancel_execution`
Campaigns: `run_experiment` (non-blocking, 1-25 paced runs), `get_experiment`
Artifacts: `list_artifacts`, `download_artifact`, `export_python`
Live: `list_devices`, `test_device`, `start_stream` (needs `confirm=true`),
`stream_status`, `get_live_session`, `stop_stream`
Projects: `create_project`, `list_projects`, `save_pipeline`, `load_pipeline`

Not sure which pipeline to build? `get_leaderboard()` ranks benchmarked pipelines
per dataset (`meanAccuracyPct` desc, 95% CI) under the canonical `within_session`
protocol — agents pick templates by ranking there and pull the winner with
`get_template(pipelineId)`.

## Look at your data first

Before building any pipeline, agents can *see* the data: channels, sampling
rate, trial/class balance, per-channel µV stats, canonical band powers, and a
PSD overview — for a public dataset or a file on disk.

> "Inspect BNCI2014_001 subject S01 before we pick a pipeline."

```python
inspect_dataset(dataset="BNCI2014_001", subject="S01", mode="all")
# → {channels: {count: 22, names: [...], flatlined: []}, samplingRate: 250,
#    trials: {count: 288, classLabels: [...], classCounts: {...}},
#    channelStats: [...], bandPowers: {...}, psd: {...},
#    computedFrom: {nSamples: ..., sampleStrategy: "stratified_sample_seed42_4_of_20"}}
```

`subject` is required (e.g. `"S01"`; get the list via `list_datasets`) and a
comma-list like `"S01,S03"` loads a cohort; `mode` is `training |
evaluation | all`.

The same works for files: `inspect_file` picks the source from the path
shape. An ABSOLUTE path reads the file from disk (local backend only —
desktop app / MCP local mode: no upload step, the data never leaves the
machine); a RELATIVE path — the one `upload_data` returns — describes the
uploaded file on ANY backend (hosted or local):

> "Look at ~/recordings/session-01.edf and tell me if the montage is sane."

```python
inspect_file(path="/Users/you/recordings/session-01.edf")  # absolute → local file
inspect_file(path="uploads/<user>/session-01.edf")         # upload_data path → upload
```

On a hosted backend absolute paths are refused — `inspect_file` then returns
guidance (`upload_data` the file and pass the returned path back, or point
the server at a local backend) instead of a dead end.

Why inspect first: **class balance drives stratification** (imbalanced classes
skew accuracy), and **flatlined channels mean a montage/reference problem**
worth fixing before training. And a units caveat: the loader **assumes
volts** — a µV-native CSV reads 1e6x too large; set `unitsScale` (e.g.
`1e-6` with `units: "uV"`) in the pipeline's `custom_data` config when
needed.

## Uploading data

Bring your own recordings instead of (or alongside) the public datasets.

> "I have a `.edf` recording at `~/recordings/session-01.edf` — upload it and build
> a pipeline around it."

The agent calls `upload_data(file_path=…)`, which registers the file with the
backend and returns the stored path; that path goes into a `custom_data` node's
config (`{"filePath": "<path>", "format": "edf", …}`) for `validate_pipeline` /
`run_pipeline` / `run_experiment`. For plain CSV/TSV/TXT without embedded
metadata, pass `sampling_rate` (Hz) — the backend silently assumes 250 Hz
otherwise; `format` overrides extension-based detection.

## Experiment campaigns

One `run_experiment` call = a paced sweep of 1-25 pipelines (at most 2 training
runs in flight) with aggregated metrics, instead of the agent babysitting 25
individual `run_pipeline` polls.

> "Compare CSP-LDA vs EEGNet on BNCI2014_001 across subjects 1-3."

The agent builds six train graphs, calls
`run_experiment(runs=[{name: "csp-lda-s1", train_graph: …}, …])`, gets an
`experimentId` back immediately, then polls `get_experiment(experiment_id)` until
status is `completed` — per-run status and, at the end,
`aggregates` like `{"kappa": {"mean": 0.61, "std": 0.08, "best": {name, value}}}`
(mean/std/best over completed runs only).

## Working with projects

Agent builds, human inspects. Pipelines the agent saves land in real studio
projects, so you can open the canvas and see exactly what ran.

> "Save this pipeline as a project called 'motor-imagery-baseline' — I'll review
> it in the studio."

`create_project(name)` makes the container, `save_pipeline(project_id,
train_graph)` writes the graph (layout auto-generated, revision conflicts retried
once) and `load_pipeline(project_id)` reads it back for editing or re-running.
With `MCP_LOCAL_USER_ID` set to your user id, the project shows up in **your**
studio project list.

## Watching a live session

While a streaming session runs, the agent can watch its telemetry and tell you
what it sees.

> "Watch my focus session and tell me when signal quality drops."

The agent polls `get_live_session(session_id)` — latest prediction, the recent
window, signal quality (`meanChannelQuality`, `snrDb`, `artifactProbability`) and
running stats — and warns when quality degrades. Each poll also resets the idle
watchdog, so a session under active watch is never auto-stopped; an abandoned one
is shut down after 15 minutes.

## Safety

`start_stream` refuses to run without `confirm=true` — it connects an EEG device and
starts a live session on a human. The `X-MCP-Key` path is machine-local only
(never accepted on Fly deployments); hosted mode authenticates with a personal
`Authorization: Bearer` token from `nimbus-mcp login` (see **Authentication**
above). Sessions started via MCP are stopped automatically after 15 idle
minutes (see the watchdog note above).
