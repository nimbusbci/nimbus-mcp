# nimbus-mcp privacy statement

_Last updated: 2026-10-19._

## Local Free needs no account

nimbus-mcp runs in a free local mode with **no Nimbus Studio login**:
templates, your own data, local execution, plots/stats, BIDS export and
local-only storage all work anonymously against the local backend (the
Nimbus Studio desktop app or a local dev backend). Login is requested only
at the first _cloud_ feature (a leaderboard submission, verified or
cross-machine experiment history, hosted compute, cloud sync,
collaboration) — as an explicit prompt naming the reason, never as a
blanket gate.

## What telemetry collects (only with your opt-in)

Telemetry is **off until you turn it on** (`nimbus-mcp telemetry on`, or
`NIMBUS_MCP_TELEMETRY=on`). You are asked once, via a notice attached to
the first tool response of a fresh install. With consent, the client
sends these events to the existing PostHog 'Nimbus' product project with
an `mcp_` prefix:

| Event | When | Properties |
| --- | --- | --- |
| `mcp_first_run` | first event after opt-in | base properties only |
| `mcp_first_local_result` | first successful local result | base + mode |
| `mcp_tool_completed` | each tool call | tool name, mode, success/failure, error **code**, duration bucket, client version |
| `mcp_login_prompt_shown` | a cloud feature asks for login | trigger, mode |
| `mcp_login_completed` | `nimbus-mcp login` succeeds | trigger |
| `mcp_sync_opt_in` | (reserved for the future sync opt-in) | — |
| `mcp_version_drift_warned` | a newer client release exists | current/latest versions, mode |

Base properties: client version, OS family, Python minor version.

## What is never collected

- Dataset contents, signal data, or derived features
- File paths or file names; upload metadata
- Your code (pipeline graphs, python_model source, scripts)
- Free-text error messages — errors travel as stable codes only
- IP address / geolocation: the IP is sent empty (`$ip`) with geoip
  disabled, and no person profile is created
  (`$process_person_profile: false`); the receiving PostHog project
  (275296) also has server-side IP discard enabled, so the connecting IP
  is not stored either

Identity is a random install UUID stored in
`~/.nimbus/telemetry-state.json`. It is **not** linked to your Nimbus
account; linkage could only ever happen through a future explicit sync
opt-in. Until then, telemetry cannot be attributed to a person.

## Opting out

- `nimbus-mcp telemetry off` — revokes consent anytime; with telemetry
  off the client makes **zero** telemetry network calls.
- `NIMBUS_MCP_TELEMETRY=off` — headless/CI control (overrides the stored
  choice); `=on` opts in the same way.
- `nimbus-mcp telemetry status` shows the current state.
- Everything local works fully with telemetry off.

## Version check

At startup the client quietly checks PyPI for the latest `nimbus-mcp`
release (2 s timeout, cached 24 h in `~/.nimbus/version-check.json`). If
you are behind, one tool response carries a `versionNotice`; it is
silent when current or offline and never blocks local work. Set `NIMBUS_MCP_VERSION_CHECK=off` to disable it entirely (no PyPI
request). The version check is independent of the telemetry setting.

## Contact

Nimbus BCI — privacy questions via the Nimbus Studio support channel.
