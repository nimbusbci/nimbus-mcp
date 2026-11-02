# MCP telemetry: PostHog queries

Project 275296 (`us.i.posthog.com`). All events are `mcp_*` and carry:

| property | meaning |
|---|---|
| `mcp_version` (`client_version` legacy) | nimbus-mcp package version |
| `transport` | `local` (stdio install) or `gateway` (hosted HTTP) |
| `mcp_client_name` / `mcp_client_version` | the MCP client's `clientInfo` (e.g. `claude-ai`, `cursor-vscode`) |
| `os_family` | `darwin` / `linux` / `windows` (server host OS on the gateway) |
| `install_id` | = `distinct_id`; random per-install UUID locally, HMAC-SHA256(`NIMBUS_MCP_SUBJECT_HASH_KEY`, sub) on the gateway (`gateway_anonymous` if the key is unset) |
| `session_id` | per stdio process locally; `Mcp-Session-Id` on the gateway |
| `is_internal` | `NIMBUS_MCP_INTERNAL=1` locally; subject/email/hash in `NIMBUS_MCP_INTERNAL_SUBJECTS` on the gateway |

IPs are discarded (`$ip=""`, `$geoip_disable=true`) and no person profiles are created.

## Enabling

* **Local:** opt-in only — `nimbus-mcp telemetry on` (or `NIMBUS_MCP_TELEMETRY=on`). Off by default.
* **Gateway:** off unless the Fly app sets `NIMBUS_MCP_GATEWAY_TELEMETRY=on`.
  `NIMBUS_MCP_TELEMETRY=off` is a hard kill switch on both.
  Set `NIMBUS_MCP_SUBJECT_HASH_KEY` (random secret, e.g. `openssl rand -hex 32`) for
  per-subject ids; without it subject-level metrics are disabled (all events use
  `gateway_anonymous`). Rotating the key resets subject ids.
  Internal accounts: `NIMBUS_MCP_INTERNAL_SUBJECTS="user_abc,dev@nimbusbci.com,sub_…"`.
  CI and the test suite set `NIMBUS_MCP_INTERNAL=1`.

## Gateway: distinct OAuth subjects per week

```sql
SELECT toStartOfWeek(timestamp) AS week, uniq(distinct_id) AS subjects
FROM events
WHERE event LIKE 'mcp_%'
  AND properties.transport = 'gateway'
  AND coalesce(properties.is_internal, false) = false
  AND distinct_id != 'gateway_anonymous'
  AND timestamp > now() - INTERVAL 12 WEEK
GROUP BY week ORDER BY week
```

## Gateway: sessions per day

`mcp_session_started` is deduped per gateway process, so restarts or multiple
machines can re-emit it for one live session: count `uniq(properties.session_id)`
for an exact figure.

```sql
SELECT toDate(timestamp) AS day, uniq(properties.session_id) AS sessions, uniq(distinct_id) AS subjects
FROM events
WHERE event = 'mcp_session_started'
  AND properties.transport = 'gateway'
  AND coalesce(properties.is_internal, false) = false
  AND timestamp > now() - INTERVAL 30 DAY
GROUP BY day ORDER BY day
```

## Tool calls per day (by transport)

```sql
SELECT toDate(timestamp) AS day, properties.transport AS transport,
       count() AS calls, countIf(properties.success = true) AS ok
FROM events
WHERE event = 'mcp_tool_completed'
  AND coalesce(properties.is_internal, false) = false
  AND timestamp > now() - INTERVAL 30 DAY
GROUP BY day, transport ORDER BY day, transport
```

## Top tools / clients

```sql
SELECT properties.tool AS tool, properties.mcp_client_name AS client, count() AS calls
FROM events
WHERE event = 'mcp_tool_completed' AND coalesce(properties.is_internal, false) = false
  AND timestamp > now() - INTERVAL 30 DAY
GROUP BY tool, client ORDER BY calls DESC LIMIT 50
```

## Version adoption (local installs)

```sql
SELECT properties.mcp_version AS version, uniq(distinct_id) AS installs
FROM events
WHERE event LIKE 'mcp_%' AND properties.transport = 'local' AND coalesce(properties.is_internal, false) = false
  AND timestamp > now() - INTERVAL 30 DAY
GROUP BY version ORDER BY installs DESC
```

Events sent before this change have no `transport`/`is_internal`; treat
`transport IS NULL` as `local` and filter old dev noise by date if needed.
