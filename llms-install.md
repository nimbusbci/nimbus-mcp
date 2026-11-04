# Installing nimbus-mcp (for AI agents, e.g. Cline)

1. Requirements: Python 3.10+ and [`uv`](https://docs.astral.sh/uv/) (or pip).
   `uvx nimbus-mcp` starts the stdio MCP server directly; no separate install is needed.
   With pip instead: `pip install nimbus-mcp`, then run `nimbus-mcp` (or `python -m nimbus_mcp`).
2. Authenticate (required for the stdio config below, which talks to the hosted API): ask the
   user to run `uvx nimbus-mcp login` (or `nimbus-mcp login` after a pip install) in their own
   terminal. It prints a URL and an 8-character code that the user approves in the browser. The
   credential is stored locally. Never ask the user to paste a token into chat.
   Check login state any time with `uvx nimbus-mcp status` (credential source, plan, quota,
   days-to-expiry, live probe).
   Skipping login is only possible on the README's Local Free tier, which needs the Nimbus Studio
   desktop app (or a local dev backend on `http://127.0.0.1:8080`) running on the same machine.
3. Add the server to Cline's MCP settings (`cline_mcp_settings.json`):

   ```json
   {
     "mcpServers": {
       "nimbus": {
         "command": "uvx",
         "args": ["nimbus-mcp"],
         "env": { "NIMBUS_API_URL": "https://nimbus-studio.fly.dev" },
         "disabled": false,
         "autoApprove": []
       }
     }
   }
   ```

   Alternative with no local install: the hosted gateway (streamable HTTP). The user signs in
   with OAuth in the browser on first use; no `login` step is needed.

   ```json
   {
     "mcpServers": {
       "nimbus-gateway": {
         "type": "streamableHttp",
         "url": "https://nimbus-mcp.fly.dev/mcp",
         "disabled": false,
         "autoApprove": []
       }
     }
   }
   ```

   For headless use (no browser), add an API key header to that entry:
   `"headers": { "Authorization": "Bearer <API key>" }`.
4. Verify: call `account.whoami`, then `catalog.datasets`. You should see the public MOABB/BNCI
   datasets. If you get `No Nimbus backend is running...`, the server has no credential and no
   local backend: have the user run `uvx nimbus-mcp login` (step 2) or start the Nimbus Studio
   desktop app, then restart the server and retry. If you get `Login required: the hosted Nimbus
   API ... rejected this request (HTTP 401)` (the config sets `NIMBUS_API_URL` to the hosted API
   but no credential is present), the fix is the same login: run `uvx nimbus-mcp login` or set
   `NIMBUS_TOKEN`, then restart the server and retry. `uvx nimbus-mcp status` shows what it found.
5. Safety: live-streaming and hardware tools act on a real person and device. Do not start them
   unless the user explicitly asks.

Docs: https://docs.nimbusbci.com · License: the MCP client is Apache-2.0; the Nimbus engine and
hosted services are proprietary (see README).
