# Installing nimbus-mcp (for AI agents, e.g. Cline)

1. Requirements: Python 3.10+ and [`uv`](https://docs.astral.sh/uv/) (or pip).
   `uvx nimbus-mcp` starts the stdio MCP server directly; no separate install is needed.
   With pip instead: `pip install nimbus-mcp`, then run `nimbus-mcp` (or `python -m nimbus_mcp`).
2. Authenticate: ask the user to run `uvx nimbus-mcp login` (or `nimbus-mcp login` after a pip
   install) in their own terminal. It prints a URL and an 8-character code that the user approves in
   the browser. The credential is stored locally. Never ask the user to paste a token into chat.
3. Add the server to Cline's MCP settings (`cline_mcp_settings.json`):

   ```json
   {
     "mcpServers": {
       "nimbus": {
         "command": "uvx",
         "args": ["nimbus-mcp"],
         "disabled": false,
         "autoApprove": []
       }
     }
   }
   ```

   Alternative with no local install: the hosted server `https://nimbus-mcp.fly.dev/mcp`
   (streamable HTTP, OAuth sign-in).
4. Verify: call `account.whoami`, then `catalog.datasets`. You should see the public MOABB/BNCI
   datasets.
5. Safety: live-streaming and hardware tools act on a real person and device. Do not start them
   unless the user explicitly asks.

Docs: https://docs.nimbusbci.com · License: the MCP client is Apache-2.0; the Nimbus engine and
hosted services are proprietary (see README).
