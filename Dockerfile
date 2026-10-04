# Hosted nimbus-mcp gateway (streamable HTTP, per-request credentials).
# Deployed at https://nimbus-mcp.fly.dev/mcp — see fly.toml next to this file.
FROM python:3.12-slim

# Non-root runtime user; exports land in /tmp (gateway mode never writes user
# artifacts — export_dir only matters for stdio installs on real machines).
RUN useradd --create-home --shell /usr/sbin/nologin nimbus
WORKDIR /app

COPY pyproject.toml README.md LICENSE.txt ./
COPY src ./src
RUN pip install --no-cache-dir . && rm -rf src pyproject.toml README.md

USER nimbus
ENV NIMBUS_MCP_HOST=0.0.0.0 \
    NIMBUS_MCP_PORT=8080 \
    NIMBUS_MCP_PATH=/mcp \
    NIMBUS_EXPORT_DIR=/tmp/nimbus-exports

EXPOSE 8080
CMD ["nimbus-mcp", "serve"]
