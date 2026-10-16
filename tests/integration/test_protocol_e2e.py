"""Protocol builder acceptance E2E against a RUNNING local backend (Task 10).

Gate: NIMBUS_MCP_E2E=1 + DATABASE_URL + a worktree backend on NIMBUS_API_URL
run with the test Clerk issuer::

    CLERK_ISSUER=https://clerk.test
    CLERK_JWKS_URL=http://127.0.0.1:8902/jwks.json

(the module serves the JWKS on 8902 — the harness copied from
``test_calibration_e2e.py``). Run from the backend venv, which provides the
``websockets`` (WS stand-in) and ``sqlalchemy``/``psycopg2`` (donor row seed)
dependencies the MCP package itself does not ship.

This is the plan's acceptance criterion in miniature, entirely through the
``protocol.*`` MCP tools plus a Studio-app WS stand-in (deploy sessions boot
over the realtime WS; MCP is HTTP-only):

    protocol.save (mini protocol, 2 s blocks × 3, sessions=3 — durationSec
    only needs > 0; 2 s keeps one session ≈ 10 s so the donor + 3 sessions
    finish in ~1.5 min, where the spec's 5 s blocks would push past 4 min)
    → start_assignment(active) → one simulated session (WS stop; its
    recording becomes the yoked donor — the Task 7 backend E2E's
    recorded-simulated-session donor pattern, rehomed via the authenticated
    /api/recordings download + a seeded uploaded_files row)
    → start_assignment(sham, donor) → 3 simulated sessions, each closed by
    MCP protocol.stop (block-annotated recording + assignment progression)
    → protocol.export_stats: inferential block (3 sessions), non-empty
    base64 figure, CSV with 9 aggregate rows (3 sessions × 3 blocks).
"""

import asyncio
import base64
import json
import os
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import httpx
import pytest
import websockets
from fastmcp import Client

pytestmark = [
    pytest.mark.skipif(os.environ.get("NIMBUS_MCP_E2E") != "1", reason="NIMBUS_MCP_E2E!=1"),
    pytest.mark.skipif(
        not os.environ.get("DATABASE_URL", "").strip(), reason="DATABASE_URL not set"
    ),
]

_TEST_ISSUER = "https://clerk.test"
_JWKS_PORT = int(os.environ.get("NIMBUS_E2E_JWKS_PORT", "8902"))
_API = os.environ.get("NIMBUS_API_URL", "http://127.0.0.1:8080").rstrip("/")
_WS = _API.replace("http", "ws", 1)
_USER_ID = "user_test_local"

BLOCK_SEC = 2.0  # documented shrink (spec allows durationSec > 0)
SESSIONS = 3


def _mini_spec() -> dict:
    """Run-unique mini protocol: pre/train/post × 2 s, sham defaults to yoked."""
    return {
        "schemaVersion": 1,
        "id": f"mcp-e2e-nf-{uuid.uuid4().hex[:8]}",
        "name": "MCP E2E Mini NF",
        "indicator": {"name": "focus", "transform": "raw"},
        "armDefaults": {"active": {"feedbackSource": "live"}, "sham": {"feedbackSource": "yoked"}},
        "blocks": [
            {"id": "pre", "kind": "assessment", "durationSec": BLOCK_SEC},
            {"id": "train", "kind": "training", "durationSec": BLOCK_SEC, "feedback": True},
            {"id": "post", "kind": "assessment", "durationSec": BLOCK_SEC},
        ],
        "threshold": {
            "source": "baseline-quantile",
            "quantile": 0.7,
            "direction": "increase",
            "schedule": {
                "rule": "success-rate",
                "targetSuccessRate": 0.7,
                "windowBlocks": 2,
                "stepSize": 0.05,
                "minSteps": -2,
                "maxSteps": 4,
            },
        },
        "sessions": SESSIONS,
        # 0.0 floor keeps simulated data from tripping the early-close criteria.
        "breakCriteria": {"signalQualityFloor": 0.0},
    }


def _b64url_uint(val: int) -> str:
    raw = val.to_bytes((val.bit_length() + 7) // 8, "big") or b"\x00"
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _mint_pro_bearer() -> tuple[str, ThreadingHTTPServer]:
    """Local RS256 Pro JWT + the JWKS server the backend verifies against."""
    import jwt
    from cryptography.hazmat.primitives.asymmetric import rsa

    kid = f"nimbus-mcp-e2e-{uuid.uuid4().hex[:8]}"
    priv = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    pub = priv.public_key().public_numbers()
    body = json.dumps(
        {
            "keys": [
                {
                    "kid": kid,
                    "kty": "RSA",
                    "use": "sig",
                    "alg": "RS256",
                    "n": _b64url_uint(pub.n),
                    "e": _b64url_uint(pub.e),
                }
            ]
        }
    ).encode()

    class _Jwks(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802 — http.server API
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_args):  # silence request logging
            pass

    server = ThreadingHTTPServer(("127.0.0.1", _JWKS_PORT), _Jwks)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    now = int(time.time())
    token = jwt.encode(
        {
            "iss": _TEST_ISSUER,
            "sub": _USER_ID,
            "iat": now,
            "nbf": now - 1,
            "exp": now + 3600,
            "email": f"{_USER_ID}@clerk.test",
            "pla": "u:nimbus_studio_pro",
            "public_metadata": {},
        },
        priv,
        algorithm="RS256",
        headers={"kid": kid, "typ": "JWT"},
    )
    return token, server


@pytest.fixture(scope="module")
def pro_bearer():
    external = os.environ.get("NIMBUS_E2E_CLERK_TOKEN", "").strip()
    if external:
        yield external
        return
    token, server = _mint_pro_bearer()
    yield token
    server.shutdown()


def _mcp_server(bearer: str):
    os.environ["NIMBUS_TOKEN"] = bearer
    os.environ["NIMBUS_API_URL"] = _API
    from nimbus_mcp.server import build_server

    return build_server()


def _seed_donor_upload(recording_path: Path) -> str:
    """Register the downloaded recording as an uploaded_files row the deploy
    user owns (the backend E2E donor pattern — get_upload_for_user keys on
    user_id; an absolute path passes the uploads-dir join unchanged)."""
    from sqlalchemy import create_engine, text

    engine = create_engine(
        os.environ["DATABASE_URL"].replace("postgresql://", "postgresql+psycopg2://", 1)
    )
    try:
        with engine.begin() as conn:
            conn.execute(
                text("INSERT INTO users (id) VALUES (:uid) ON CONFLICT (id) DO NOTHING"),
                {"uid": _USER_ID},
            )
            return str(
                conn.execute(
                    text("""
                        INSERT INTO uploaded_files
                            (user_id, path, filename, original_filename, size, format)
                        VALUES (:uid, :path, :filename, :original, :size, 'hdf5')
                        RETURNING id::text
                    """),
                    {
                        "uid": _USER_ID,
                        "path": str(recording_path),
                        "filename": recording_path.name,
                        "original": recording_path.name,
                        "size": recording_path.stat().st_size,
                    },
                ).scalar_one()
            )
    finally:
        engine.dispose()


async def _run_simulated_session(
    poller: Client,
    bearer: str,
    started: dict,
    spec: dict,
    *,
    stop_via: str,
) -> dict:
    """One simulated deploy session: WS init (Studio-app stand-in) → start →
    blocks elapse server-side → stop (WS stop_ack for the donor capture, or
    MCP protocol.stop for the counted sessions)."""
    sid = started["sessionId"]
    blocks = spec["blocks"]
    out: dict = {"ok": False}
    async with websockets.connect(f"{_WS}/ws/realtime/{sid}", max_size=None) as ws:
        await ws.send(json.dumps({"type": "auth", "clerkToken": bearer}))
        await ws.recv()  # connected
        await ws.send(
            json.dumps(
                {
                    "type": "init_deployment",
                    "modelExecutionId": started["executionId"],
                    # start_assignment already connected the simulated device
                    # to this session id; init reuses it (simulated = free tier).
                    "dataSource": {
                        "type": "simulated",
                        "config": {
                            "deviceType": "simulated",
                            "channels": started["channels"],
                            "samplingRate": started["samplingRate"],
                        },
                    },
                    "streamingParams": {
                        "chunkSize": 125,
                        "windowSize": 0,
                        "overlap": 0,
                        "updateRate": 4.0,
                        "aggregationMethod": "weighted_vote",
                        "confidenceThreshold": 0.5,
                    },
                }
            )
        )
        init = json.loads(await asyncio.wait_for(ws.recv(), 20))
        assert init.get("type") == "initialized", f"init failed: {init}"
        # Live scheduler snapshot rides the ack (mounts the feedback panel).
        assert init["protocol"]["sessionsPlanned"] == SESSIONS
        await ws.recv()  # protocol_block (block 0)
        await ws.recv()  # protocol_progress
        await ws.send(json.dumps({"type": "start"}))
        ack = json.loads(await asyncio.wait_for(ws.recv(), 10))
        assert ack.get("statusKind") == "start_ack", ack

        # The simulated device loop feeds chunks server-side; watch for the
        # final block, then let it elapse. Bounded drains so the exit flag is
        # re-checked even when frames arrive faster than the recv timeout.
        total_sec = sum(b["durationSec"] for b in blocks)
        deadline = time.monotonic() + total_sec + 20
        saw_last = False
        while time.monotonic() < deadline:
            for _ in range(12):
                try:
                    msg = json.loads(await asyncio.wait_for(ws.recv(), 0.25))
                except asyncio.TimeoutError:
                    break
                if msg.get("type") == "protocol_block" and msg.get("index") == len(blocks) - 1:
                    saw_last = True
                    break
            if saw_last:
                await asyncio.sleep(BLOCK_SEC + 1.5)
                break
            snap = await poller.call_tool("protocol.session_status", {"session_id": sid})
            if (snap.data.get("protocol") or {}).get("blockId") == blocks[-1]["id"]:
                saw_last = True
                await asyncio.sleep(BLOCK_SEC + 1.5)
                break
            await asyncio.sleep(0.4)
        assert saw_last, "never observed the final protocol block"

        if stop_via == "mcp":
            stopped = await poller.call_tool("protocol.stop", {"session_id": sid})
            assert stopped.data["stopped"] is True, stopped.data
            out["ok"] = True
        else:
            await ws.send(json.dumps({"type": "stop"}))
            stop_deadline = time.monotonic() + 10
            while time.monotonic() < stop_deadline:
                try:
                    msg = json.loads(await asyncio.wait_for(ws.recv(), 2))
                except (websockets.ConnectionClosed, asyncio.TimeoutError):
                    break
                if msg.get("statusKind") == "stop_ack":
                    assert msg.get("recordingSaved") is True, msg
                    out["ok"] = True
                    out["filename"] = msg.get("filename")
                    break
    return out


async def test_protocol_builder_acceptance(pro_bearer, tmp_path: Path):
    t0 = time.monotonic()
    spec = _mini_spec()
    async with Client(_mcp_server(pro_bearer)) as c:
        # ── document lifecycle: validate → validate(bad) → save ──
        verdict = await c.call_tool("protocol.validate", {"spec": spec})
        assert verdict.data["valid"] is True
        assert verdict.data["spec"]["id"] == spec["id"]
        bad = await c.call_tool("protocol.validate", {"spec": {"id": "x"}})
        assert bad.data["valid"] is False and bad.data["errors"]
        saved = await c.call_tool("protocol.save", {"spec": spec, "confirm": True})
        assert saved.data["protocolId"] == spec["id"] and saved.data["version"] == 1

        # ── active arm: one recorded simulated session → yoked donor ──
        active = await c.call_tool(
            "protocol.start_assignment",
            {
                "protocol_id": spec["id"],
                "protocol_version": 1,
                "subject_label": "S01",
                "arm": "active",
                "confirm": True,
            },
        )
        assert active.data["started"] is True
        assert active.data["sessionId"] and active.data["executionId"]
        donor_run = await _run_simulated_session(
            c, pro_bearer, active.data, spec, stop_via="ws"
        )
        assert donor_run["ok"] and donor_run["filename"], donor_run

        async with httpx.AsyncClient(
            base_url=_API, headers={"Authorization": f"Bearer {pro_bearer}"}, timeout=30
        ) as raw:
            dl = await raw.get(f"/api/recordings/{donor_run['filename']}")
            assert dl.status_code == 200, dl.text[:200]
        donor_path = tmp_path / donor_run["filename"]
        donor_path.write_bytes(dl.content)
        donor_upload_id = _seed_donor_upload(donor_path)
        print(f"\n[protocol] donor recording {donor_run['filename']} → upload {donor_upload_id}")

        # ── sham arm (yoked donor) → 3 sessions to completion ──
        sham = await c.call_tool(
            "protocol.start_assignment",
            {
                "protocol_id": spec["id"],
                "protocol_version": 1,
                "subject_label": "S01",
                "arm": "sham",
                "yoked_donor_upload_id": donor_upload_id,
                "confirm": True,
            },
        )
        assert sham.data["started"] is True
        for i in range(SESSIONS):
            run = await _run_simulated_session(c, pro_bearer, sham.data, spec, stop_via="mcp")
            assert run["ok"], f"sham session {i + 1} failed: {run}"

        # ── stats: the acceptance criterion ──
        stats = await c.call_tool(
            "protocol.export_stats", {"assignment_id": sham.data["assignmentId"]}
        )
    data = stats.data
    assert data["sessionIndex"] == SESSIONS and data["sessionsPlanned"] == SESSIONS
    assert data["status"] == "completed"
    # Inferential block present (paired t over 3 sessions).
    assert data["inferential"]["inferential"] is True
    assert data["inferential"]["nSessions"] == SESSIONS
    assert isinstance(data["inferential"]["t"], float)
    # Figure: non-empty base64 PNG.
    image = data["figure"]["image"] or ""
    assert image and base64.b64decode(image, validate=True)[:4] == b"\x89PNG"
    # CSV: one aggregate row per session × block = 3 × 3.
    rows = [ln for ln in (data["csv"] or "").strip().splitlines()[1:] if ln.strip()]
    assert len(rows) == 9, f"expected 9 aggregate rows, got {len(rows)}: {rows}"
    print(f"\n[protocol] acceptance flow completed in {time.monotonic() - t0:.1f}s")
