"""Calibration E2E against a RUNNING local backend with test Clerk auth.

Gate: NIMBUS_MCP_E2E=1 + a backend on NIMBUS_API_URL. Skipped otherwise —
this is the local demo gate that proves the full agent loop:

    calibration.start(mi, 5 trials/class, confirm) → poll calibration.status
    (currentTrial/progress/eta while running) → uploadId when complete →
    calibration.train (DEFAULT mi path, no template_id) → poll execution.get
    → execution.results has metrics.kappa.

Auth: calibration graphs carry a ``hardware_device`` node, and node-level
freemium policy (freemium_execute_gate) applies to the X-MCP-Key local
principal unless the backend is flagged Pro (``MCP_LOCAL_IS_PRO`` — the
desktop app sets it for a signed-in Pro session; verified by a live-process
smoke). This E2E uses the claim-free route instead: the MCP client rides a
Bearer JWT minted the
same way the backend integration tests mint theirs (tests/utils/clerk_token
``_local_jwks_and_token``: local RS256 keypair, ``pla: u:nimbus_studio_pro``,
issuer https://clerk.test). The backend must run with::

    CLERK_ISSUER=https://clerk.test
    CLERK_JWKS_URL=http://127.0.0.1:8902/jwks.json

This module serves that JWKS on 8902 for the backend's verifier. Set
NIMBUS_E2E_CLERK_TOKEN instead to reuse an externally minted token.

The train call deliberately uses the default-mi path: it rewires the custom_data
node of ``mi_headband_csp_lda`` onto the recorded upload, so any config keys the
training engine needs (mode/paradigm/nClasses/classLabels) must survive the
rewire — this test is the catcher for that contract.
"""

import base64
import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from fastmcp import Client

pytestmark = pytest.mark.skipif(os.environ.get("NIMBUS_MCP_E2E") != "1", reason="NIMBUS_MCP_E2E!=1")

_TEST_ISSUER = "https://clerk.test"
_JWKS_PORT = int(os.environ.get("NIMBUS_E2E_JWKS_PORT", "8902"))


def _b64url_uint(val: int) -> str:
    """Base64url-encode an unsigned integer without padding (JWK n/e encoding)."""
    raw = val.to_bytes((val.bit_length() + 7) // 8, "big") or b"\x00"
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _mint_pro_bearer() -> tuple[str, ThreadingHTTPServer]:
    """Local RS256 Pro JWT + an HTTP server the backend fetches the JWKS from."""
    import uuid

    import jwt
    from cryptography.hazmat.primitives.asymmetric import rsa

    # Unique kid per run: the backend caches JWKS for 1 h, so a re-run under a
    # repeated kid would verify against the stale key (401). A fresh kid trips
    # the backend's rotation refresh, which re-fetches this server's JWKS.
    kid = f"nimbus-mcp-e2e-{uuid.uuid4().hex[:8]}"
    priv = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    pub = priv.public_key().public_numbers()
    jwks = {
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
    body = json.dumps(jwks).encode()

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
            "sub": "user_test_local",
            "iat": now,
            "nbf": now - 1,
            "exp": now + 3600,
            "email": "user_test_local@clerk.test",
            # Backend integration conftest's Pro claim (auth_capabilities:
            # ``nimbus_studio_pro`` slug → isPro → hardware/calibration allowed).
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
    """Bearer JWT the MCP client sends AND the backend verifies (via the JWKS)."""
    external = os.environ.get("NIMBUS_E2E_CLERK_TOKEN", "").strip()
    if external:
        yield external
        return
    token, server = _mint_pro_bearer()
    yield token
    server.shutdown()


def _server(bearer: str):
    os.environ["NIMBUS_TOKEN"] = bearer  # NimbusClient prefers the token (Bearer mode)
    from nimbus_mcp.server import build_server

    return build_server()


async def test_start_refuses_without_confirm():
    """The confirm gate refuses before any HTTP leaves the process."""
    async with Client(_server("unused-local-bearer")) as c:
        result = await c.call_tool(
            "calibration.start", {"paradigm": "mi", "trials_per_class": 5}
        )
    assert result.data["started"] is False
    assert "confirm" in result.data["message"]


async def test_calibration_record_and_train_default_mi(pro_bearer):
    async with Client(_server(pro_bearer)) as c:
        # ── Record: agent-guided synthetic MI session ──
        # 5 trials/class is the backend's minimum (trial_protocol schema);
        # the smallest legal session keeps the E2E fast.
        t_start = time.monotonic()
        started = await c.call_tool(
            "calibration.start",
            {
                "paradigm": "mi",
                "trials_per_class": 5,
                "confirm": True,
                "name": "mcp calibration e2e",
            },
        )
        assert started.data["started"] is True
        calib_id = started.data["executionId"]
        assert calib_id

        saw_intermediate = False
        saw_engine_eta = False
        phase = None
        # Budget: the template's 60 s baseline + 10 trials × ~7.5 s + finalize.
        deadline = time.time() + 420
        while time.time() < deadline:
            snap = await c.call_tool("calibration.status", {"execution_id": calib_id})
            phase = snap.data["phase"]
            status = snap.data["status"]
            if phase == "complete":
                break
            assert status not in ("failed", "cancelled"), f"calibration ended as {status}"
            # Running shape: trial counter + progress tally on every snapshot.
            # The very first poll can race the engine registration (no-engine
            # fallback: deviceConnected False, eta None) — the SSOT ETA is
            # required once the live engine feeds the snapshot.
            assert snap.data["currentTrial"] is not None
            assert snap.data["progress"] is not None
            saw_intermediate = True
            if snap.data["deviceConnected"]:
                assert snap.data["estimatedRemainingSec"] is not None
                saw_engine_eta = True
            time.sleep(2)
        assert phase == "complete", f"calibration phase stuck at {phase}"
        assert saw_intermediate, "never observed a running snapshot (currentTrial/progress)"
        assert saw_engine_eta, "never observed an engine snapshot with the SSOT ETA"
        t_recorded = time.monotonic() - t_start
        upload = snap.data["calibration"]
        assert upload and upload["uploadId"], f"no recorded upload: {upload}"
        print(f"\n[calibration] recorded in {t_recorded:.1f}s → upload {upload['uploadId']}")

        # ── Train: DEFAULT mi path — no template_id ──
        t_train_start = time.monotonic()
        trained = await c.call_tool(
            "calibration.train", {"execution_id": calib_id, "name": "mcp calibration e2e model"}
        )
        assert trained.data["calibrationUploadId"] == upload["uploadId"]
        train_id = trained.data["trainExecutionId"]
        assert train_id

        status = "running"
        deadline = time.time() + 600
        while time.time() < deadline:
            summary = await c.call_tool("execution.get", {"execution_id": train_id})
            status = summary.data["status"]
            if status in ("completed", "failed", "cancelled"):
                break
            time.sleep(3)
        assert status == "completed", f"training ended as {status}: {summary.data.get('error')}"
        results = await c.call_tool("execution.results", {"execution_id": train_id})
        assert "kappa" in results.data["metrics"], f"metrics: {results.data['metrics']}"
        t_trained = time.monotonic() - t_train_start
        print(
            f"\n[calibration] trained in {t_trained:.1f}s → "
            f"kappa={results.data['metrics']['kappa']}"
        )
