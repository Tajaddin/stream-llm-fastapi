"""End-to-end integration tests over the ASGI protocol.

These tests drive the ASGI app directly (no httpx) so we can:

* explicitly emit ``http.disconnect`` mid-stream (httpx's ASGITransport does
  not reliably forward disconnects, which is the trap most SSE servers fail
  to test against)
* observe ``http.response.body`` messages as they're produced and time the
  disconnect after a known number of them

The three hero tests:

1. **Client disconnect mid-stream** — disconnect after a few chunks, then
   verify the upstream LLM stops producing and the session is in PAUSED.
2. **Backpressure under a slow client** — we delay ASGI ``send`` calls so
   sse-starlette's writer slows; the producer should not run more than
   (consumed + queue_maxsize) chunks ahead.
3. **Mid-stream upstream error → resume** — the LLM raises after N tokens,
   the SSE response emits an ``error`` event with the partial accumulator,
   and a follow-up ``/resume`` re-emits the same partial + error.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field

import pytest

from stream_llm_fastapi.app import create_app
from stream_llm_fastapi.backends import MockLLM
from stream_llm_fastapi.session import SessionState


def _scope(method: str, path: str) -> dict:
    return {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "method": method,
        "scheme": "http",
        "path": path,
        "raw_path": path.encode(),
        "query_string": b"",
        "root_path": "",
        "headers": [
            (b"content-type", b"application/json"),
            (b"host", b"test"),
            (b"accept", b"text/event-stream"),
        ],
        "client": ("127.0.0.1", 12345),
        "server": ("test", 80),
    }


@dataclass
class CapturedResponse:
    status: int | None = None
    headers: list[tuple[bytes, bytes]] = field(default_factory=list)
    body_chunks: list[bytes] = field(default_factory=list)
    complete: bool = False

    @property
    def body_text(self) -> str:
        return b"".join(self.body_chunks).decode("utf-8", errors="replace")

    def parse_sse(self) -> list[dict]:
        events: list[dict] = []
        current: dict = {}
        for raw_line in self.body_text.splitlines():
            line = raw_line.rstrip("\r")
            if not line.strip():
                if current:
                    if isinstance(current.get("data"), list):
                        joined = "\n".join(current["data"])
                        try:
                            current["data"] = json.loads(joined)
                        except (json.JSONDecodeError, ValueError):
                            current["data"] = joined
                    events.append(current)
                    current = {}
                continue
            if line.startswith(":"):
                continue
            if ":" in line:
                key, _, val = line.partition(":")
                key = key.strip()
                val = val.lstrip(" ")
                if key == "data":
                    current.setdefault("data", []).append(val)
                else:
                    current[key] = val
        if current:
            if isinstance(current.get("data"), list):
                joined = "\n".join(current["data"])
                try:
                    current["data"] = json.loads(joined)
                except (json.JSONDecodeError, ValueError):
                    current["data"] = joined
            events.append(current)
        return events


async def _run_request(
    app,
    scope: dict,
    body: bytes,
    *,
    disconnect_after_chunks: int | None = None,
    send_delay_per_chunk: float = 0.0,
) -> CapturedResponse:
    """Invoke the ASGI app and capture the full SSE response.

    Optionally trigger ``http.disconnect`` after the server has emitted
    ``disconnect_after_chunks`` response body chunks.
    """
    cap = CapturedResponse()
    body_sent = False
    disconnect_sent = False
    incoming: asyncio.Queue = asyncio.Queue()

    async def receive():
        nonlocal body_sent
        if not body_sent:
            body_sent = True
            return {"type": "http.request", "body": body, "more_body": False}
        return await incoming.get()

    async def send(msg):
        nonlocal disconnect_sent
        if msg["type"] == "http.response.start":
            cap.status = msg["status"]
            cap.headers = list(msg.get("headers", []))
        elif msg["type"] == "http.response.body":
            chunk = msg.get("body", b"") or b""
            cap.body_chunks.append(chunk)
            if send_delay_per_chunk:
                await asyncio.sleep(send_delay_per_chunk)
            if (
                disconnect_after_chunks is not None
                and not disconnect_sent
                and sum(1 for b in cap.body_chunks if b'"index"' in b) >= disconnect_after_chunks
            ):
                disconnect_sent = True
                await incoming.put({"type": "http.disconnect"})
            if not msg.get("more_body", False):
                cap.complete = True

    await app(scope, receive, send)
    return cap


@pytest.mark.asyncio
async def test_client_disconnect_cancels_producer():
    """Hero test #1: client disconnects mid-stream. The upstream LLM must
    stop producing tokens (no wasted upstream cost) and the session must
    end up in PAUSED state so it can be resumed."""
    backend = MockLLM(response=" ".join(["tok"] * 200), delay=0.01)
    app = create_app(backend_factory=lambda: backend, queue_maxsize=4)

    cap = await _run_request(
        app,
        _scope("POST", "/v1/stream"),
        b'{"prompt":"hi"}',
        disconnect_after_chunks=3,
    )

    # Give the server's cancellation chain a beat to settle.
    await asyncio.sleep(0.1)

    assert cap.status == 200
    # Producer must stop well before the full 200 tokens.
    assert backend.chunks_yielded < 200, (
        f"producer kept yielding after disconnect: {backend.chunks_yielded}"
    )
    assert backend.chunks_yielded < 30, (
        f"producer should stop quickly after disconnect; yielded {backend.chunks_yielded}"
    )

    # Session preserved in PAUSED state.
    registry = app.state.registry
    paused = [s for s in registry.sessions.values() if s.state == SessionState.PAUSED]
    assert paused, (
        f"expected a paused session; got states "
        f"{[s.state.value for s in registry.sessions.values()]}"
    )


@pytest.mark.asyncio
async def test_backpressure_under_slow_consumer():
    """Hero test #2: a slow ASGI consumer (send_delay_per_chunk) must
    backpressure the producer through the bounded queue. The producer
    is allowed to run at most ``queue_maxsize`` chunks ahead of the
    delivered SSE events."""
    queue_maxsize = 4
    backend = MockLLM(response=" ".join(["x"] * 60), delay=0.0)
    app = create_app(backend_factory=lambda: backend, queue_maxsize=queue_maxsize)

    cap = await _run_request(
        app,
        _scope("POST", "/v1/stream"),
        b'{"prompt":"p"}',
        send_delay_per_chunk=0.02,
        disconnect_after_chunks=10,
    )

    await asyncio.sleep(0.1)

    # Producer should NOT have run away to the full 60 chunks.
    # Tolerance: a few chunks may be in the SSE writer's local buffer between
    # the queue and the network. queue_maxsize + writer-in-flight + 1 (chunk
    # currently being produced) is a safe upper bound.
    delivered_event_count = sum(
        1 for b in cap.body_chunks if b'"index"' in b
    )
    assert backend.chunks_yielded <= delivered_event_count + queue_maxsize + 4, (
        f"backpressure broken: delivered={delivered_event_count}, "
        f"yielded={backend.chunks_yielded}, maxsize={queue_maxsize}"
    )


@pytest.mark.asyncio
async def test_mid_stream_error_then_resume():
    """Hero test #3: the LLM fails mid-stream. The SSE response emits an
    `error` event with the partial accumulator. A follow-up GET /resume
    replays the partial and re-emits the error (FAILED sessions stay FAILED)."""
    backend = MockLLM(
        response="alpha beta gamma delta epsilon zeta eta theta",
        fail_after=3,
    )
    app = create_app(backend_factory=lambda: backend, queue_maxsize=8)

    cap = await _run_request(app, _scope("POST", "/v1/stream"), b'{"prompt":"go"}')
    assert cap.status == 200
    events = cap.parse_sse()

    msg_events = [e for e in events if e.get("event", "message") == "message" or "event" not in e]
    msg_events = [e for e in msg_events if isinstance(e.get("data"), dict) and "index" in e["data"]]
    err_events = [e for e in events if e.get("event") == "error"]
    done_events = [e for e in events if e.get("event") == "done"]

    assert msg_events, f"no message events were emitted; raw: {cap.body_text!r}"
    assert err_events, f"expected an error event; events: {[e.get('event') for e in events]}"
    assert not done_events, "should NOT have emitted a done event for a failed stream"

    err_payload = err_events[0]["data"]
    assert isinstance(err_payload, dict)
    assert "session" in err_payload
    assert err_payload["partial_text"].startswith("alpha")
    assert "simulated upstream failure" in err_payload["error"]

    session_id = err_payload["session"]
    partial_text_first = err_payload["partial_text"]

    # Now resume. Path includes the session_id.
    cap2 = await _run_request(
        app,
        _scope("GET", f"/v1/stream/{session_id}/resume"),
        b"",
    )
    assert cap2.status == 200
    events2 = cap2.parse_sse()
    resume_msgs = [
        e for e in events2
        if isinstance(e.get("data"), dict) and "text" in e["data"]
    ]
    replayed = "".join(e["data"]["text"] for e in resume_msgs)
    assert replayed == partial_text_first, (
        f"resume replay mismatch: replayed={replayed!r}, original={partial_text_first!r}"
    )

    resume_errs = [e for e in events2 if e.get("event") == "error"]
    assert resume_errs, "resume should re-emit the error event for FAILED sessions"


@pytest.mark.asyncio
async def test_resume_after_disconnect_replays_without_restarting_upstream():
    """A client that disconnects mid-stream can resume: it gets every chunk
    produced after the last one it saw, then a terminal event. The upstream
    call is not re-run, and the session stays resumable."""
    backend = MockLLM(response=" ".join(f"t{i}" for i in range(40)), delay=0.01)
    app = create_app(backend_factory=lambda: backend, queue_maxsize=4)

    cap = await _run_request(
        app,
        _scope("POST", "/v1/stream"),
        b'{"prompt":"hi"}',
        disconnect_after_chunks=3,
    )
    await asyncio.sleep(0.1)

    session = next(iter(app.state.registry.sessions.values()))
    assert session.state == SessionState.PAUSED
    seen = [e["data"] for e in cap.parse_sse() if e.get("event") == "message"]
    last = seen[-1]["index"]
    yielded = backend.chunks_yielded

    for _ in range(2):
        scope = _scope("GET", f"/v1/stream/{session.session_id}/resume")
        scope["query_string"] = f"from_index={last}".encode()
        cap2 = await asyncio.wait_for(_run_request(app, scope, b""), 5)

        assert cap2.status == 200
        events = cap2.parse_sse()
        replayed = [e["data"] for e in events if e.get("event") == "message"]
        assert [d["index"] for d in replayed] == list(range(last + 1, len(session.accumulator)))
        assert "".join(d["text"] for d in seen + replayed) == session.partial_text
        assert events[-1]["event"] == "error"
        assert events[-1]["data"]["partial_text"] == session.partial_text

    assert session.state == SessionState.PAUSED
    assert backend.chunks_yielded == yielded, "resume must not re-run the upstream call"


@pytest.mark.asyncio
async def test_stream_rejected_when_session_limit_reached():
    app = create_app(backend_factory=lambda: MockLLM(), max_sessions=1)
    busy = app.state.registry.new_session(prompt="busy", backend=MockLLM(delay=60.0))
    busy.producer_task = asyncio.create_task(busy.run())

    cap = await _run_request(app, _scope("POST", "/v1/stream"), b'{"prompt":"hi"}')

    assert cap.status == 503
    busy.producer_task.cancel()
    await asyncio.gather(busy.producer_task, return_exceptions=True)
