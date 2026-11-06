# stream-llm-fastapi

> Production-grade FastAPI + SSE for LLM completions. Three hero properties — **client disconnect cancels upstream**, **slow client backpressures the producer**, **mid-stream errors are resumable** — each guarded by a direct-ASGI integration test that drives `http.disconnect` itself instead of relying on a test client to do it.

[![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE) [![Tests](https://img.shields.io/badge/tests-6%20passing-brightgreen)](#tests) [![Python](https://img.shields.io/badge/python-3.10%2B-blue)]()

## The three problems most SSE LLM endpoints get wrong

| Problem | Naive impl | This repo |
|---|---|---|
| Client disconnects mid-stream | LLM keeps generating tokens, burning $ for nobody | Producer task is cancelled, session moves to `PAUSED`, upstream stops within a few tokens |
| Slow client (mobile, congested network) | Server buffers unbounded queue, OOMs under load | Bounded `asyncio.Queue` blocks the producer until the consumer drains — backpressure proven by test |
| Upstream LLM error 6 tokens in | Client gets a closed connection, no recovery path | Server emits an `event: error` payload with the partial accumulator + session id; `GET /v1/stream/{id}/resume` replays the partial and re-emits the error |

## How the test surface proves the contract

Most SSE servers test the happy path with a client that drains every chunk. The hard parts are the failure paths, and those depend on `http.disconnect` actually reaching the ASGI app. `httpx.ASGITransport` does **not** reliably emit `http.disconnect` when the response context exits, which is why most repos quietly leave these properties untested.

This repo includes a **direct ASGI test harness** (`tests/test_app_integration.py:_run_request`) that:

1. constructs a minimal ASGI `scope`,
2. drives the app coroutine with a custom `receive()`/`send()` pair,
3. emits `http.disconnect` after observing N body chunks,
4. captures every `http.response.body` message for assertion.

That's how all three failure properties are actually verified end-to-end.

## Quickstart

```bash
pip install -e ".[anthropic]"
export ANTHROPIC_API_KEY=sk-ant-...
stream-llm-fastapi --backend anthropic --port 8000
```

In a second terminal:

```bash
curl -N -X POST http://127.0.0.1:8000/v1/stream \
  -H "Content-Type: application/json" \
  -d '{"prompt":"Write 3 lines about kittens."}'
```

You'll see SSE events like:

```
data: {"session": "rk2A...", "index": 0, "text": "Kitt", "finish": false}
data: {"session": "rk2A...", "index": 1, "text": "ens are", "finish": false}
...
event: done
data: {"session": "rk2A..."}
```

## API

| Method | Path | Behavior |
|---|---|---|
| `POST` | `/v1/stream` | Start a new completion. Returns SSE: `message` per chunk, `done` on success, `error` on upstream failure. |
| `GET`  | `/v1/stream/{session_id}/resume?from_index=N` | Re-attach to a paused, failed or finished session. Replays the accumulator after `from_index`, then `done` (finished) or `error` (failed, or cut off by a disconnect). The completion is not re-run. |
| `DELETE` | `/v1/stream/{session_id}` | Drop the session (cancels producer if still running). |
| `GET`  | `/healthz` | `{"status": "ok", "active_sessions": N}` |

## Architecture

```
                          ┌──────────────────┐
POST /v1/stream  ──►      │  StreamSession   │
                          │                  │
                          │  ┌────────────┐  │
                          │  │ accumulator│  │   <─ index, partial_text
                          │  └────────────┘  │
                          │                  │
                          │ asyncio.Queue    │   <─ bounded (queue_maxsize)
                          │ (maxsize=16)     │      → backpressure
                          │                  │
                          └─┬──────────────┬─┘
                            │              │
                  produce   │              │   consume
                            ▼              ▼
                     ┌─────────────┐  ┌─────────────┐
                     │ LLMBackend  │  │ SSE writer  │
                     │  .stream()  │  │             │
                     └─────────────┘  └─────────────┘
```

**The queue is the load-bearing primitive.** A bounded `asyncio.Queue(maxsize=N)` between the producer (LLM iterator) and the consumer (SSE writer) gives you:

* **Backpressure for free**: if the writer slows, `queue.put()` blocks the producer at chunk `consumed + N`. No unbounded buffering.
* **Cancellation surface**: cancel the producer task and `queue.put()` raises `CancelledError`. State machine moves to `PAUSED` and the partial accumulator survives.
* **Resume without reading from a buffer of unknown size**: the accumulator is the source of truth, the queue is just the transport.

## Backend protocol

Any class with `model: str` and `stream(prompt) -> AsyncIterator[Chunk]` is a backend. Ships with:

* **`MockLLM`** — deterministic, supports `delay`, `fail_after`, and an `on_cancel` callback. Used by every test.
* **`AnthropicLLM`** — wraps `anthropic.AsyncAnthropic.messages.stream()`, normalises text deltas to `Chunk(text, index, finish)`.

OpenAI / Groq / Bedrock adapters are mechanical — same interface, different SDK call.

## Tests

```bash
pip install -e ".[dev]"
pytest -v
```

```
tests/test_app_integration.py::test_client_disconnect_cancels_producer PASSED
tests/test_app_integration.py::test_backpressure_under_slow_consumer    PASSED
tests/test_app_integration.py::test_mid_stream_error_then_resume        PASSED
tests/test_session.py::test_session_runs_to_done                        PASSED
tests/test_session.py::test_session_failure_preserves_partial           PASSED
tests/test_session.py::test_backpressure_blocks_producer_when_consumer_slow PASSED
6 passed in 1.20s
```

### What each integration test checks

* **`test_client_disconnect_cancels_producer`** — after 3 chunks are delivered, the harness emits `http.disconnect`. The producer stops within a handful of tokens (asserted `< 30` of 200), and the session is in `PAUSED` state in the registry.
* **`test_backpressure_under_slow_consumer`** — `send()` sleeps 20 ms per chunk. With `queue_maxsize=4`, the producer never runs more than `delivered + maxsize + writer_in_flight` tokens ahead.
* **`test_mid_stream_error_then_resume`** — `MockLLM(fail_after=3)`. The first response emits 3 `message` events then an `error` event with `partial_text="alpha beta gamma"`. A `GET /resume` replays the same partial and re-emits the same error event (FAILED sessions stay FAILED).

## Project layout

```
.
├── src/stream_llm_fastapi/
│   ├── __init__.py
│   ├── backends.py     # LLMBackend protocol + MockLLM + AnthropicLLM
│   ├── session.py      # StreamSession state machine + StreamRegistry
│   ├── app.py          # FastAPI app factory + SSE generator
│   └── cli.py          # `stream-llm-fastapi` server CLI
├── tests/
│   ├── test_session.py            # state-machine unit tests
│   └── test_app_integration.py    # direct-ASGI integration tests
└── pyproject.toml
```

## Limitations

**In-process session registry.** Sessions live in memory. A real deployment behind a load balancer needs Redis (or similar) so a resume request can hit a different worker. The `StreamRegistry` interface is intentionally small — swapping the dict for a Redis-backed implementation is a few hundred lines.

**No streaming auth/quota gate.** There's no per-key rate limit or auth middleware shipped here. Wire your own FastAPI dependency for those.

**SSE only, no WebSocket fallback.** SSE is one-way (server → client) and works fine for chat completions. If you need full duplex (e.g., interrupt-mid-generation from the client), use the same state machine behind a WebSocket route.

**No multi-turn / tool execution.** Each `POST /v1/stream` is a single completion. Multi-turn tool calls would need a session that holds the conversation array — different scope.

## License

MIT — see [LICENSE](LICENSE).
