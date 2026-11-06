"""FastAPI app with SSE endpoints.

Routes:

* ``POST /v1/stream`` — start a new streamed completion. Returns an SSE
  response with ``data: {"index": N, "text": "...", "session": "..."}``
  events and a final ``event: done`` or ``event: error`` event.
* ``GET /v1/stream/{session_id}/resume`` — re-attach to a paused, failed or
  finished session: replay the chunks after ``from_index``, then ``done`` or
  ``error``. The completion is not re-run.
* ``GET /healthz`` — liveness.
"""

from __future__ import annotations

import asyncio
import json
from typing import Callable

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from stream_llm_fastapi.backends import LLMBackend, MockLLM
from stream_llm_fastapi.session import (
    SessionState,
    StreamRegistry,
    StreamSession,
)


class StreamRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=64_000)


def _format_chunk(session_id: str, chunk) -> str:
    return json.dumps(
        {
            "session": session_id,
            "index": chunk.index,
            "text": chunk.text,
            "finish": chunk.finish,
        },
        ensure_ascii=False,
    )


async def _generate_sse(
    session: StreamSession,
    registry: StreamRegistry,
    request: Request,
    *,
    skip_to_index: int = -1,
):
    """Yield SSE events while honoring client disconnect detection."""
    # A stopped producer means the accumulator is final: replay it and end
    # with the terminal event. A paused session is never restarted, because
    # the backend would run the completion again from the first token.
    if session.state in (SessionState.DONE, SessionState.FAILED, SessionState.PAUSED):
        for i, txt in enumerate(session.accumulator):
            if i <= skip_to_index:
                continue
            yield {
                "event": "message",
                "data": json.dumps(
                    {"session": session.session_id, "index": i, "text": txt, "finish": False}
                ),
            }
        if session.state == SessionState.DONE:
            yield {"event": "done", "data": json.dumps({"session": session.session_id})}
        else:
            yield {
                "event": "error",
                "data": json.dumps(
                    {
                        "session": session.session_id,
                        "error": session.last_error or "unknown",
                        "last_index": session.last_index,
                        "partial_text": session.partial_text,
                    }
                ),
            }
        return
    if session.producer_task is None:
        session.producer_task = asyncio.create_task(session.run())

    try:
        try:
            async for chunk in registry.consume(session):
                if chunk.index <= skip_to_index:
                    continue
                yield {"event": "message", "data": _format_chunk(session.session_id, chunk)}
                if chunk.finish:
                    yield {"event": "done", "data": json.dumps({"session": session.session_id})}
                    return
        finally:
            # Whether we exit cleanly, via CancelledError, or via GeneratorExit
            # from aclose(), the upstream producer must not keep generating
            # tokens nobody will read. Fire-and-forget the cancel — we are not
            # allowed to await new I/O in a GeneratorExit unwind path.
            if session.producer_task and not session.producer_task.done():
                session.producer_task.cancel()

        # Producer finished without an explicit finish chunk
        if session.state == SessionState.DONE:
            yield {"event": "done", "data": json.dumps({"session": session.session_id})}
        elif session.state == SessionState.FAILED:
            yield {
                "event": "error",
                "data": json.dumps(
                    {
                        "session": session.session_id,
                        "error": session.last_error or "unknown",
                        "last_index": session.last_index,
                        "partial_text": session.partial_text,
                    }
                ),
            }
    except asyncio.CancelledError:
        raise


def create_app(
    backend_factory: Callable[[], LLMBackend] | None = None,
    *,
    queue_maxsize: int = 16,
) -> FastAPI:
    """Build a FastAPI app wired to ``backend_factory``.

    ``backend_factory`` is called once per stream so tests can pass per-request
    state (delays, fail_after, etc).
    """
    app = FastAPI(title="stream-llm-fastapi")
    registry = StreamRegistry(queue_maxsize=queue_maxsize)
    app.state.registry = registry
    if backend_factory is None:
        backend_factory = lambda: MockLLM()
    app.state.backend_factory = backend_factory

    @app.get("/healthz")
    async def healthz():
        return {"status": "ok", "active_sessions": len(registry.sessions)}

    @app.post("/v1/stream")
    async def stream_completion(req: StreamRequest, request: Request):
        backend = app.state.backend_factory()
        session = registry.new_session(prompt=req.prompt, backend=backend)

        async def event_iter():
            async for evt in _generate_sse(session, registry, request):
                yield evt

        return EventSourceResponse(event_iter())

    @app.get("/v1/stream/{session_id}/resume")
    async def resume(session_id: str, request: Request, from_index: int = -1):
        session = registry.get(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="unknown session")
        if session.state == SessionState.RUNNING:
            raise HTTPException(status_code=409, detail="session is still running")

        async def event_iter():
            async for evt in _generate_sse(
                session, registry, request, skip_to_index=from_index
            ):
                yield evt

        return EventSourceResponse(event_iter())

    @app.delete("/v1/stream/{session_id}")
    async def drop_session(session_id: str):
        session = registry.get(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="unknown session")
        if session.producer_task and not session.producer_task.done():
            session.producer_task.cancel()
        registry.drop(session_id)
        return JSONResponse({"deleted": session_id})

    return app
