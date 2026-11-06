"""Session state machine: one streamed completion = one ``StreamSession``.

State transitions:

* ``RUNNING`` — actively pulling from the LLM and pushing to a queue
* ``PAUSED`` — client disconnected; partial accumulator preserved for resume
* ``DONE`` — final chunk delivered; safe to garbage-collect
* ``FAILED`` — upstream raised; partial accumulator + last error preserved

The queue has bounded capacity to provide **backpressure**: if the consumer
(HTTP response writer) stops draining, the producer (LLM iterator) blocks
on ``queue.put`` and stops requesting new tokens. This caps memory use even
when one client is slow and the model is fast.
"""

from __future__ import annotations

import asyncio
import enum
import secrets
import time
from dataclasses import dataclass, field
from typing import AsyncIterator

from stream_llm_fastapi.backends import Chunk, LLMBackend, LLMError


class SessionState(str, enum.Enum):
    RUNNING = "running"
    PAUSED = "paused"
    DONE = "done"
    FAILED = "failed"


_SENTINEL = object()


@dataclass
class StreamSession:
    """One ongoing streamed completion."""

    session_id: str
    prompt: str
    backend: LLMBackend
    queue: asyncio.Queue
    state: SessionState = SessionState.RUNNING
    accumulator: list[str] = field(default_factory=list)
    last_index: int = -1
    last_error: str | None = None
    producer_task: asyncio.Task | None = None
    created_at: float = field(default_factory=time.time)

    async def run(self) -> None:
        """Producer: pull from backend and push onto the queue.

        Marks state DONE/FAILED on completion. Always closes the queue with
        a sentinel so consumers don't deadlock.
        """
        try:
            async for chunk in self.backend.stream(self.prompt):
                self.accumulator.append(chunk.text)
                self.last_index = chunk.index
                await self.queue.put(chunk)
                if chunk.finish:
                    self.state = SessionState.DONE
                    break
            else:
                # iterator ended without an explicit finish chunk
                self.state = SessionState.DONE
        except asyncio.CancelledError:
            # consumer disconnected — preserve partial for resume
            self.state = SessionState.PAUSED
            self.last_error = "interrupted: client disconnected before the stream finished"
            # Best-effort sentinel; if queue is full there's nobody draining
            # anyway and we don't want to block cleanup on it.
            try:
                self.queue.put_nowait(_SENTINEL)
            except asyncio.QueueFull:
                pass
            raise
        except LLMError as exc:
            self.state = SessionState.FAILED
            self.last_error = str(exc)
        except Exception as exc:  # noqa: BLE001
            self.state = SessionState.FAILED
            self.last_error = f"{type(exc).__name__}: {exc}"
        # For DONE / FAILED, push the sentinel so any waiting consumer wakes.
        if self.state in (SessionState.DONE, SessionState.FAILED):
            try:
                self.queue.put_nowait(_SENTINEL)
            except asyncio.QueueFull:
                await self.queue.put(_SENTINEL)

    @property
    def partial_text(self) -> str:
        return "".join(self.accumulator)


class StreamRegistry:
    """In-process map of session_id → StreamSession.

    Tracks paused/failed sessions so /resume can pick them up. Production
    deployments would back this with Redis to survive process restarts.
    """

    def __init__(self, queue_maxsize: int = 16) -> None:
        self.sessions: dict[str, StreamSession] = {}
        self.queue_maxsize = queue_maxsize

    def new_session(self, prompt: str, backend: LLMBackend) -> StreamSession:
        session_id = secrets.token_urlsafe(12)
        queue: asyncio.Queue = asyncio.Queue(maxsize=self.queue_maxsize)
        session = StreamSession(
            session_id=session_id, prompt=prompt, backend=backend, queue=queue
        )
        self.sessions[session_id] = session
        return session

    def get(self, session_id: str) -> StreamSession | None:
        return self.sessions.get(session_id)

    def drop(self, session_id: str) -> None:
        self.sessions.pop(session_id, None)

    async def consume(self, session: StreamSession) -> AsyncIterator[Chunk]:
        """Drain the session's queue, yielding chunks until DONE / FAILED / sentinel."""
        while True:
            item = await session.queue.get()
            if item is _SENTINEL:
                break
            yield item
            if item.finish:
                break
