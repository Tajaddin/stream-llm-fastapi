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
    created_at: float = field(default_factory=time.monotonic)
    finished_at: float | None = None

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
            self.finished_at = time.monotonic()
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
        self.finished_at = time.monotonic()
        # For DONE / FAILED, push the sentinel so any waiting consumer wakes.
        if self.state in (SessionState.DONE, SessionState.FAILED):
            try:
                self.queue.put_nowait(_SENTINEL)
            except asyncio.QueueFull:
                await self.queue.put(_SENTINEL)

    @property
    def partial_text(self) -> str:
        return "".join(self.accumulator)


class SessionLimitError(RuntimeError):
    """The registry is full of running sessions; nothing can be evicted."""


class StreamRegistry:
    """In-process map of session_id → StreamSession.

    Tracks paused/failed sessions so /resume can pick them up. Production
    deployments would back this with Redis to survive process restarts.

    Memory is bounded two ways: a session that has stopped (done, failed or
    paused) is dropped ``session_ttl`` seconds later, and the registry holds
    at most ``max_sessions``. At the cap the oldest idle sessions make room:
    stopped ones, and ones whose response never started its producer. If
    every session is still running, :meth:`new_session` raises
    :class:`SessionLimitError`.
    """

    def __init__(
        self,
        queue_maxsize: int = 16,
        *,
        session_ttl: float = 300.0,
        max_sessions: int = 1000,
    ) -> None:
        self.sessions: dict[str, StreamSession] = {}
        self.queue_maxsize = queue_maxsize
        self.session_ttl = session_ttl
        self.max_sessions = max_sessions

    def _expired(self, session: StreamSession, now: float) -> bool:
        if session.finished_at is not None:
            return now - session.finished_at > self.session_ttl
        # A response that was cancelled before its first event never starts
        # a producer, so nothing would ever mark the session finished.
        return session.producer_task is None and now - session.created_at > self.session_ttl

    def _evict(self) -> None:
        now = time.monotonic()
        for sid in [sid for sid, s in self.sessions.items() if self._expired(s, now)]:
            del self.sessions[sid]
        excess = len(self.sessions) - self.max_sessions + 1
        if excess > 0:
            idle = sorted(
                (
                    s
                    for s in self.sessions.values()
                    if s.finished_at is not None or s.producer_task is None
                ),
                key=lambda s: s.finished_at if s.finished_at is not None else s.created_at,
            )
            for s in idle[:excess]:
                del self.sessions[s.session_id]

    def new_session(self, prompt: str, backend: LLMBackend) -> StreamSession:
        self._evict()
        if len(self.sessions) >= self.max_sessions:
            raise SessionLimitError(f"{self.max_sessions} sessions are already running")
        session_id = secrets.token_urlsafe(12)
        queue: asyncio.Queue = asyncio.Queue(maxsize=self.queue_maxsize)
        session = StreamSession(
            session_id=session_id, prompt=prompt, backend=backend, queue=queue
        )
        self.sessions[session_id] = session
        return session

    def get(self, session_id: str) -> StreamSession | None:
        session = self.sessions.get(session_id)
        if session is not None and self._expired(session, time.monotonic()):
            del self.sessions[session_id]
            return None
        return session

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
