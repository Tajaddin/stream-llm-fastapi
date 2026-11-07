"""Unit tests for StreamSession / StreamRegistry state machine."""

from __future__ import annotations

import asyncio
import time

import pytest

from stream_llm_fastapi.backends import MockLLM
from stream_llm_fastapi.session import SessionLimitError, SessionState, StreamRegistry


async def _drain(session, registry):
    out = []
    async for chunk in registry.consume(session):
        out.append(chunk)
    return out


@pytest.mark.asyncio
async def test_session_runs_to_done():
    registry = StreamRegistry(queue_maxsize=4)
    session = registry.new_session(prompt="p", backend=MockLLM(response="a b c"))
    session.producer_task = asyncio.create_task(session.run())

    chunks = await _drain(session, registry)

    await session.producer_task
    assert session.state == SessionState.DONE
    finish_chunks = [c for c in chunks if c.finish]
    assert len(finish_chunks) == 1
    assert session.partial_text == "a b c"


@pytest.mark.asyncio
async def test_session_failure_preserves_partial():
    registry = StreamRegistry(queue_maxsize=4)
    backend = MockLLM(response="one two three four", fail_after=2)
    session = registry.new_session(prompt="p", backend=backend)
    session.producer_task = asyncio.create_task(session.run())

    chunks = await _drain(session, registry)

    await session.producer_task
    assert session.state == SessionState.FAILED
    assert session.last_error and "simulated upstream failure" in session.last_error
    # 2 chunks made it out before the failure
    assert sum(1 for c in chunks if c.text) == 2
    assert session.partial_text == "one two "


@pytest.mark.asyncio
async def test_backpressure_blocks_producer_when_consumer_slow():
    """With queue_maxsize=2 and a 100-token response, the producer must block
    after 2 puts if the consumer never reads."""
    registry = StreamRegistry(queue_maxsize=2)
    backend = MockLLM(response=" ".join(str(i) for i in range(100)), delay=0.001)
    session = registry.new_session(prompt="p", backend=backend)
    session.producer_task = asyncio.create_task(session.run())

    # Let producer try to fill the queue, but never drain it.
    await asyncio.sleep(0.2)

    assert session.queue.qsize() <= 2, "queue exceeded maxsize"
    assert backend.chunks_yielded <= 3, (
        f"producer was not throttled by backpressure: yielded {backend.chunks_yielded}"
    )

    session.producer_task.cancel()
    try:
        await session.producer_task
    except asyncio.CancelledError:
        pass


async def _finished_session(registry):
    session = registry.new_session(prompt="p", backend=MockLLM(response="a b"))
    session.producer_task = asyncio.create_task(session.run())
    await _drain(session, registry)
    await session.producer_task
    return session


@pytest.mark.asyncio
async def test_finished_sessions_expire_after_ttl():
    registry = StreamRegistry(session_ttl=60.0)
    old = await _finished_session(registry)
    fresh = await _finished_session(registry)
    old.finished_at -= 61.0

    registry.new_session(prompt="p", backend=MockLLM())

    assert old.session_id not in registry.sessions
    assert registry.get(fresh.session_id) is fresh


def test_session_whose_response_never_started_expires():
    registry = StreamRegistry(session_ttl=60.0)
    session = registry.new_session(prompt="p", backend=MockLLM())
    session.created_at -= 61.0

    assert registry.get(session.session_id) is None


@pytest.mark.asyncio
async def test_wall_clock_jump_does_not_expire_sessions(monkeypatch):
    registry = StreamRegistry(session_ttl=60.0)
    finished = await _finished_session(registry)
    never_started = registry.new_session(prompt="p", backend=MockLLM())
    wall = time.time()
    monkeypatch.setattr(time, "time", lambda: wall + 3600.0)

    assert registry.get(finished.session_id) is finished
    assert registry.get(never_started.session_id) is never_started


def _running_session(registry):
    session = registry.new_session(prompt="p", backend=MockLLM(delay=60.0))
    session.producer_task = asyncio.create_task(session.run())
    return session


async def _stop(*sessions):
    for session in sessions:
        session.producer_task.cancel()
    await asyncio.gather(*(s.producer_task for s in sessions), return_exceptions=True)


@pytest.mark.asyncio
async def test_session_cap_evicts_oldest_finished_then_refuses():
    registry = StreamRegistry(max_sessions=2)
    finished = await _finished_session(registry)
    running = _running_session(registry)

    third = _running_session(registry)
    assert finished.session_id not in registry.sessions
    assert set(registry.sessions) == {running.session_id, third.session_id}

    with pytest.raises(SessionLimitError):
        registry.new_session(prompt="p", backend=MockLLM())
    await _stop(running, third)


@pytest.mark.asyncio
async def test_session_cap_does_not_evict_a_session_that_has_not_started_yet():
    registry = StreamRegistry(max_sessions=2)
    starting = registry.new_session(prompt="p", backend=MockLLM())
    running = _running_session(registry)

    with pytest.raises(SessionLimitError):
        registry.new_session(prompt="p", backend=MockLLM())

    assert set(registry.sessions) == {starting.session_id, running.session_id}
    await _stop(running)
