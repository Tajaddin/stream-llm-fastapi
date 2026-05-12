"""Unit tests for StreamSession / StreamRegistry state machine."""

from __future__ import annotations

import asyncio

import pytest

from stream_llm_fastapi.backends import MockLLM
from stream_llm_fastapi.session import SessionState, StreamRegistry


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
