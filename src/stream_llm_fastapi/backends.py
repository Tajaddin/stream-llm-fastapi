"""LLM backend protocol + Mock + Anthropic adapter.

A backend is anything that exposes ``stream(prompt) -> AsyncIterator[Chunk]``.
The HTTP layer doesn't care which provider produced the tokens.
"""

from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass
from typing import AsyncIterator, Awaitable, Callable, Protocol


@dataclass
class Chunk:
    """One incremental piece of model output."""

    text: str
    index: int
    finish: bool = False


class LLMError(RuntimeError):
    """Backend-side failure during a streamed generation."""


class LLMBackend(Protocol):
    """Anything that yields :class:`Chunk` objects for a prompt."""

    model: str

    def stream(self, prompt: str) -> AsyncIterator[Chunk]: ...


# ---------------------------------------------------------------------------
# MockLLM
# ---------------------------------------------------------------------------


class MockLLM:
    """Deterministic backend for tests.

    Splits ``response`` into space-separated chunks and yields them with
    ``delay`` seconds between each. If ``fail_after`` is set, raises
    :class:`LLMError` after that many chunks (useful for the resume test).
    """

    def __init__(
        self,
        response: str = "hello world from mock",
        delay: float = 0.0,
        fail_after: int | None = None,
        on_cancel: Callable[[], Awaitable[None]] | None = None,
    ) -> None:
        self.model = "mock"
        self.response = response
        self.delay = delay
        self.fail_after = fail_after
        self.on_cancel = on_cancel
        self.chunks_yielded = 0
        self.cancelled = False

    async def stream(self, prompt: str) -> AsyncIterator[Chunk]:
        parts = self.response.split(" ")
        try:
            for i, part in enumerate(parts):
                if self.fail_after is not None and i >= self.fail_after:
                    raise LLMError(f"simulated upstream failure after {i} chunks")
                if self.delay:
                    await asyncio.sleep(self.delay)
                self.chunks_yielded += 1
                yield Chunk(text=part + (" " if i < len(parts) - 1 else ""), index=i, finish=False)
            yield Chunk(text="", index=len(parts), finish=True)
        except (asyncio.CancelledError, GeneratorExit):
            # GeneratorExit fires when the async-gen is aclose()'d by the
            # consumer task being cancelled. We treat both signals as "client
            # is gone, stop producing." We can't await new I/O in a
            # GeneratorExit handler, but we can run the sync flag + a fast
            # awaitable if one is provided.
            self.cancelled = True
            if self.on_cancel:
                try:
                    await self.on_cancel()
                except BaseException:  # noqa: BLE001
                    pass
            raise


# ---------------------------------------------------------------------------
# AnthropicLLM
# ---------------------------------------------------------------------------


class AnthropicLLM:
    """Wraps anthropic.AsyncAnthropic.messages.stream().

    Normalises Anthropic deltas into :class:`Chunk`. Lazy imports the SDK so
    ``stream-llm-fastapi`` runs in environments without anthropic installed.

    ``timeout`` (seconds) bounds the connect and each read of the stream, so a
    stalled upstream fails the session instead of holding it for the SDK's
    10-minute default.
    """

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        max_tokens: int = 1024,
        system: str | None = None,
        timeout: float = 60.0,
        max_retries: int = 2,
    ) -> None:
        from anthropic import AsyncAnthropic  # noqa: F401  (lazy)

        self.model = model or os.environ.get("STREAM_LLM_MODEL", "claude-haiku-4-5-20251001")
        self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.max_tokens = max_tokens
        self.system = system
        self.timeout = timeout
        self.max_retries = max_retries

    async def stream(self, prompt: str) -> AsyncIterator[Chunk]:
        from anthropic import AsyncAnthropic

        client = AsyncAnthropic(
            api_key=self._api_key, timeout=self.timeout, max_retries=self.max_retries
        )
        kwargs: dict = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        if self.system:
            kwargs["system"] = self.system

        index = 0
        try:
            async with client.messages.stream(**kwargs) as stream:
                async for delta in stream.text_stream:
                    if delta:
                        yield Chunk(text=delta, index=index, finish=False)
                        index += 1
            yield Chunk(text="", index=index, finish=True)
        except Exception as exc:  # noqa: BLE001
            raise LLMError(f"anthropic stream failed: {exc}") from exc
