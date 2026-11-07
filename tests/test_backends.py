"""AnthropicLLM client configuration, checked against a stub SDK module."""

from __future__ import annotations

import sys
import types

import pytest

from stream_llm_fastapi.backends import AnthropicLLM


class _StubStream:
    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    @property
    def text_stream(self):
        async def deltas():
            yield "hi"

        return deltas()


@pytest.fixture
def sdk_clients(monkeypatch):
    clients = []

    class AsyncAnthropic:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.messages = types.SimpleNamespace(stream=lambda **_: _StubStream())
            clients.append(self)

    module = types.ModuleType("anthropic")
    module.AsyncAnthropic = AsyncAnthropic
    monkeypatch.setitem(sys.modules, "anthropic", module)
    return clients


@pytest.mark.asyncio
async def test_anthropic_client_gets_explicit_timeout_and_retries(sdk_clients):
    chunks = [c async for c in AnthropicLLM(api_key="test").stream("p")]

    assert [c.text for c in chunks] == ["hi", ""]
    assert sdk_clients[0].kwargs["timeout"] == 60.0
    assert sdk_clients[0].kwargs["max_retries"] == 2


@pytest.mark.asyncio
async def test_anthropic_timeout_and_retries_are_configurable(sdk_clients):
    llm = AnthropicLLM(api_key="test", timeout=5.0, max_retries=0)
    [c async for c in llm.stream("p")]

    assert sdk_clients[0].kwargs["timeout"] == 5.0
    assert sdk_clients[0].kwargs["max_retries"] == 0
