"""stream-llm-fastapi — SSE streaming for LLMs done right.

Exports:

* :class:`LLMBackend`, :class:`MockLLM`, :class:`AnthropicLLM` — backend protocol + reference implementations
* :class:`StreamSession` — in-process state machine for one streamed completion
* :class:`StreamRegistry` — keeps partial sessions for resume after disconnect
* :func:`create_app` — FastAPI application factory
"""

from stream_llm_fastapi.backends import (
    AnthropicLLM,
    Chunk,
    LLMBackend,
    LLMError,
    MockLLM,
)
from stream_llm_fastapi.session import (
    StreamRegistry,
    StreamSession,
    SessionState,
)
from stream_llm_fastapi.app import create_app

__all__ = [
    "AnthropicLLM",
    "Chunk",
    "LLMBackend",
    "LLMError",
    "MockLLM",
    "StreamRegistry",
    "StreamSession",
    "SessionState",
    "create_app",
]
