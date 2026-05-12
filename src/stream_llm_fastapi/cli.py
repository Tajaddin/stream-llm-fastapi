"""``stream-llm-fastapi`` CLI — launch the demo server."""

from __future__ import annotations

import argparse
import os

import uvicorn

from stream_llm_fastapi.app import create_app
from stream_llm_fastapi.backends import AnthropicLLM, MockLLM


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the stream-llm-fastapi server")
    parser.add_argument("--backend", choices=["mock", "anthropic"], default="mock")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--queue-maxsize", type=int, default=16)
    args = parser.parse_args()

    if args.backend == "anthropic":
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise SystemExit("ANTHROPIC_API_KEY not set")
        factory = lambda: AnthropicLLM()
    else:
        factory = lambda: MockLLM(response="hello from mock " * 20, delay=0.05)

    app = create_app(backend_factory=factory, queue_maxsize=args.queue_maxsize)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
