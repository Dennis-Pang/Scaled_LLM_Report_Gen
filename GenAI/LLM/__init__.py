"""
SGLang LLM Module - OpenAI-compatible client for SGLang endpoints.

Usage:
    from LLM import SGLangLLM, test_connection, ensure_server, get_available_gpus

    llm = SGLangLLM()
    response = llm.chat("Hello, how are you?")
"""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from .client import SGLangLLM, ensure_server, get_available_gpus, test_connection

__all__ = ["SGLangLLM", "test_connection", "ensure_server", "get_available_gpus"]


def __getattr__(name: str) -> Any:
    if name in __all__:
        from . import client as _client

        return getattr(_client, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(list(globals().keys()) + __all__)
