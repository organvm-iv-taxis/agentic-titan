"""
LLM Adapters - Model-agnostic interface layer.

Provides a unified interface for multiple LLM providers:
- Ollama (local)
- Claude API (Anthropic)
- OpenAI API
- Groq
- Local GGUF models

Inspired by: aionui auto-detect and fallback patterns
"""

from adapters.base import LLMAdapter, LLMResponse, LLMConfig
from adapters.router import LLMRouter, get_router

__all__ = [
    "LLMAdapter",
    "LLMResponse",
    "LLMConfig",
    "LLMRouter",
    "get_router",
]
