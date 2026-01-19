"""
LLM Router - Intelligent provider selection and fallback.

Features:
- Auto-detection of available providers
- Fallback chains
- Cost-based routing
- Capability-based selection

Inspired by: aionui auto-detect and fallback patterns
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, AsyncIterator

from adapters.base import (
    LLMAdapter,
    LLMConfig,
    LLMMessage,
    LLMProvider,
    LLMResponse,
    OllamaAdapter,
    AnthropicAdapter,
    OpenAIAdapter,
    GroqAdapter,
    Tool,
)
from agents.framework.errors import LLMAdapterError

logger = logging.getLogger("titan.adapters.router")


class RoutingStrategy(str, Enum):
    """Routing strategies for provider selection."""

    COST_OPTIMIZED = "cost_optimized"   # Prefer cheaper/local options
    QUALITY_FIRST = "quality_first"     # Prefer best quality
    SPEED_FIRST = "speed_first"         # Prefer fastest
    ROUND_ROBIN = "round_robin"         # Rotate between providers
    FALLBACK = "fallback"               # Use fallback chain


@dataclass
class ProviderInfo:
    """Information about a provider."""

    provider: LLMProvider
    available: bool
    models: list[str]
    supports_tools: bool
    supports_streaming: bool
    cost_tier: int  # 1=free, 2=cheap, 3=standard, 4=expensive
    quality_tier: int  # 1=basic, 2=good, 3=great, 4=best
    speed_tier: int  # 1=slow, 2=medium, 3=fast, 4=fastest


# Known provider characteristics
PROVIDER_INFO: dict[LLMProvider, dict[str, Any]] = {
    LLMProvider.OLLAMA: {
        "cost_tier": 1,
        "quality_tier": 2,
        "speed_tier": 2,
        "supports_tools": False,
    },
    LLMProvider.ANTHROPIC: {
        "cost_tier": 3,
        "quality_tier": 4,
        "speed_tier": 3,
        "supports_tools": True,
    },
    LLMProvider.OPENAI: {
        "cost_tier": 3,
        "quality_tier": 4,
        "speed_tier": 3,
        "supports_tools": True,
    },
    LLMProvider.GROQ: {
        "cost_tier": 2,
        "quality_tier": 3,
        "speed_tier": 4,
        "supports_tools": False,
    },
}


# Default models per provider
DEFAULT_MODELS: dict[LLMProvider, str] = {
    LLMProvider.OLLAMA: "llama3.2",
    LLMProvider.ANTHROPIC: "claude-sonnet-4-20250514",
    LLMProvider.OPENAI: "gpt-4o-mini",
    LLMProvider.GROQ: "llama-3.3-70b-versatile",
}


class LLMRouter:
    """
    Routes requests to appropriate LLM providers.

    Handles:
    - Provider auto-detection
    - Fallback chains
    - Cost/quality/speed optimization
    - Capability matching
    """

    def __init__(
        self,
        strategy: RoutingStrategy = RoutingStrategy.FALLBACK,
        preferred_providers: list[LLMProvider] | None = None,
    ) -> None:
        self.strategy = strategy
        self.preferred_providers = preferred_providers or []

        self._providers: dict[LLMProvider, ProviderInfo] = {}
        self._adapters: dict[LLMProvider, LLMAdapter] = {}
        self._fallback_chain: list[LLMProvider] = []
        self._initialized = False

    async def initialize(self) -> None:
        """Detect and initialize available providers."""
        if self._initialized:
            return

        # Check each provider
        for provider in LLMProvider:
            available, models = await self._check_provider(provider)
            info = PROVIDER_INFO.get(provider, {})

            self._providers[provider] = ProviderInfo(
                provider=provider,
                available=available,
                models=models,
                supports_tools=info.get("supports_tools", False),
                supports_streaming=True,
                cost_tier=info.get("cost_tier", 3),
                quality_tier=info.get("quality_tier", 2),
                speed_tier=info.get("speed_tier", 2),
            )

            if available:
                logger.info(f"Provider available: {provider.value} ({len(models)} models)")

        # Build fallback chain based on strategy
        self._build_fallback_chain()

        self._initialized = True
        logger.info(
            f"Router initialized with strategy={self.strategy.value}, "
            f"available={[p.value for p, i in self._providers.items() if i.available]}"
        )

    async def _check_provider(self, provider: LLMProvider) -> tuple[bool, list[str]]:
        """Check if a provider is available."""
        try:
            if provider == LLMProvider.OLLAMA:
                return await self._check_ollama()
            elif provider == LLMProvider.ANTHROPIC:
                return self._check_anthropic()
            elif provider == LLMProvider.OPENAI:
                return self._check_openai()
            elif provider == LLMProvider.GROQ:
                return self._check_groq()
        except Exception as e:
            logger.debug(f"Provider {provider.value} check failed: {e}")
        return False, []

    async def _check_ollama(self) -> tuple[bool, list[str]]:
        """Check Ollama availability."""
        try:
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    "http://localhost:11434/api/tags",
                    timeout=2.0,
                )
                if response.status_code == 200:
                    data = response.json()
                    models = [m["name"] for m in data.get("models", [])]
                    return True, models
        except Exception:
            pass
        return False, []

    def _check_anthropic(self) -> tuple[bool, list[str]]:
        """Check Anthropic API availability."""
        if os.environ.get("ANTHROPIC_API_KEY"):
            return True, [
                "claude-opus-4-20250514",
                "claude-sonnet-4-20250514",
                "claude-3-5-haiku-20241022",
            ]
        return False, []

    def _check_openai(self) -> tuple[bool, list[str]]:
        """Check OpenAI API availability."""
        if os.environ.get("OPENAI_API_KEY"):
            return True, ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo"]
        return False, []

    def _check_groq(self) -> tuple[bool, list[str]]:
        """Check Groq API availability."""
        if os.environ.get("GROQ_API_KEY"):
            return True, [
                "llama-3.3-70b-versatile",
                "mixtral-8x7b-32768",
                "gemma2-9b-it",
            ]
        return False, []

    def _build_fallback_chain(self) -> None:
        """Build the fallback chain based on strategy."""
        available = [p for p, i in self._providers.items() if i.available]

        if self.preferred_providers:
            # Use preferred order, then add remaining
            chain = [p for p in self.preferred_providers if p in available]
            chain.extend([p for p in available if p not in chain])
        else:
            # Build based on strategy
            if self.strategy == RoutingStrategy.COST_OPTIMIZED:
                chain = sorted(
                    available,
                    key=lambda p: self._providers[p].cost_tier,
                )
            elif self.strategy == RoutingStrategy.QUALITY_FIRST:
                chain = sorted(
                    available,
                    key=lambda p: -self._providers[p].quality_tier,
                )
            elif self.strategy == RoutingStrategy.SPEED_FIRST:
                chain = sorted(
                    available,
                    key=lambda p: -self._providers[p].speed_tier,
                )
            else:
                # Default fallback order
                chain = available

        self._fallback_chain = chain
        logger.debug(f"Fallback chain: {[p.value for p in chain]}")

    def _get_adapter(self, provider: LLMProvider, model: str | None = None) -> LLMAdapter:
        """Get or create an adapter for a provider."""
        if provider in self._adapters:
            return self._adapters[provider]

        # Get default model or use first available from provider
        model = model or DEFAULT_MODELS.get(provider, "default")

        # For Ollama, use first available model if default isn't available
        if provider == LLMProvider.OLLAMA and provider in self._providers:
            available_models = self._providers[provider].models
            if available_models and model not in available_models:
                model = available_models[0]
                logger.info(f"Using available Ollama model: {model}")

        config = LLMConfig(
            provider=provider,
            model=model,
        )

        adapter: LLMAdapter
        if provider == LLMProvider.OLLAMA:
            adapter = OllamaAdapter(config)
        elif provider == LLMProvider.ANTHROPIC:
            adapter = AnthropicAdapter(config)
        elif provider == LLMProvider.OPENAI:
            adapter = OpenAIAdapter(config)
        elif provider == LLMProvider.GROQ:
            adapter = GroqAdapter(config)
        else:
            raise LLMAdapterError(f"Unknown provider: {provider}")

        self._adapters[provider] = adapter
        return adapter

    def _select_provider(
        self,
        requires_tools: bool = False,
        preferred_model: str | None = None,
    ) -> LLMProvider:
        """Select the best provider for a request."""
        for provider in self._fallback_chain:
            info = self._providers[provider]

            # Check tool requirement
            if requires_tools and not info.supports_tools:
                continue

            return provider

        raise LLMAdapterError("No suitable provider available")

    async def complete(
        self,
        messages: list[LLMMessage],
        *,
        system: str | None = None,
        tools: list[Tool] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        provider: LLMProvider | None = None,
        model: str | None = None,
    ) -> LLMResponse:
        """
        Route a completion request.

        Args:
            messages: Conversation messages
            system: System prompt
            tools: Available tools
            temperature: Sampling temperature
            max_tokens: Max output tokens
            provider: Force specific provider
            model: Force specific model

        Returns:
            LLM response
        """
        await self._ensure_initialized()

        # Select provider
        if provider:
            if not self._providers[provider].available:
                raise LLMAdapterError(f"Provider {provider.value} not available")
        else:
            provider = self._select_provider(requires_tools=bool(tools))

        adapter = self._get_adapter(provider, model)

        # Try with fallback
        last_error: Exception | None = None
        for fallback_provider in self._fallback_chain:
            if provider and fallback_provider != provider:
                continue

            try:
                adapter = self._get_adapter(fallback_provider, model)
                response = await adapter.complete(
                    messages,
                    system=system,
                    tools=tools,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                return response
            except Exception as e:
                last_error = e
                logger.warning(f"Provider {fallback_provider.value} failed: {e}")
                if provider:
                    break  # Don't fallback if specific provider requested

        raise LLMAdapterError(
            f"All providers failed: {last_error}",
            provider=provider.value if provider else None,
        )

    async def stream(
        self,
        messages: list[LLMMessage],
        *,
        system: str | None = None,
        tools: list[Tool] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        provider: LLMProvider | None = None,
        model: str | None = None,
    ) -> AsyncIterator[str]:
        """
        Route a streaming request.

        Args:
            messages: Conversation messages
            system: System prompt
            tools: Available tools
            temperature: Sampling temperature
            max_tokens: Max output tokens
            provider: Force specific provider
            model: Force specific model

        Yields:
            Response tokens
        """
        await self._ensure_initialized()

        if provider:
            if not self._providers[provider].available:
                raise LLMAdapterError(f"Provider {provider.value} not available")
        else:
            provider = self._select_provider(requires_tools=bool(tools))

        adapter = self._get_adapter(provider, model)

        async for token in adapter.stream(
            messages,
            system=system,
            tools=tools,
            temperature=temperature,
            max_tokens=max_tokens,
        ):
            yield token

    async def embed(
        self,
        text: str,
        provider: LLMProvider | None = None,
    ) -> list[float]:
        """
        Generate embeddings.

        Args:
            text: Text to embed
            provider: Force specific provider

        Returns:
            Embedding vector
        """
        await self._ensure_initialized()

        # Prefer OpenAI for embeddings if available
        if not provider:
            if self._providers.get(LLMProvider.OPENAI, ProviderInfo(
                provider=LLMProvider.OPENAI,
                available=False,
                models=[],
                supports_tools=False,
                supports_streaming=False,
                cost_tier=3,
                quality_tier=3,
                speed_tier=3,
            )).available:
                provider = LLMProvider.OPENAI
            else:
                provider = self._fallback_chain[0] if self._fallback_chain else LLMProvider.OLLAMA

        adapter = self._get_adapter(provider)
        return await adapter.embed(text)

    def list_providers(self) -> list[ProviderInfo]:
        """List all providers and their status."""
        return list(self._providers.values())

    def list_available_providers(self) -> list[LLMProvider]:
        """List available providers."""
        return [p for p, i in self._providers.items() if i.available]

    async def health_check(self) -> dict[str, bool]:
        """Check health of all providers."""
        await self._ensure_initialized()
        health = {}
        for provider, info in self._providers.items():
            if info.available:
                adapter = self._get_adapter(provider)
                health[provider.value] = await adapter.health_check()
            else:
                health[provider.value] = False
        return health

    async def _ensure_initialized(self) -> None:
        """Ensure router is initialized."""
        if not self._initialized:
            await self.initialize()

    def __repr__(self) -> str:
        available = [p.value for p in self.list_available_providers()]
        return f"<LLMRouter strategy={self.strategy.value} available={available}>"


# Singleton router
_default_router: LLMRouter | None = None


def get_router(
    strategy: RoutingStrategy = RoutingStrategy.FALLBACK,
) -> LLMRouter:
    """Get the default LLM router."""
    global _default_router
    if _default_router is None:
        _default_router = LLMRouter(strategy=strategy)
    return _default_router


async def reset_router() -> None:
    """Reset the default router."""
    global _default_router
    _default_router = None
