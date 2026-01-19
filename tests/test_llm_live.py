"""
Live tests for LLM adapters.

Run with: python -m pytest tests/test_llm_live.py -v -s

These tests require actual LLM services:
- Ollama: Running locally on port 11434
- Anthropic: ANTHROPIC_API_KEY env var
- OpenAI: OPENAI_API_KEY env var
"""

from __future__ import annotations

import asyncio
import os
import pytest

from adapters.base import LLMMessage, LLMConfig, LLMProvider, OllamaAdapter, AnthropicAdapter, OpenAIAdapter
from adapters.router import LLMRouter, get_router, reset_router


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def ollama_available() -> bool:
    """Check if Ollama is running."""
    import httpx
    try:
        r = httpx.get("http://localhost:11434/api/tags", timeout=2)
        return r.status_code == 200
    except Exception:
        return False


@pytest.fixture
def anthropic_available() -> bool:
    """Check if Anthropic API key is set."""
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


@pytest.fixture
def openai_available() -> bool:
    """Check if OpenAI API key is set."""
    return bool(os.environ.get("OPENAI_API_KEY"))


# ============================================================================
# Router Tests
# ============================================================================

class TestRouter:
    """Test the LLM router."""

    @pytest.fixture(autouse=True)
    async def reset(self):
        """Reset router between tests."""
        await reset_router()
        yield
        await reset_router()

    @pytest.mark.asyncio
    async def test_router_initialization(self):
        """Test router auto-detects providers."""
        router = LLMRouter()
        await router.initialize()

        providers = router.list_providers()
        assert len(providers) > 0

        # Print what's available
        print("\n=== Available Providers ===")
        for p in providers:
            status = "✓" if p.available else "✗"
            print(f"  {status} {p.provider.value}: models={p.models[:3]}")

    @pytest.mark.asyncio
    async def test_router_complete(self):
        """Test router can complete a simple request."""
        router = get_router()
        await router.initialize()

        available = router.list_available_providers()
        if not available:
            pytest.skip("No LLM providers available")

        messages = [LLMMessage(role="user", content="Say 'hello' and nothing else.")]
        response = await router.complete(messages, max_tokens=20)

        print(f"\n=== Router Response ({response.provider}) ===")
        print(f"  Content: {response.content}")
        print(f"  Tokens: {response.total_tokens}")

        assert response.content
        assert "hello" in response.content.lower()


# ============================================================================
# Ollama Tests
# ============================================================================

class TestOllama:
    """Test Ollama adapter."""

    @pytest.mark.asyncio
    async def test_ollama_complete(self, ollama_available: bool):
        """Test Ollama completion."""
        if not ollama_available:
            pytest.skip("Ollama not running")

        # Get first available model
        import httpx
        r = httpx.get("http://localhost:11434/api/tags")
        models = r.json().get("models", [])
        if not models:
            pytest.skip("No Ollama models available")

        model = models[0]["name"]
        print(f"\n=== Using Ollama model: {model} ===")

        config = LLMConfig(provider=LLMProvider.OLLAMA, model=model)
        adapter = OllamaAdapter(config)

        messages = [LLMMessage(role="user", content="What is 2+2? Answer with just the number.")]
        response = await adapter.complete(messages, max_tokens=10)

        print(f"  Response: {response.content}")
        assert "4" in response.content

    @pytest.mark.asyncio
    async def test_ollama_stream(self, ollama_available: bool):
        """Test Ollama streaming."""
        if not ollama_available:
            pytest.skip("Ollama not running")

        import httpx
        r = httpx.get("http://localhost:11434/api/tags")
        models = r.json().get("models", [])
        if not models:
            pytest.skip("No Ollama models available")

        model = models[0]["name"]
        config = LLMConfig(provider=LLMProvider.OLLAMA, model=model)
        adapter = OllamaAdapter(config)

        messages = [LLMMessage(role="user", content="Count from 1 to 5.")]

        print(f"\n=== Ollama Streaming ({model}) ===")
        print("  ", end="")
        full_response = ""
        async for token in adapter.stream(messages, max_tokens=50):
            print(token, end="", flush=True)
            full_response += token
        print()

        assert full_response


# ============================================================================
# Anthropic Tests
# ============================================================================

class TestAnthropic:
    """Test Anthropic adapter."""

    @pytest.mark.asyncio
    async def test_anthropic_complete(self, anthropic_available: bool):
        """Test Anthropic completion."""
        if not anthropic_available:
            pytest.skip("ANTHROPIC_API_KEY not set")

        config = LLMConfig(
            provider=LLMProvider.ANTHROPIC,
            model="claude-3-5-haiku-20241022",
        )
        adapter = AnthropicAdapter(config)

        messages = [LLMMessage(role="user", content="What is the capital of France? One word answer.")]
        response = await adapter.complete(messages, max_tokens=20)

        print(f"\n=== Anthropic Response ===")
        print(f"  Content: {response.content}")
        print(f"  Tokens: {response.total_tokens}")

        assert "paris" in response.content.lower()

    @pytest.mark.asyncio
    async def test_anthropic_stream(self, anthropic_available: bool):
        """Test Anthropic streaming."""
        if not anthropic_available:
            pytest.skip("ANTHROPIC_API_KEY not set")

        config = LLMConfig(
            provider=LLMProvider.ANTHROPIC,
            model="claude-3-5-haiku-20241022",
        )
        adapter = AnthropicAdapter(config)

        messages = [LLMMessage(role="user", content="Count from 1 to 5.")]

        print(f"\n=== Anthropic Streaming ===")
        print("  ", end="")
        full_response = ""
        async for token in adapter.stream(messages, max_tokens=50):
            print(token, end="", flush=True)
            full_response += token
        print()

        assert full_response


# ============================================================================
# OpenAI Tests
# ============================================================================

class TestOpenAI:
    """Test OpenAI adapter."""

    @pytest.mark.asyncio
    async def test_openai_complete(self, openai_available: bool):
        """Test OpenAI completion."""
        if not openai_available:
            pytest.skip("OPENAI_API_KEY not set")

        config = LLMConfig(
            provider=LLMProvider.OPENAI,
            model="gpt-4o-mini",
        )
        adapter = OpenAIAdapter(config)

        messages = [LLMMessage(role="user", content="What color is the sky? One word answer.")]
        response = await adapter.complete(messages, max_tokens=20)

        print(f"\n=== OpenAI Response ===")
        print(f"  Content: {response.content}")
        print(f"  Tokens: {response.total_tokens}")

        assert "blue" in response.content.lower()

    @pytest.mark.asyncio
    async def test_openai_embed(self, openai_available: bool):
        """Test OpenAI embeddings."""
        if not openai_available:
            pytest.skip("OPENAI_API_KEY not set")

        config = LLMConfig(
            provider=LLMProvider.OPENAI,
            model="gpt-4o-mini",
        )
        adapter = OpenAIAdapter(config)

        embedding = await adapter.embed("Hello, world!")

        print(f"\n=== OpenAI Embedding ===")
        print(f"  Dimensions: {len(embedding)}")
        print(f"  First 5: {embedding[:5]}")

        assert len(embedding) > 0
        assert all(isinstance(x, float) for x in embedding)


# ============================================================================
# Integration Test
# ============================================================================

class TestIntegration:
    """Integration tests with real agents."""

    @pytest.mark.asyncio
    async def test_researcher_agent(self):
        """Test ResearcherAgent with real LLM."""
        await reset_router()
        router = get_router()
        await router.initialize()

        if not router.list_available_providers():
            pytest.skip("No LLM providers available")

        from agents.archetypes.researcher import ResearcherAgent

        agent = ResearcherAgent(topic="Python type hints")
        result = await agent.run()

        print(f"\n=== Researcher Agent Result ===")
        print(f"  Success: {result.success}")
        print(f"  Turns: {result.turns_taken}")
        print(f"  Questions: {len(result.result.questions) if result.result else 0}")
        if result.result and result.result.summary:
            print(f"  Summary: {result.result.summary[:200]}...")

        assert result.success
        assert result.result
        assert result.result.questions
        assert result.result.summary


if __name__ == "__main__":
    # Quick test
    async def main():
        print("Testing LLM adapters...")

        router = get_router()
        await router.initialize()

        available = router.list_available_providers()
        print(f"Available providers: {[p.value for p in available]}")

        if available:
            messages = [LLMMessage(role="user", content="Say hello")]
            response = await router.complete(messages, max_tokens=20)
            print(f"Response: {response.content}")
        else:
            print("No providers available. Set API keys or start Ollama.")

    asyncio.run(main())
