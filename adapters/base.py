"""
LLM Adapter Base - Abstract interface for LLM providers.

Provides a unified API for:
- Text completion
- Streaming responses
- Embeddings
- Tool/function calling
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, AsyncIterator

logger = logging.getLogger("titan.adapters")


class LLMProvider(str, Enum):
    """Supported LLM providers."""

    OLLAMA = "ollama"
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    GROQ = "groq"
    LOCAL = "local"


@dataclass
class LLMConfig:
    """Configuration for LLM adapter."""

    provider: LLMProvider
    model: str
    api_key: str | None = None  # allow-secret
    base_url: str | None = None
    temperature: float = 0.7
    max_tokens: int = 4096
    timeout: float = 60.0
    context_window: int = 8192

    # Prompt caching configuration
    enable_prompt_caching: bool = True
    cache_control_type: str = "ephemeral"  # Anthropic cache control type


@dataclass
class LLMMessage:
    """A message in the conversation."""

    role: str  # "user", "assistant", "system"
    content: str
    name: str | None = None
    tool_calls: list[dict[str, Any]] | None = None
    tool_call_id: str | None = None


@dataclass
class LLMResponse:
    """Response from LLM."""

    content: str
    model: str
    provider: str
    finish_reason: str | None = None
    usage: dict[str, int] = field(default_factory=dict)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    raw_response: Any = None

    # Prompt caching metrics
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.usage.get("total_tokens", 0)

    @property
    def cache_hit_ratio(self) -> float:
        """Calculate cache hit ratio (0.0 to 1.0)."""
        total_cache = self.cache_creation_input_tokens + self.cache_read_input_tokens
        if total_cache == 0:
            return 0.0
        return self.cache_read_input_tokens / total_cache


@dataclass
class Tool:
    """Tool/function definition for LLM."""

    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema


class LLMAdapter(ABC):
    """
    Abstract base class for LLM adapters.

    Implementations must provide:
    - complete(): Single completion
    - stream(): Streaming completion
    - embed(): Text embeddings
    """

    provider: LLMProvider

    def __init__(self, config: LLMConfig) -> None:
        self.config = config
        self._client: Any = None

    @abstractmethod
    async def complete(
        self,
        messages: list[LLMMessage],
        *,
        system: str | None = None,
        tools: list[Tool] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """
        Generate a completion.

        Args:
            messages: Conversation messages
            system: System prompt
            tools: Available tools
            temperature: Sampling temperature
            max_tokens: Max output tokens

        Returns:
            LLM response
        """
        pass

    @abstractmethod
    async def stream(
        self,
        messages: list[LLMMessage],
        *,
        system: str | None = None,
        tools: list[Tool] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[str]:
        """
        Stream a completion token by token.

        Args:
            messages: Conversation messages
            system: System prompt
            tools: Available tools
            temperature: Sampling temperature
            max_tokens: Max output tokens

        Yields:
            Response tokens
        """
        pass

    @abstractmethod
    async def embed(self, text: str) -> list[float]:
        """
        Generate embeddings for text.

        Args:
            text: Text to embed

        Returns:
            Embedding vector
        """
        pass

    def supports_tools(self) -> bool:
        """Check if adapter supports tool calling."""
        return True

    def get_context_window(self) -> int:
        """Get context window size."""
        return self.config.context_window

    async def health_check(self) -> bool:
        """Check if the adapter is healthy."""
        try:
            response = await self.complete(
                [LLMMessage(role="user", content="Say 'ok'")],
                max_tokens=5,
            )
            return bool(response.content)
        except Exception:
            return False

    async def __aenter__(self) -> LLMAdapter:
        return self

    async def __aexit__(self, *args: Any) -> None:
        pass


# ============================================================================
# Provider Implementations
# ============================================================================


class OllamaAdapter(LLMAdapter):
    """Adapter for Ollama (local LLM)."""

    provider = LLMProvider.OLLAMA

    def __init__(self, config: LLMConfig) -> None:
        super().__init__(config)
        self.base_url = config.base_url or "http://localhost:11434"

    async def complete(
        self,
        messages: list[LLMMessage],
        *,
        system: str | None = None,
        tools: list[Tool] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        import httpx

        # Build messages
        ollama_messages = []
        if system:
            ollama_messages.append({"role": "system", "content": system})
        for msg in messages:
            ollama_messages.append({"role": msg.role, "content": msg.content})

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": self.config.model,
                    "messages": ollama_messages,
                    "options": {
                        "temperature": temperature or self.config.temperature,
                        "num_predict": max_tokens or self.config.max_tokens,
                    },
                    "stream": False,
                },
                timeout=self.config.timeout,
            )
            response.raise_for_status()
            data = response.json()

        return LLMResponse(
            content=data["message"]["content"],
            model=self.config.model,
            provider=self.provider.value,
            finish_reason="stop",
            usage={
                "prompt_tokens": data.get("prompt_eval_count", 0),
                "completion_tokens": data.get("eval_count", 0),
                "total_tokens": data.get("prompt_eval_count", 0)
                + data.get("eval_count", 0),
            },
            raw_response=data,
        )

    async def stream(
        self,
        messages: list[LLMMessage],
        *,
        system: str | None = None,
        tools: list[Tool] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[str]:
        import httpx
        import json

        ollama_messages = []
        if system:
            ollama_messages.append({"role": "system", "content": system})
        for msg in messages:
            ollama_messages.append({"role": msg.role, "content": msg.content})

        async with httpx.AsyncClient() as client:
            async with client.stream(
                "POST",
                f"{self.base_url}/api/chat",
                json={
                    "model": self.config.model,
                    "messages": ollama_messages,
                    "options": {
                        "temperature": temperature or self.config.temperature,
                        "num_predict": max_tokens or self.config.max_tokens,
                    },
                    "stream": True,
                },
                timeout=self.config.timeout,
            ) as response:
                async for line in response.aiter_lines():
                    if line:
                        data = json.loads(line)
                        if "message" in data and "content" in data["message"]:
                            yield data["message"]["content"]

    async def embed(self, text: str) -> list[float]:
        import httpx

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/api/embeddings",
                json={
                    "model": self.config.model,
                    "prompt": text,
                },
                timeout=self.config.timeout,
            )
            response.raise_for_status()
            data = response.json()

        return data["embedding"]

    def supports_tools(self) -> bool:
        # Most Ollama models don't support native tool calling
        return False


class AnthropicAdapter(LLMAdapter):
    """Adapter for Anthropic Claude API with prompt caching support."""

    provider = LLMProvider.ANTHROPIC

    def __init__(self, config: LLMConfig) -> None:
        super().__init__(config)
        if not config.api_key:
            import os

            config.api_key = os.environ.get("ANTHROPIC_API_KEY")  # allow-secret

    def _apply_cache_control(
        self,
        messages: list[dict[str, Any]],
        system: str | None,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]] | str]:
        """
        Apply cache control to messages for prompt caching.

        The cache_control is applied to the last user message content block,
        which tells Anthropic to cache all content up to that point.

        Returns:
            Tuple of (modified_messages, system_content)
        """
        if not self.config.enable_prompt_caching:
            return messages, system or ""

        # Process system message with cache control if large enough
        system_content: list[dict[str, Any]] | str = system or ""
        if system and len(system) > 1024:  # Only cache substantial system prompts
            system_content = [
                {
                    "type": "text",
                    "text": system,
                    "cache_control": {"type": self.config.cache_control_type},
                }
            ]

        # Apply cache control to last user message
        if messages:
            modified_messages = []
            for i, msg in enumerate(messages):
                if i == len(messages) - 1 and msg["role"] == "user":
                    # Last user message - apply cache control
                    content = msg["content"]
                    if isinstance(content, str):
                        modified_messages.append({
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": content,
                                    "cache_control": {
                                        "type": self.config.cache_control_type
                                    },
                                }
                            ],
                        })
                    else:
                        # Already structured content
                        modified_messages.append(msg)
                else:
                    modified_messages.append(msg)

            return modified_messages, system_content

        return messages, system_content

    async def complete(
        self,
        messages: list[LLMMessage],
        *,
        system: str | None = None,
        tools: list[Tool] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        from anthropic import AsyncAnthropic

        client = AsyncAnthropic(api_key=self.config.api_key)  # allow-secret

        # Convert messages
        anthropic_messages = []
        for msg in messages:
            anthropic_messages.append({"role": msg.role, "content": msg.content})

        # Apply prompt caching
        anthropic_messages, system_content = self._apply_cache_control(
            anthropic_messages, system
        )

        # Convert tools
        anthropic_tools = None
        if tools:
            anthropic_tools = [
                {
                    "name": t.name,
                    "description": t.description,
                    "input_schema": t.parameters,
                }
                for t in tools
            ]

        # Build request kwargs
        request_kwargs: dict[str, Any] = {
            "model": self.config.model,
            "messages": anthropic_messages,
            "max_tokens": max_tokens or self.config.max_tokens,
            "temperature": temperature or self.config.temperature,
        }

        if system_content:
            request_kwargs["system"] = system_content
        if anthropic_tools:
            request_kwargs["tools"] = anthropic_tools

        response = await client.messages.create(**request_kwargs)

        # Extract content
        content = ""
        tool_calls = []
        for block in response.content:
            if hasattr(block, "text"):
                content += block.text
            elif hasattr(block, "type") and block.type == "tool_use":
                tool_calls.append(
                    {
                        "id": block.id,
                        "name": block.name,
                        "arguments": block.input,
                    }
                )

        # Extract cache metrics
        cache_creation = getattr(response.usage, "cache_creation_input_tokens", 0) or 0
        cache_read = getattr(response.usage, "cache_read_input_tokens", 0) or 0

        return LLMResponse(
            content=content,
            model=self.config.model,
            provider=self.provider.value,
            finish_reason=response.stop_reason,
            usage={
                "prompt_tokens": response.usage.input_tokens,
                "completion_tokens": response.usage.output_tokens,
                "total_tokens": response.usage.input_tokens
                + response.usage.output_tokens,
            },
            tool_calls=tool_calls,
            raw_response=response,
            cache_creation_input_tokens=cache_creation,
            cache_read_input_tokens=cache_read,
        )

    async def stream(
        self,
        messages: list[LLMMessage],
        *,
        system: str | None = None,
        tools: list[Tool] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[str]:
        from anthropic import AsyncAnthropic

        client = AsyncAnthropic(api_key=self.config.api_key)  # allow-secret

        anthropic_messages = []
        for msg in messages:
            anthropic_messages.append({"role": msg.role, "content": msg.content})

        async with client.messages.stream(
            model=self.config.model,
            messages=anthropic_messages,
            system=system or "",
            max_tokens=max_tokens or self.config.max_tokens,
            temperature=temperature or self.config.temperature,
        ) as stream:
            async for text in stream.text_stream:
                yield text

    async def embed(self, text: str) -> list[float]:
        # Anthropic doesn't have a native embedding API
        # Fall back to hash embedding
        from hive.memory import hash_embedding

        return hash_embedding(text)


class OpenAIAdapter(LLMAdapter):
    """Adapter for OpenAI API."""

    provider = LLMProvider.OPENAI

    def __init__(self, config: LLMConfig) -> None:
        super().__init__(config)
        if not config.api_key:
            import os

            config.api_key = os.environ.get("OPENAI_API_KEY")  # allow-secret

    async def complete(
        self,
        messages: list[LLMMessage],
        *,
        system: str | None = None,
        tools: list[Tool] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        from openai import AsyncOpenAI

        client = AsyncOpenAI(api_key=self.config.api_key)  # allow-secret

        # Build messages
        openai_messages = []
        if system:
            openai_messages.append({"role": "system", "content": system})
        for msg in messages:
            openai_messages.append({"role": msg.role, "content": msg.content})

        # Convert tools
        openai_tools = None
        if tools:
            openai_tools = [
                {
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.parameters,
                    },
                }
                for t in tools
            ]

        response = await client.chat.completions.create(
            model=self.config.model,
            messages=openai_messages,
            max_tokens=max_tokens or self.config.max_tokens,
            temperature=temperature or self.config.temperature,
            tools=openai_tools,
        )

        choice = response.choices[0]
        content = choice.message.content or ""
        tool_calls = []

        if choice.message.tool_calls:
            for tc in choice.message.tool_calls:
                tool_calls.append(
                    {
                        "id": tc.id,
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    }
                )

        return LLMResponse(
            content=content,
            model=self.config.model,
            provider=self.provider.value,
            finish_reason=choice.finish_reason,
            usage={
                "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                "completion_tokens": response.usage.completion_tokens
                if response.usage
                else 0,
                "total_tokens": response.usage.total_tokens if response.usage else 0,
            },
            tool_calls=tool_calls,
            raw_response=response,
        )

    async def stream(
        self,
        messages: list[LLMMessage],
        *,
        system: str | None = None,
        tools: list[Tool] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[str]:
        from openai import AsyncOpenAI

        client = AsyncOpenAI(api_key=self.config.api_key)  # allow-secret

        openai_messages = []
        if system:
            openai_messages.append({"role": "system", "content": system})
        for msg in messages:
            openai_messages.append({"role": msg.role, "content": msg.content})

        stream = await client.chat.completions.create(
            model=self.config.model,
            messages=openai_messages,
            max_tokens=max_tokens or self.config.max_tokens,
            temperature=temperature or self.config.temperature,
            stream=True,
        )

        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    async def embed(self, text: str) -> list[float]:
        from openai import AsyncOpenAI

        client = AsyncOpenAI(api_key=self.config.api_key)  # allow-secret

        response = await client.embeddings.create(
            model="text-embedding-3-small",
            input=text,
        )

        return response.data[0].embedding


class GroqAdapter(LLMAdapter):
    """Adapter for Groq API (fast inference)."""

    provider = LLMProvider.GROQ

    def __init__(self, config: LLMConfig) -> None:
        super().__init__(config)
        if not config.api_key:
            import os

            config.api_key = os.environ.get("GROQ_API_KEY")  # allow-secret

    async def complete(
        self,
        messages: list[LLMMessage],
        *,
        system: str | None = None,
        tools: list[Tool] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        import httpx

        groq_messages = []
        if system:
            groq_messages.append({"role": "system", "content": system})
        for msg in messages:
            groq_messages.append({"role": msg.role, "content": msg.content})

        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.config.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.config.model,
                    "messages": groq_messages,
                    "max_tokens": max_tokens or self.config.max_tokens,
                    "temperature": temperature or self.config.temperature,
                },
                timeout=self.config.timeout,
            )
            response.raise_for_status()
            data = response.json()

        choice = data["choices"][0]
        return LLMResponse(
            content=choice["message"]["content"],
            model=self.config.model,
            provider=self.provider.value,
            finish_reason=choice.get("finish_reason"),
            usage=data.get("usage", {}),
            raw_response=data,
        )

    async def stream(
        self,
        messages: list[LLMMessage],
        *,
        system: str | None = None,
        tools: list[Tool] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[str]:
        import httpx
        import json

        groq_messages = []
        if system:
            groq_messages.append({"role": "system", "content": system})
        for msg in messages:
            groq_messages.append({"role": msg.role, "content": msg.content})

        async with httpx.AsyncClient() as client:
            async with client.stream(
                "POST",
                "https://api.groq.com/openai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.config.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.config.model,
                    "messages": groq_messages,
                    "max_tokens": max_tokens or self.config.max_tokens,
                    "temperature": temperature or self.config.temperature,
                    "stream": True,
                },
                timeout=self.config.timeout,
            ) as response:
                async for line in response.aiter_lines():
                    if line.startswith("data: ") and line != "data: [DONE]":
                        data = json.loads(line[6:])
                        if data["choices"][0].get("delta", {}).get("content"):
                            yield data["choices"][0]["delta"]["content"]

    async def embed(self, text: str) -> list[float]:
        # Groq doesn't have embeddings, fall back
        from hive.memory import hash_embedding

        return hash_embedding(text)

    def supports_tools(self) -> bool:
        return False
