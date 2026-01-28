"""
Search Tool - Web search with source citations.

Provides search grounding capabilities:
- Web search via multiple providers
- Source citation tracking
- Result ranking and filtering
- Integration with RAG for context injection

Reference: vendor/cli/gemini-cli search grounding patterns
"""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable

from tools.base import Tool, ToolParameter, ToolResult, register_tool

logger = logging.getLogger("titan.tools.search")


# ============================================================================
# Data Structures
# ============================================================================


@dataclass
class SearchSource:
    """A source from search results."""

    url: str
    title: str
    snippet: str
    domain: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    relevance_score: float = 0.0

    def __post_init__(self) -> None:
        if not self.domain and self.url:
            # Extract domain from URL
            match = re.search(r"https?://([^/]+)", self.url)
            if match:
                self.domain = match.group(1)

    def to_dict(self) -> dict[str, Any]:
        return {
            "url": self.url,
            "title": self.title,
            "snippet": self.snippet,
            "domain": self.domain,
            "relevance_score": self.relevance_score,
        }

    def to_citation(self, index: int) -> str:
        """Format as a numbered citation."""
        return f"[{index}] {self.title} - {self.url}"


@dataclass
class SearchResults:
    """Results from a search query."""

    query: str
    sources: list[SearchSource]
    total_results: int = 0
    search_time_ms: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "sources": [s.to_dict() for s in self.sources],
            "total_results": self.total_results,
            "search_time_ms": self.search_time_ms,
        }

    def format_with_citations(self) -> str:
        """Format results with numbered citations."""
        lines = [f"Search results for: {self.query}\n"]

        for i, source in enumerate(self.sources, start=1):
            lines.append(f"\n[{i}] {source.title}")
            lines.append(f"    URL: {source.url}")
            lines.append(f"    {source.snippet[:200]}...")

        lines.append("\nSources:")
        for i, source in enumerate(self.sources, start=1):
            lines.append(source.to_citation(i))

        return "\n".join(lines)


# ============================================================================
# Search Providers
# ============================================================================


class SearchProvider:
    """Base class for search providers."""

    name: str = "base"

    async def search(
        self,
        query: str,
        max_results: int = 10,
    ) -> SearchResults:
        """Perform a search."""
        raise NotImplementedError


class DuckDuckGoProvider(SearchProvider):
    """DuckDuckGo search provider (no API key required)."""

    name = "duckduckgo"

    async def search(
        self,
        query: str,
        max_results: int = 10,
    ) -> SearchResults:
        """Search using DuckDuckGo HTML scraping."""
        import asyncio

        try:
            import httpx
        except ImportError:
            raise ImportError("httpx is required for web search")

        start_time = datetime.now()
        sources: list[SearchSource] = []

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                # Use DuckDuckGo HTML interface
                url = "https://html.duckduckgo.com/html/"
                response = await client.post(
                    url,
                    data={"q": query},
                    headers={
                        "User-Agent": "Mozilla/5.0 (compatible; TitanAgent/1.0)"
                    },
                )
                response.raise_for_status()
                html = response.text

                # Parse results (basic HTML parsing)
                # Look for result links
                result_pattern = re.compile(
                    r'<a[^>]+class="result__a"[^>]+href="([^"]+)"[^>]*>([^<]+)</a>',
                    re.IGNORECASE,
                )
                snippet_pattern = re.compile(
                    r'<a[^>]+class="result__snippet"[^>]*>([^<]+)</a>',
                    re.IGNORECASE,
                )

                urls = result_pattern.findall(html)
                snippets = snippet_pattern.findall(html)

                for i, (url, title) in enumerate(urls[:max_results]):
                    snippet = snippets[i] if i < len(snippets) else ""
                    # Clean up HTML entities
                    title = re.sub(r"&[^;]+;", " ", title).strip()
                    snippet = re.sub(r"&[^;]+;", " ", snippet).strip()

                    sources.append(
                        SearchSource(
                            url=url,
                            title=title,
                            snippet=snippet,
                            relevance_score=1.0 - (i * 0.1),
                        )
                    )

        except Exception as e:
            logger.warning(f"DuckDuckGo search failed: {e}")

        search_time = int((datetime.now() - start_time).total_seconds() * 1000)

        return SearchResults(
            query=query,
            sources=sources,
            total_results=len(sources),
            search_time_ms=search_time,
        )


class MockSearchProvider(SearchProvider):
    """Mock search provider for testing."""

    name = "mock"

    async def search(
        self,
        query: str,
        max_results: int = 10,
    ) -> SearchResults:
        """Return mock search results."""
        sources = []
        for i in range(min(3, max_results)):
            sources.append(
                SearchSource(
                    url=f"https://example.com/{query.replace(' ', '-')}-{i}",
                    title=f"Result {i + 1} for: {query}",
                    snippet=f"This is a mock result for the search query '{query}'. "
                    f"It provides relevant information about the topic.",
                    relevance_score=1.0 - (i * 0.2),
                )
            )

        return SearchResults(
            query=query,
            sources=sources,
            total_results=len(sources),
            search_time_ms=50,
        )


# ============================================================================
# Provider Registry
# ============================================================================

_providers: dict[str, SearchProvider] = {
    "duckduckgo": DuckDuckGoProvider(),
    "mock": MockSearchProvider(),
}


def get_provider(name: str = "duckduckgo") -> SearchProvider:
    """Get a search provider by name."""
    if name not in _providers:
        logger.warning(f"Unknown provider '{name}', using mock")
        return _providers["mock"]
    return _providers[name]


def register_provider(provider: SearchProvider) -> None:
    """Register a custom search provider."""
    _providers[provider.name] = provider


# ============================================================================
# Search Tool Implementation
# ============================================================================


class SearchTool(Tool):
    """
    Web search tool with source citations.

    Allows agents to search the web and get grounded results
    with proper source citations.
    """

    @property
    def name(self) -> str:
        return "search"

    @property
    def description(self) -> str:
        return (
            "Web search tool with source citations. "
            "Use to find current information, documentation, or answers from the web. "
            "Results include URLs and snippets that should be cited."
        )

    @property
    def parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="query",
                type="string",
                description="Search query",
                required=True,
            ),
            ToolParameter(
                name="max_results",
                type="integer",
                description="Maximum number of results (default: 5)",
                required=False,
            ),
            ToolParameter(
                name="provider",
                type="string",
                description="Search provider: 'duckduckgo' (default) or 'mock'",
                required=False,
            ),
            ToolParameter(
                name="format",
                type="string",
                description="Output format: 'full' (default) or 'citations'",
                required=False,
                enum=["full", "citations"],
            ),
        ]

    async def execute(self, **kwargs: Any) -> ToolResult:
        query = kwargs.get("query", "")
        if not query:
            return ToolResult(
                success=False,
                output=None,
                error="'query' parameter is required",
            )

        max_results = kwargs.get("max_results", 5)
        provider_name = kwargs.get("provider", "duckduckgo")
        output_format = kwargs.get("format", "full")

        try:
            provider = get_provider(provider_name)
            results = await provider.search(query, max_results)

            if output_format == "citations":
                return ToolResult(
                    success=True,
                    output=results.format_with_citations(),
                )
            else:
                return ToolResult(
                    success=True,
                    output=results.to_dict(),
                )

        except Exception as e:
            logger.exception(f"Search tool error: {e}")
            return ToolResult(
                success=False,
                output=None,
                error=f"Search failed: {e}",
            )


# Register the tool
search_tool = SearchTool()
register_tool(search_tool)


# ============================================================================
# Citation Helper
# ============================================================================


class CitationManager:
    """
    Manages citations across a conversation.

    Tracks sources used and generates citation lists.
    """

    def __init__(self) -> None:
        self._sources: dict[str, SearchSource] = {}
        self._citation_index: int = 0

    def add_source(self, source: SearchSource) -> int:
        """
        Add a source and return its citation number.

        Args:
            source: SearchSource to add

        Returns:
            Citation number (1-indexed)
        """
        # Use URL as key to deduplicate
        if source.url not in self._sources:
            self._citation_index += 1
            source.relevance_score = self._citation_index
            self._sources[source.url] = source

        # Return the citation number
        for i, (url, _) in enumerate(self._sources.items(), start=1):
            if url == source.url:
                return i
        return self._citation_index

    def get_citations(self) -> list[str]:
        """Get formatted citation list."""
        citations = []
        for i, source in enumerate(self._sources.values(), start=1):
            citations.append(source.to_citation(i))
        return citations

    def format_citations_block(self) -> str:
        """Format all citations as a block."""
        if not self._sources:
            return ""

        lines = ["\n---\n**Sources:**"]
        for i, source in enumerate(self._sources.values(), start=1):
            lines.append(f"[{i}] [{source.title}]({source.url})")
        return "\n".join(lines)

    def clear(self) -> None:
        """Clear all citations."""
        self._sources.clear()
        self._citation_index = 0
