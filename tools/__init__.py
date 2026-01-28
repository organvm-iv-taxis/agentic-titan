"""
Titan Tools - Executable tools for agents.

Provides:
- Tool protocol and registry
- Built-in tools (file, web, shell)
- RAG (Retrieval Augmented Generation)
- Document processing (PDF, DOCX, XLSX, PPTX)
- Web search with citations
- MCP bridge for external tools
"""

from tools.base import (
    Tool,
    ToolResult,
    ToolParameter,
    ToolRegistry,
    get_registry,
    register_tool,
)
from tools.executor import ToolExecutor, get_executor

# Import tools to register them
from tools.rag import RAGTool, RAGStore, get_store as get_rag_store
from tools.pdf import PDFTool, extract_pdf_text
from tools.search import SearchTool, SearchResults, CitationManager
from tools.documents import DOCXTool, XLSXTool, PPTXTool

__all__ = [
    # Base
    "Tool",
    "ToolResult",
    "ToolParameter",
    "ToolRegistry",
    "get_registry",
    "register_tool",
    # Executor
    "ToolExecutor",
    "get_executor",
    # RAG
    "RAGTool",
    "RAGStore",
    "get_rag_store",
    # PDF
    "PDFTool",
    "extract_pdf_text",
    # Search
    "SearchTool",
    "SearchResults",
    "CitationManager",
    # Documents
    "DOCXTool",
    "XLSXTool",
    "PPTXTool",
]
