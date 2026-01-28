"""
PDF Tool - PDF document processing and extraction.

Provides capabilities for:
- Text extraction from PDFs
- Page-level parsing
- Metadata extraction
- Integration with RAG for document indexing

Reference: vendor/cookbooks/claude-cookbooks/ PDF processing patterns
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from tools.base import Tool, ToolParameter, ToolResult, register_tool

logger = logging.getLogger("titan.tools.pdf")


# ============================================================================
# Data Structures
# ============================================================================


@dataclass
class PDFPage:
    """A single page from a PDF document."""

    page_number: int
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "page_number": self.page_number,
            "text": self.text,
            "char_count": len(self.text),
            "metadata": self.metadata,
        }


@dataclass
class PDFDocument:
    """A parsed PDF document."""

    path: str
    pages: list[PDFPage]
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def page_count(self) -> int:
        return len(self.pages)

    @property
    def total_text(self) -> str:
        """Get all text concatenated."""
        return "\n\n".join(page.text for page in self.pages)

    @property
    def total_chars(self) -> int:
        return sum(len(page.text) for page in self.pages)

    def get_page(self, page_number: int) -> PDFPage | None:
        """Get a specific page (1-indexed)."""
        for page in self.pages:
            if page.page_number == page_number:
                return page
        return None

    def get_text_range(
        self,
        start_page: int = 1,
        end_page: int | None = None,
    ) -> str:
        """Get text from a range of pages."""
        end_page = end_page or self.page_count
        texts = []
        for page in self.pages:
            if start_page <= page.page_number <= end_page:
                texts.append(f"[Page {page.page_number}]\n{page.text}")
        return "\n\n".join(texts)

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "page_count": self.page_count,
            "total_chars": self.total_chars,
            "metadata": self.metadata,
            "pages": [p.to_dict() for p in self.pages],
        }


# ============================================================================
# PDF Processing Functions
# ============================================================================


def extract_pdf_pypdf(path: str | Path) -> PDFDocument:
    """
    Extract text from PDF using pypdf.

    Args:
        path: Path to PDF file

    Returns:
        PDFDocument with extracted content
    """
    try:
        from pypdf import PdfReader
    except ImportError:
        raise ImportError(
            "pypdf is required for PDF processing. "
            "Install with: pip install 'agentic-titan[documents]'"
        )

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"PDF file not found: {path}")

    reader = PdfReader(str(path))
    pages: list[PDFPage] = []

    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        # Clean up extracted text
        text = _clean_extracted_text(text)
        pages.append(PDFPage(page_number=i, text=text))

    # Extract metadata
    metadata = {}
    if reader.metadata:
        for key in ["/Title", "/Author", "/Subject", "/Creator", "/Producer"]:
            value = reader.metadata.get(key)
            if value:
                # Remove leading slash from key
                clean_key = key.lstrip("/").lower()
                metadata[clean_key] = str(value)

    return PDFDocument(
        path=str(path),
        pages=pages,
        metadata=metadata,
    )


def _clean_extracted_text(text: str) -> str:
    """Clean up extracted PDF text."""
    # Remove excessive whitespace
    text = re.sub(r"\s+", " ", text)
    # Fix common extraction issues
    text = re.sub(r"(\w)-\s+(\w)", r"\1\2", text)  # Rejoin hyphenated words
    # Normalize line endings
    text = text.strip()
    return text


def extract_pdf_text(path: str | Path) -> str:
    """
    Simple text extraction from PDF.

    Args:
        path: Path to PDF file

    Returns:
        Extracted text content
    """
    doc = extract_pdf_pypdf(path)
    return doc.total_text


# ============================================================================
# PDF Tool Implementation
# ============================================================================


class PDFTool(Tool):
    """
    PDF document processing tool.

    Allows agents to:
    - Extract text from PDF documents
    - Get specific pages
    - Extract metadata
    - Search within documents
    """

    @property
    def name(self) -> str:
        return "pdf"

    @property
    def description(self) -> str:
        return (
            "PDF document processing tool. "
            "Actions: 'extract' (get all text), 'pages' (get specific pages), "
            "'metadata' (get document info), 'search' (find text in document)."
        )

    @property
    def parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="action",
                type="string",
                description="Action to perform: 'extract', 'pages', 'metadata', 'search'",
                required=True,
                enum=["extract", "pages", "metadata", "search"],
            ),
            ToolParameter(
                name="path",
                type="string",
                description="Path to the PDF file",
                required=True,
            ),
            ToolParameter(
                name="start_page",
                type="integer",
                description="Starting page number (1-indexed, for 'pages' action)",
                required=False,
            ),
            ToolParameter(
                name="end_page",
                type="integer",
                description="Ending page number (inclusive, for 'pages' action)",
                required=False,
            ),
            ToolParameter(
                name="query",
                type="string",
                description="Search query (for 'search' action)",
                required=False,
            ),
            ToolParameter(
                name="max_chars",
                type="integer",
                description="Maximum characters to return (default: 50000)",
                required=False,
            ),
        ]

    async def execute(self, **kwargs: Any) -> ToolResult:
        action = kwargs.get("action", "")
        path = kwargs.get("path", "")
        max_chars = kwargs.get("max_chars", 50000)

        if not path:
            return ToolResult(
                success=False,
                output=None,
                error="'path' parameter is required",
            )

        try:
            doc = extract_pdf_pypdf(path)

            if action == "extract":
                text = doc.total_text
                if len(text) > max_chars:
                    text = text[:max_chars] + f"\n\n[Truncated: {len(doc.total_text)} total chars]"

                return ToolResult(
                    success=True,
                    output={
                        "text": text,
                        "page_count": doc.page_count,
                        "total_chars": doc.total_chars,
                        "truncated": len(doc.total_text) > max_chars,
                    },
                )

            elif action == "pages":
                start_page = kwargs.get("start_page", 1)
                end_page = kwargs.get("end_page", doc.page_count)

                # Validate page numbers
                if start_page < 1:
                    start_page = 1
                if end_page > doc.page_count:
                    end_page = doc.page_count

                text = doc.get_text_range(start_page, end_page)
                if len(text) > max_chars:
                    text = text[:max_chars] + "\n\n[Truncated]"

                return ToolResult(
                    success=True,
                    output={
                        "text": text,
                        "start_page": start_page,
                        "end_page": end_page,
                        "page_count": doc.page_count,
                    },
                )

            elif action == "metadata":
                return ToolResult(
                    success=True,
                    output={
                        "path": doc.path,
                        "page_count": doc.page_count,
                        "total_chars": doc.total_chars,
                        "metadata": doc.metadata,
                    },
                )

            elif action == "search":
                query = kwargs.get("query", "")
                if not query:
                    return ToolResult(
                        success=False,
                        output=None,
                        error="'query' parameter required for search action",
                    )

                # Search for query in all pages
                results = []
                query_lower = query.lower()

                for page in doc.pages:
                    if query_lower in page.text.lower():
                        # Find context around match
                        text_lower = page.text.lower()
                        idx = text_lower.find(query_lower)
                        start = max(0, idx - 100)
                        end = min(len(page.text), idx + len(query) + 100)
                        context = page.text[start:end]
                        if start > 0:
                            context = "..." + context
                        if end < len(page.text):
                            context = context + "..."

                        results.append({
                            "page": page.page_number,
                            "context": context,
                        })

                return ToolResult(
                    success=True,
                    output={
                        "query": query,
                        "matches": len(results),
                        "results": results[:20],  # Limit results
                    },
                )

            else:
                return ToolResult(
                    success=False,
                    output=None,
                    error=f"Unknown action: {action}",
                )

        except FileNotFoundError as e:
            return ToolResult(
                success=False,
                output=None,
                error=str(e),
            )
        except ImportError as e:
            return ToolResult(
                success=False,
                output=None,
                error=str(e),
            )
        except Exception as e:
            logger.exception(f"PDF tool error: {e}")
            return ToolResult(
                success=False,
                output=None,
                error=f"Failed to process PDF: {e}",
            )


# Register the tool
pdf_tool = PDFTool()
register_tool(pdf_tool)
