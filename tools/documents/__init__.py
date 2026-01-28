"""
Document Tools - Office document manipulation.

Provides tools for working with:
- DOCX (Microsoft Word documents)
- XLSX (Microsoft Excel spreadsheets)
- PPTX (Microsoft PowerPoint presentations)

Reference: vendor/agents/skills/ document manipulation patterns
"""

from tools.documents.docx import DOCXTool, docx_tool
from tools.documents.xlsx import XLSXTool, xlsx_tool
from tools.documents.pptx import PPTXTool, pptx_tool

__all__ = [
    "DOCXTool",
    "docx_tool",
    "XLSXTool",
    "xlsx_tool",
    "PPTXTool",
    "pptx_tool",
]
