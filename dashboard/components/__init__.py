"""
Dashboard Components - Modular UI components for the Titan dashboard.

Provides:
- ConversationManager: Multi-conversation support
- DiffViewer: Code diff visualization
- FileBrowser: Workspace file navigation
"""

from dashboard.components.conversations import ConversationManager, Conversation
from dashboard.components.diff import DiffViewer, FileDiff
from dashboard.components.filebrowser import FileBrowser, FileEntry

__all__ = [
    "ConversationManager",
    "Conversation",
    "DiffViewer",
    "FileDiff",
    "FileBrowser",
    "FileEntry",
]
