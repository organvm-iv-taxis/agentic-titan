"""
File Browser - Workspace file navigation component.

Provides:
- Directory tree view
- File content preview
- Search functionality
- Git status integration

Reference: vendor/cli/aionui/ file tree patterns
"""

from __future__ import annotations

import logging
import os
import stat
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger("titan.dashboard.filebrowser")


# ============================================================================
# Data Structures
# ============================================================================


@dataclass
class FileEntry:
    """A file or directory entry."""

    name: str
    path: str
    is_dir: bool
    size: int = 0
    modified: datetime | None = None
    permissions: str = ""

    # Directory-specific
    children: list[FileEntry] | None = None

    # File-specific
    extension: str = ""
    mime_type: str = ""

    # Git status
    git_status: str = ""  # "M", "A", "D", "?", etc.

    # UI state
    expanded: bool = False
    selected: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "path": self.path,
            "is_dir": self.is_dir,
            "size": self.size,
            "modified": self.modified.isoformat() if self.modified else None,
            "permissions": self.permissions,
            "extension": self.extension,
            "git_status": self.git_status,
            "expanded": self.expanded,
            "selected": self.selected,
            "children": [c.to_dict() for c in self.children] if self.children else None,
        }

    @property
    def icon(self) -> str:
        """Get icon for file type."""
        if self.is_dir:
            return "folder"

        # By extension
        icons = {
            ".py": "python",
            ".js": "javascript",
            ".ts": "typescript",
            ".jsx": "react",
            ".tsx": "react",
            ".html": "html",
            ".css": "css",
            ".json": "json",
            ".yaml": "yaml",
            ".yml": "yaml",
            ".md": "markdown",
            ".txt": "text",
            ".sh": "shell",
            ".sql": "database",
            ".png": "image",
            ".jpg": "image",
            ".svg": "image",
            ".pdf": "pdf",
        }
        return icons.get(self.extension.lower(), "file")

    @property
    def size_formatted(self) -> str:
        """Get human-readable size."""
        if self.is_dir:
            return ""

        units = ["B", "KB", "MB", "GB"]
        size = float(self.size)
        unit_idx = 0

        while size >= 1024 and unit_idx < len(units) - 1:
            size /= 1024
            unit_idx += 1

        if unit_idx == 0:
            return f"{int(size)} {units[unit_idx]}"
        return f"{size:.1f} {units[unit_idx]}"


# ============================================================================
# File Browser
# ============================================================================


class FileBrowser:
    """
    File browser component for workspace navigation.

    Provides:
    - Directory listing
    - File search
    - Git status integration
    - Content preview
    """

    def __init__(
        self,
        root_path: str | Path,
        max_depth: int = 10,
        ignore_patterns: list[str] | None = None,
    ) -> None:
        self.root_path = Path(root_path).resolve()
        self.max_depth = max_depth
        self.ignore_patterns = ignore_patterns or [
            ".git",
            "__pycache__",
            "node_modules",
            ".venv",
            "venv",
            ".env",
            ".DS_Store",
            "*.pyc",
            "*.pyo",
            ".idea",
            ".vscode",
        ]

        # Cache
        self._tree_cache: FileEntry | None = None
        self._git_status: dict[str, str] = {}

    def get_tree(
        self,
        path: str | Path | None = None,
        depth: int = 2,
        include_hidden: bool = False,
    ) -> FileEntry:
        """
        Get directory tree.

        Args:
            path: Starting path (default: root)
            depth: Maximum depth to traverse
            include_hidden: Include hidden files

        Returns:
            FileEntry tree
        """
        path = Path(path) if path else self.root_path
        path = path.resolve()

        # Validate path is under root
        try:
            path.relative_to(self.root_path)
        except ValueError:
            raise ValueError(f"Path {path} is outside root {self.root_path}")

        return self._build_tree(path, depth, include_hidden)

    def _build_tree(
        self,
        path: Path,
        depth: int,
        include_hidden: bool,
        current_depth: int = 0,
    ) -> FileEntry:
        """Build directory tree recursively."""
        entry = self._create_entry(path)

        if not entry.is_dir or current_depth >= depth:
            return entry

        # List directory contents
        children = []
        try:
            for child_path in sorted(path.iterdir()):
                # Skip ignored patterns
                if self._should_ignore(child_path, include_hidden):
                    continue

                child_entry = self._build_tree(
                    child_path,
                    depth,
                    include_hidden,
                    current_depth + 1,
                )
                children.append(child_entry)

        except PermissionError:
            pass

        # Sort: directories first, then alphabetically
        children.sort(key=lambda e: (not e.is_dir, e.name.lower()))
        entry.children = children

        return entry

    def _create_entry(self, path: Path) -> FileEntry:
        """Create a FileEntry from a path."""
        try:
            stat_info = path.stat()
            is_dir = path.is_dir()

            entry = FileEntry(
                name=path.name or str(path),
                path=str(path.relative_to(self.root_path)),
                is_dir=is_dir,
                size=stat_info.st_size if not is_dir else 0,
                modified=datetime.fromtimestamp(stat_info.st_mtime),
                permissions=stat.filemode(stat_info.st_mode),
                extension=path.suffix if not is_dir else "",
            )

            # Add git status
            rel_path = str(path.relative_to(self.root_path))
            if rel_path in self._git_status:
                entry.git_status = self._git_status[rel_path]

            return entry

        except Exception as e:
            logger.warning(f"Error reading {path}: {e}")
            return FileEntry(
                name=path.name,
                path=str(path),
                is_dir=False,
            )

    def _should_ignore(self, path: Path, include_hidden: bool) -> bool:
        """Check if path should be ignored."""
        name = path.name

        # Hidden files
        if not include_hidden and name.startswith("."):
            return True

        # Ignore patterns
        for pattern in self.ignore_patterns:
            if pattern.startswith("*"):
                # Extension pattern
                if name.endswith(pattern[1:]):
                    return True
            elif name == pattern:
                return True

        return False

    def get_entry(self, path: str | Path) -> FileEntry | None:
        """Get a specific file entry."""
        full_path = self.root_path / path
        if full_path.exists():
            return self._create_entry(full_path)
        return None

    def list_dir(
        self,
        path: str | Path | None = None,
        include_hidden: bool = False,
    ) -> list[FileEntry]:
        """
        List directory contents (flat).

        Args:
            path: Directory path (default: root)
            include_hidden: Include hidden files

        Returns:
            List of FileEntry
        """
        dir_path = self.root_path / path if path else self.root_path

        if not dir_path.is_dir():
            return []

        entries = []
        try:
            for child_path in sorted(dir_path.iterdir()):
                if self._should_ignore(child_path, include_hidden):
                    continue
                entries.append(self._create_entry(child_path))

            # Sort: directories first
            entries.sort(key=lambda e: (not e.is_dir, e.name.lower()))

        except PermissionError:
            pass

        return entries

    def read_file(
        self,
        path: str | Path,
        max_size: int = 1_000_000,  # 1MB
    ) -> str | None:
        """
        Read file content.

        Args:
            path: File path
            max_size: Maximum size to read

        Returns:
            File content or None
        """
        full_path = self.root_path / path
        full_path = full_path.resolve()

        # Validate
        try:
            full_path.relative_to(self.root_path)
        except ValueError:
            logger.warning(f"Access denied: {path} is outside root")
            return None

        if not full_path.is_file():
            return None

        # Check size
        size = full_path.stat().st_size
        if size > max_size:
            logger.warning(f"File too large: {size} bytes")
            return None

        try:
            return full_path.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            logger.warning(f"Error reading {path}: {e}")
            return None

    def search(
        self,
        query: str,
        path: str | Path | None = None,
        max_results: int = 50,
    ) -> list[FileEntry]:
        """
        Search for files by name.

        Args:
            query: Search query
            path: Starting path (default: root)
            max_results: Maximum results

        Returns:
            Matching FileEntry list
        """
        start_path = self.root_path / path if path else self.root_path
        query_lower = query.lower()

        results = []

        def search_recursive(dir_path: Path, depth: int = 0) -> None:
            if len(results) >= max_results or depth > self.max_depth:
                return

            try:
                for child in dir_path.iterdir():
                    if len(results) >= max_results:
                        return

                    if self._should_ignore(child, include_hidden=False):
                        continue

                    # Check name match
                    if query_lower in child.name.lower():
                        results.append(self._create_entry(child))

                    # Recurse into directories
                    if child.is_dir():
                        search_recursive(child, depth + 1)

            except PermissionError:
                pass

        search_recursive(start_path)

        # Sort by name
        results.sort(key=lambda e: e.name.lower())
        return results

    def refresh_git_status(self) -> None:
        """Refresh git status for the workspace."""
        self._git_status.clear()

        try:
            import subprocess

            result = subprocess.run(
                ["git", "status", "--porcelain", "-uall"],
                cwd=str(self.root_path),
                capture_output=True,
                text=True,
            )

            if result.returncode == 0:
                for line in result.stdout.strip().split("\n"):
                    if line:
                        status = line[:2].strip()
                        file_path = line[3:]
                        self._git_status[file_path] = status

        except Exception as e:
            logger.debug(f"Git status unavailable: {e}")

    def get_breadcrumbs(self, path: str | Path) -> list[dict[str, str]]:
        """
        Get breadcrumb navigation for a path.

        Args:
            path: File path

        Returns:
            List of breadcrumb items
        """
        breadcrumbs = [{"name": self.root_path.name, "path": ""}]

        if path:
            parts = Path(path).parts
            current = ""
            for part in parts:
                current = str(Path(current) / part)
                breadcrumbs.append({"name": part, "path": current})

        return breadcrumbs


# ============================================================================
# Convenience Functions
# ============================================================================


def create_file_browser(
    root_path: str | Path = ".",
    ignore_patterns: list[str] | None = None,
) -> FileBrowser:
    """
    Create a file browser for a directory.

    Args:
        root_path: Root directory
        ignore_patterns: Patterns to ignore

    Returns:
        FileBrowser instance
    """
    return FileBrowser(root_path, ignore_patterns=ignore_patterns)


def get_file_tree(
    root_path: str | Path = ".",
    depth: int = 2,
) -> dict[str, Any]:
    """
    Get file tree as dict.

    Args:
        root_path: Root directory
        depth: Maximum depth

    Returns:
        Tree dict
    """
    browser = FileBrowser(root_path)
    tree = browser.get_tree(depth=depth)
    return tree.to_dict()
