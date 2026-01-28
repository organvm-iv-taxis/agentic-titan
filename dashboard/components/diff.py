"""
Diff Viewer - Code diff visualization component.

Provides:
- Side-by-side diff view
- Unified diff view
- Syntax highlighting support
- Line-level change tracking

Reference: vendor/cli/aionui/ code diff patterns
"""

from __future__ import annotations

import difflib
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger("titan.dashboard.diff")


# ============================================================================
# Data Structures
# ============================================================================


class ChangeType(str, Enum):
    """Type of change in a diff."""

    ADDED = "added"
    REMOVED = "removed"
    MODIFIED = "modified"
    UNCHANGED = "unchanged"


@dataclass
class LineDiff:
    """A single line in a diff."""

    line_number_old: int | None
    line_number_new: int | None
    content: str
    change_type: ChangeType

    def to_dict(self) -> dict[str, Any]:
        return {
            "line_old": self.line_number_old,
            "line_new": self.line_number_new,
            "content": self.content,
            "type": self.change_type.value,
        }


@dataclass
class Hunk:
    """A contiguous block of changes."""

    old_start: int
    old_count: int
    new_start: int
    new_count: int
    lines: list[LineDiff] = field(default_factory=list)
    header: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "old_start": self.old_start,
            "old_count": self.old_count,
            "new_start": self.new_start,
            "new_count": self.new_count,
            "header": self.header,
            "lines": [l.to_dict() for l in self.lines],
        }


@dataclass
class FileDiff:
    """Diff for a single file."""

    old_path: str
    new_path: str
    hunks: list[Hunk] = field(default_factory=list)

    # Statistics
    additions: int = 0
    deletions: int = 0

    # Content
    old_content: str = ""
    new_content: str = ""

    # Status
    is_new: bool = False
    is_deleted: bool = False
    is_renamed: bool = False
    is_binary: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "old_path": self.old_path,
            "new_path": self.new_path,
            "hunks": [h.to_dict() for h in self.hunks],
            "additions": self.additions,
            "deletions": self.deletions,
            "is_new": self.is_new,
            "is_deleted": self.is_deleted,
            "is_renamed": self.is_renamed,
            "is_binary": self.is_binary,
        }

    @property
    def change_summary(self) -> str:
        """Get a summary of changes."""
        parts = []
        if self.additions:
            parts.append(f"+{self.additions}")
        if self.deletions:
            parts.append(f"-{self.deletions}")
        return ", ".join(parts) or "No changes"


# ============================================================================
# Diff Viewer
# ============================================================================


class DiffViewer:
    """
    Code diff viewer component.

    Generates diffs in multiple formats for display.
    """

    def __init__(
        self,
        context_lines: int = 3,
        max_line_length: int = 120,
    ) -> None:
        self.context_lines = context_lines
        self.max_line_length = max_line_length

    def compute_diff(
        self,
        old_content: str,
        new_content: str,
        old_path: str = "old",
        new_path: str = "new",
    ) -> FileDiff:
        """
        Compute diff between two versions.

        Args:
            old_content: Original content
            new_content: New content
            old_path: Original file path
            new_path: New file path

        Returns:
            FileDiff with computed changes
        """
        file_diff = FileDiff(
            old_path=old_path,
            new_path=new_path,
            old_content=old_content,
            new_content=new_content,
        )

        # Handle special cases
        if not old_content and new_content:
            file_diff.is_new = True
        elif old_content and not new_content:
            file_diff.is_deleted = True
        elif old_path != new_path:
            file_diff.is_renamed = True

        # Split into lines
        old_lines = old_content.splitlines(keepends=True)
        new_lines = new_content.splitlines(keepends=True)

        # Generate unified diff
        diff_gen = difflib.unified_diff(
            old_lines,
            new_lines,
            fromfile=old_path,
            tofile=new_path,
            n=self.context_lines,
        )

        # Parse diff into hunks
        current_hunk: Hunk | None = None
        old_line = 0
        new_line = 0

        for line in diff_gen:
            # Skip header lines
            if line.startswith("---") or line.startswith("+++"):
                continue

            # Hunk header
            if line.startswith("@@"):
                if current_hunk:
                    file_diff.hunks.append(current_hunk)

                # Parse hunk header: @@ -start,count +start,count @@
                parts = line.split()
                old_info = parts[1][1:].split(",")
                new_info = parts[2][1:].split(",")

                old_start = int(old_info[0])
                old_count = int(old_info[1]) if len(old_info) > 1 else 1
                new_start = int(new_info[0])
                new_count = int(new_info[1]) if len(new_info) > 1 else 1

                current_hunk = Hunk(
                    old_start=old_start,
                    old_count=old_count,
                    new_start=new_start,
                    new_count=new_count,
                    header=line.strip(),
                )

                old_line = old_start
                new_line = new_start
                continue

            if not current_hunk:
                continue

            # Process line
            content = line[1:] if line else ""

            if line.startswith("+"):
                current_hunk.lines.append(LineDiff(
                    line_number_old=None,
                    line_number_new=new_line,
                    content=content.rstrip("\n"),
                    change_type=ChangeType.ADDED,
                ))
                file_diff.additions += 1
                new_line += 1

            elif line.startswith("-"):
                current_hunk.lines.append(LineDiff(
                    line_number_old=old_line,
                    line_number_new=None,
                    content=content.rstrip("\n"),
                    change_type=ChangeType.REMOVED,
                ))
                file_diff.deletions += 1
                old_line += 1

            else:
                current_hunk.lines.append(LineDiff(
                    line_number_old=old_line,
                    line_number_new=new_line,
                    content=content.rstrip("\n"),
                    change_type=ChangeType.UNCHANGED,
                ))
                old_line += 1
                new_line += 1

        # Add last hunk
        if current_hunk:
            file_diff.hunks.append(current_hunk)

        return file_diff

    def to_unified(self, file_diff: FileDiff) -> str:
        """
        Generate unified diff format.

        Args:
            file_diff: FileDiff to format

        Returns:
            Unified diff string
        """
        lines = []
        lines.append(f"--- {file_diff.old_path}")
        lines.append(f"+++ {file_diff.new_path}")

        for hunk in file_diff.hunks:
            lines.append(hunk.header)
            for line in hunk.lines:
                if line.change_type == ChangeType.ADDED:
                    lines.append(f"+{line.content}")
                elif line.change_type == ChangeType.REMOVED:
                    lines.append(f"-{line.content}")
                else:
                    lines.append(f" {line.content}")

        return "\n".join(lines)

    def to_side_by_side(self, file_diff: FileDiff) -> list[dict[str, Any]]:
        """
        Generate side-by-side diff data.

        Args:
            file_diff: FileDiff to format

        Returns:
            List of row dicts for side-by-side display
        """
        rows = []

        for hunk in file_diff.hunks:
            # Add hunk separator
            rows.append({
                "type": "hunk_header",
                "header": hunk.header,
            })

            for line in hunk.lines:
                if line.change_type == ChangeType.UNCHANGED:
                    rows.append({
                        "type": "unchanged",
                        "left_num": line.line_number_old,
                        "left_content": line.content,
                        "right_num": line.line_number_new,
                        "right_content": line.content,
                    })
                elif line.change_type == ChangeType.REMOVED:
                    rows.append({
                        "type": "removed",
                        "left_num": line.line_number_old,
                        "left_content": line.content,
                        "right_num": None,
                        "right_content": "",
                    })
                elif line.change_type == ChangeType.ADDED:
                    rows.append({
                        "type": "added",
                        "left_num": None,
                        "left_content": "",
                        "right_num": line.line_number_new,
                        "right_content": line.content,
                    })

        return rows

    def to_html(
        self,
        file_diff: FileDiff,
        style: str = "unified",
    ) -> str:
        """
        Generate HTML diff.

        Args:
            file_diff: FileDiff to format
            style: "unified" or "side-by-side"

        Returns:
            HTML string
        """
        if style == "side-by-side":
            return self._to_html_side_by_side(file_diff)
        return self._to_html_unified(file_diff)

    def _to_html_unified(self, file_diff: FileDiff) -> str:
        """Generate unified HTML diff."""
        html = ['<div class="diff-viewer unified">']
        html.append(f'<div class="diff-header">')
        html.append(f'<span class="diff-old-path">{file_diff.old_path}</span>')
        html.append(f' → ')
        html.append(f'<span class="diff-new-path">{file_diff.new_path}</span>')
        html.append(f'<span class="diff-stats">+{file_diff.additions} -{file_diff.deletions}</span>')
        html.append('</div>')

        html.append('<table class="diff-table">')

        for hunk in file_diff.hunks:
            html.append(f'<tr class="hunk-header"><td colspan="3">{_escape_html(hunk.header)}</td></tr>')

            for line in hunk.lines:
                css_class = line.change_type.value
                prefix = ""
                if line.change_type == ChangeType.ADDED:
                    prefix = "+"
                elif line.change_type == ChangeType.REMOVED:
                    prefix = "-"
                else:
                    prefix = " "

                line_num = line.line_number_new or line.line_number_old or ""
                content = _escape_html(line.content)

                html.append(f'<tr class="diff-line {css_class}">')
                html.append(f'<td class="line-num">{line_num}</td>')
                html.append(f'<td class="line-prefix">{prefix}</td>')
                html.append(f'<td class="line-content">{content}</td>')
                html.append('</tr>')

        html.append('</table>')
        html.append('</div>')

        return "\n".join(html)

    def _to_html_side_by_side(self, file_diff: FileDiff) -> str:
        """Generate side-by-side HTML diff."""
        html = ['<div class="diff-viewer side-by-side">']
        html.append(f'<div class="diff-header">')
        html.append(f'<span class="diff-stats">+{file_diff.additions} -{file_diff.deletions}</span>')
        html.append('</div>')

        html.append('<table class="diff-table">')
        html.append('<colgroup>')
        html.append('<col class="line-num-col">')
        html.append('<col class="content-col">')
        html.append('<col class="line-num-col">')
        html.append('<col class="content-col">')
        html.append('</colgroup>')

        html.append('<thead>')
        html.append(f'<tr><th colspan="2">{_escape_html(file_diff.old_path)}</th>')
        html.append(f'<th colspan="2">{_escape_html(file_diff.new_path)}</th></tr>')
        html.append('</thead>')

        html.append('<tbody>')

        for row in self.to_side_by_side(file_diff):
            if row["type"] == "hunk_header":
                html.append(f'<tr class="hunk-header"><td colspan="4">{_escape_html(row["header"])}</td></tr>')
            else:
                css_class = row["type"]
                left_num = row["left_num"] or ""
                left_content = _escape_html(row["left_content"])
                right_num = row["right_num"] or ""
                right_content = _escape_html(row["right_content"])

                html.append(f'<tr class="diff-line {css_class}">')
                html.append(f'<td class="line-num">{left_num}</td>')
                html.append(f'<td class="line-content left">{left_content}</td>')
                html.append(f'<td class="line-num">{right_num}</td>')
                html.append(f'<td class="line-content right">{right_content}</td>')
                html.append('</tr>')

        html.append('</tbody>')
        html.append('</table>')
        html.append('</div>')

        return "\n".join(html)


def _escape_html(text: str) -> str:
    """Escape HTML special characters."""
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#39;")
    )


# ============================================================================
# Convenience Functions
# ============================================================================


def create_diff(
    old_content: str,
    new_content: str,
    old_path: str = "old",
    new_path: str = "new",
) -> FileDiff:
    """
    Create a diff between two content strings.

    Args:
        old_content: Original content
        new_content: New content
        old_path: Original file path
        new_path: New file path

    Returns:
        FileDiff object
    """
    viewer = DiffViewer()
    return viewer.compute_diff(old_content, new_content, old_path, new_path)


def diff_to_html(
    old_content: str,
    new_content: str,
    old_path: str = "old",
    new_path: str = "new",
    style: str = "unified",
) -> str:
    """
    Generate HTML diff from two content strings.

    Args:
        old_content: Original content
        new_content: New content
        old_path: Original file path
        new_path: New file path
        style: "unified" or "side-by-side"

    Returns:
        HTML string
    """
    viewer = DiffViewer()
    file_diff = viewer.compute_diff(old_content, new_content, old_path, new_path)
    return viewer.to_html(file_diff, style)
