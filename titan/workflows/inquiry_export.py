"""
Titan Workflows - Inquiry Export

Utilities for exporting inquiry sessions to markdown files with YAML frontmatter.
Generates well-formatted documents suitable for documentation, sharing, and archival.

The export format matches the style from expand_AI_inquiry.jsx.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from titan.workflows.inquiry_engine import InquirySession


def slugify(text: str) -> str:
    """Convert text to URL-friendly slug."""
    # Convert to lowercase
    text = text.lower()
    # Replace non-alphanumeric with hyphens
    text = re.sub(r"[^a-z0-9]+", "-", text)
    # Remove leading/trailing hyphens
    text = text.strip("-")
    return text


def export_stage_to_markdown(
    session: InquirySession,
    stage_index: int,
) -> str:
    """
    Generate markdown with frontmatter for a single stage.

    Args:
        session: The inquiry session
        stage_index: Index of the stage to export

    Returns:
        Markdown string with YAML frontmatter
    """
    if stage_index >= len(session.results):
        raise ValueError(f"Stage index {stage_index} not yet completed")

    result = session.results[stage_index]
    stage = session.workflow.stages[stage_index]

    timestamp = result.timestamp.isoformat()
    date_formatted = result.timestamp.strftime("%B %d, %Y")

    topic_slug = slugify(session.topic)
    stage_slug = slugify(stage.name)

    frontmatter = f"""---
title: "{stage.name} - {session.topic}"
description: "{stage.description}"
topic: "{session.topic}"
stage: "{stage.name}"
ai_role: "{stage.role}"
stage_number: {stage_index + 1}
total_stages: {session.total_stages}
inquiry_type: "expansive_collaborative"
session_id: "{session.id}"
generated_date: "{timestamp}"
model_used: "{result.model_used}"
tokens_used: {result.tokens_used}
duration_ms: {result.duration_ms}
tags:
  - expansive-inquiry
  - ai-collaboration
  - {stage_slug}
  - cognitive-exploration
  - {topic_slug}
metadata:
  methodology: "Multi-AI Collaborative Inquiry"
  approach: "{stage.role}"
  cognitive_style: "{stage.cognitive_style.value}"
  complexity: "deep"
  domain: "cross-disciplinary"
---

"""

    body = f"""# {stage.name}: {session.topic}

{stage.emoji} **AI Role:** {stage.role}
**Generated:** {date_formatted}
**Stage:** {stage_index + 1} of {session.total_stages}
**Model:** {result.model_used}

## Overview

{stage.description}

## Inquiry Results

{result.content}

---

*This document was generated as part of an Expansive Inquiry AI Collaboration System.
Each stage builds upon previous insights to create a comprehensive exploration of the topic.*

**Execution Details:**
- Duration: {result.duration_ms}ms
- Tokens: ~{result.tokens_used}
- Session: `{session.id}`

**Next Steps:**
- Review findings from previous stages
- Identify patterns and connections
- Prepare for subsequent inquiry phases
"""

    return frontmatter + body


def export_session_to_markdown(
    session: InquirySession,
    include_toc: bool = True,
) -> str:
    """
    Generate combined markdown for entire inquiry session.

    Args:
        session: The completed inquiry session
        include_toc: Whether to include a table of contents

    Returns:
        Markdown string with all stages
    """
    timestamp = session.created_at.isoformat()
    date_formatted = session.created_at.strftime("%B %d, %Y")
    if session.completed_at:
        completed_at = session.completed_at.strftime("%B %d, %Y %H:%M")
    else:
        completed_at = "In Progress"
    topic_slug = slugify(session.topic)

    # Calculate totals
    total_tokens = sum(r.tokens_used for r in session.results)
    total_duration_ms = sum(r.duration_ms for r in session.results)

    # Build tags from all stages
    stage_tags = [slugify(s.name) for s in session.workflow.stages]
    core_tags = ["expansive-inquiry", "ai-collaboration", topic_slug]
    tags_yaml = "\n".join(f"  - {tag}" for tag in core_tags + stage_tags)

    frontmatter = f"""---
title: "Collaborative Inquiry: {session.topic}"
description: "{session.workflow.description}"
topic: "{session.topic}"
workflow: "{session.workflow.name}"
session_id: "{session.id}"
status: "{session.status.value}"
stages_completed: {len(session.results)}
total_stages: {session.total_stages}
inquiry_type: "expansive_collaborative"
created_date: "{timestamp}"
completed_date: "{session.completed_at.isoformat() if session.completed_at else ""}"
total_tokens: {total_tokens}
total_duration_ms: {total_duration_ms}
tags:
{tags_yaml}
metadata:
  methodology: "Multi-AI Collaborative Inquiry"
  complexity: "deep"
  domain: "cross-disciplinary"
---

"""

    # Build table of contents if requested
    toc = ""
    if include_toc and session.results:
        toc = "## Table of Contents\n\n"
        for i, result in enumerate(session.results):
            stage = session.workflow.stages[i]
            anchor = slugify(f"{stage.name}-{stage.role}")
            toc += f"{i + 1}. [{stage.name}](#{anchor}) - {stage.role}\n"
        toc += "\n---\n\n"

    # Header
    header = f"""# Collaborative Inquiry: {session.topic}

**Workflow:** {session.workflow.name}
**Created:** {date_formatted}
**Completed:** {completed_at}
**Status:** {session.status.value.title()}

## Overview

{session.workflow.description}

This inquiry explored **"{session.topic}"** through {session.total_stages} distinct
cognitive perspectives, each building upon previous insights to create a comprehensive
understanding.

---

"""

    # Build body with all stages
    body = ""
    for i, result in enumerate(session.results):
        stage = session.workflow.stages[i]
        anchor = slugify(f"{stage.name}-{stage.role}")

        body += f"""
## {stage.emoji} {stage.name}: {stage.role} {{#{anchor}}}

**Cognitive Style:** {stage.cognitive_style.value.replace("_", " ").title()}
**Model:** {result.model_used}
**Duration:** {result.duration_ms}ms

### Stage Overview

{stage.description}

### Results

{result.content}

---

"""

    # Summary section if complete
    summary = ""
    if session.is_complete:
        summary = f"""
## Collaborative Inquiry Complete

The system has completed all {session.total_stages} stages of expansive inquiry.
Each AI specialist contributed their unique perspective:

"""
        for i, result in enumerate(session.results):
            stage = session.workflow.stages[i]
            summary += f"- **{stage.role}**: {stage.description}\n"

        summary += f"""
Notice how the different AI roles built upon each other's insights to reveal
dimensions that no single AI could have discovered alone.

### Execution Summary

| Metric | Value |
|--------|-------|
| Total Stages | {len(session.results)} |
| Total Tokens | ~{total_tokens:,} |
| Total Duration | {total_duration_ms:,}ms ({total_duration_ms / 1000:.1f}s) |
| Session ID | `{session.id}` |

---

*Generated by the Titan Expansive Inquiry System*
"""

    return frontmatter + header + toc + body + summary


def export_session_to_files(
    session: InquirySession,
    output_dir: str,
) -> list[str]:
    """
    Export session to multiple markdown files (one per stage + summary).

    Args:
        session: The inquiry session
        output_dir: Directory to write files to

    Returns:
        List of file paths created
    """
    import os

    os.makedirs(output_dir, exist_ok=True)
    files_created = []

    topic_slug = slugify(session.topic)[:30]

    # Export individual stages
    for i in range(len(session.results)):
        stage = session.workflow.stages[i]
        stage_slug = slugify(stage.name)
        filename = f"{i + 1:02d}-{stage_slug}-{topic_slug}.md"
        filepath = os.path.join(output_dir, filename)

        content = export_stage_to_markdown(session, i)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)

        files_created.append(filepath)

    # Export combined document
    summary_filename = f"00-summary-{topic_slug}.md"
    summary_filepath = os.path.join(output_dir, summary_filename)

    summary_content = export_session_to_markdown(session)
    with open(summary_filepath, "w", encoding="utf-8") as f:
        f.write(summary_content)

    files_created.insert(0, summary_filepath)

    return files_created
