"""
Conversation Manager - Multi-conversation support for the dashboard.

Enables:
- Multiple concurrent conversations
- Session persistence
- Conversation history
- Tab-based switching

Reference: vendor/cli/aionui/ multi-conversation patterns
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

logger = logging.getLogger("titan.dashboard.conversations")


# ============================================================================
# Data Structures
# ============================================================================


@dataclass
class Message:
    """A single message in a conversation."""

    id: str
    role: str  # "user", "assistant", "system"
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)

    # Tool calls
    tool_calls: list[dict[str, Any]] | None = None
    tool_results: list[dict[str, Any]] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
            "tool_calls": self.tool_calls,
            "tool_results": self.tool_results,
        }


@dataclass
class Conversation:
    """
    A conversation session with an agent.

    Contains message history and metadata.
    """

    id: str
    title: str = "New Conversation"
    agent_id: str | None = None
    agent_name: str = ""

    # Messages
    messages: list[Message] = field(default_factory=list)

    # State
    active: bool = True
    pinned: bool = False

    # Timestamps
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def message_count(self) -> int:
        return len(self.messages)

    @property
    def preview(self) -> str:
        """Get preview text from last message."""
        if self.messages:
            last = self.messages[-1]
            content = last.content[:100]
            if len(last.content) > 100:
                content += "..."
            return content
        return "No messages"

    def add_message(
        self,
        role: str,
        content: str,
        metadata: dict[str, Any] | None = None,
        tool_calls: list[dict[str, Any]] | None = None,
    ) -> Message:
        """Add a message to the conversation."""
        message = Message(
            id=f"msg-{uuid.uuid4().hex[:8]}",
            role=role,
            content=content,
            metadata=metadata or {},
            tool_calls=tool_calls,
        )
        self.messages.append(message)
        self.updated_at = datetime.now()

        # Auto-title from first user message
        if len(self.messages) == 1 and role == "user":
            self.title = content[:50]
            if len(content) > 50:
                self.title += "..."

        return message

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "agent_id": self.agent_id,
            "agent_name": self.agent_name,
            "message_count": self.message_count,
            "active": self.active,
            "pinned": self.pinned,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "preview": self.preview,
            "metadata": self.metadata,
        }

    def to_dict_full(self) -> dict[str, Any]:
        """Full dict including messages."""
        data = self.to_dict()
        data["messages"] = [m.to_dict() for m in self.messages]
        return data


# ============================================================================
# Conversation Manager
# ============================================================================


class ConversationManager:
    """
    Manages multiple concurrent conversations.

    Provides:
    - Create/switch/delete conversations
    - Persist conversation history
    - Search across conversations
    """

    def __init__(self, max_conversations: int = 50) -> None:
        self.max_conversations = max_conversations
        self._conversations: dict[str, Conversation] = {}
        self._active_id: str | None = None

    def create(
        self,
        title: str = "New Conversation",
        agent_id: str | None = None,
        agent_name: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> Conversation:
        """
        Create a new conversation.

        Args:
            title: Conversation title
            agent_id: Associated agent ID
            agent_name: Associated agent name
            metadata: Additional metadata

        Returns:
            New Conversation
        """
        conv_id = f"conv-{uuid.uuid4().hex[:8]}"

        conversation = Conversation(
            id=conv_id,
            title=title,
            agent_id=agent_id,
            agent_name=agent_name,
            metadata=metadata or {},
        )

        self._conversations[conv_id] = conversation
        self._active_id = conv_id

        # Enforce max conversations
        self._enforce_limit()

        logger.info(f"Created conversation: {conv_id}")
        return conversation

    def get(self, conversation_id: str) -> Conversation | None:
        """Get a conversation by ID."""
        return self._conversations.get(conversation_id)

    def get_active(self) -> Conversation | None:
        """Get the currently active conversation."""
        if self._active_id:
            return self._conversations.get(self._active_id)
        return None

    def set_active(self, conversation_id: str) -> bool:
        """
        Set a conversation as active.

        Args:
            conversation_id: Conversation ID

        Returns:
            True if successful
        """
        if conversation_id in self._conversations:
            self._active_id = conversation_id
            return True
        return False

    def delete(self, conversation_id: str) -> bool:
        """
        Delete a conversation.

        Args:
            conversation_id: Conversation ID

        Returns:
            True if deleted
        """
        if conversation_id in self._conversations:
            conv = self._conversations[conversation_id]

            # Don't delete pinned conversations
            if conv.pinned:
                logger.warning(f"Cannot delete pinned conversation: {conversation_id}")
                return False

            del self._conversations[conversation_id]

            # Update active if needed
            if self._active_id == conversation_id:
                self._active_id = list(self._conversations.keys())[0] if self._conversations else None

            logger.info(f"Deleted conversation: {conversation_id}")
            return True
        return False

    def list(
        self,
        include_inactive: bool = True,
        sort_by: str = "updated_at",
    ) -> list[Conversation]:
        """
        List all conversations.

        Args:
            include_inactive: Include inactive conversations
            sort_by: Sort field ("updated_at", "created_at", "title")

        Returns:
            List of conversations
        """
        conversations = list(self._conversations.values())

        if not include_inactive:
            conversations = [c for c in conversations if c.active]

        # Sort
        if sort_by == "updated_at":
            conversations.sort(key=lambda c: c.updated_at, reverse=True)
        elif sort_by == "created_at":
            conversations.sort(key=lambda c: c.created_at, reverse=True)
        elif sort_by == "title":
            conversations.sort(key=lambda c: c.title.lower())

        # Put pinned first
        pinned = [c for c in conversations if c.pinned]
        unpinned = [c for c in conversations if not c.pinned]

        return pinned + unpinned

    def search(self, query: str, max_results: int = 10) -> list[Conversation]:
        """
        Search conversations by content.

        Args:
            query: Search query
            max_results: Maximum results

        Returns:
            Matching conversations
        """
        query_lower = query.lower()
        results = []

        for conv in self._conversations.values():
            # Check title
            if query_lower in conv.title.lower():
                results.append((conv, 2.0))  # Higher score for title match
                continue

            # Check messages
            for msg in conv.messages:
                if query_lower in msg.content.lower():
                    results.append((conv, 1.0))
                    break

        # Sort by score
        results.sort(key=lambda x: x[1], reverse=True)

        return [r[0] for r in results[:max_results]]

    def pin(self, conversation_id: str) -> bool:
        """Pin a conversation."""
        conv = self.get(conversation_id)
        if conv:
            conv.pinned = True
            return True
        return False

    def unpin(self, conversation_id: str) -> bool:
        """Unpin a conversation."""
        conv = self.get(conversation_id)
        if conv:
            conv.pinned = False
            return True
        return False

    def add_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        **kwargs: Any,
    ) -> Message | None:
        """Add a message to a conversation."""
        conv = self.get(conversation_id)
        if conv:
            return conv.add_message(role, content, **kwargs)
        return None

    def get_messages(
        self,
        conversation_id: str,
        limit: int | None = None,
    ) -> list[Message]:
        """Get messages from a conversation."""
        conv = self.get(conversation_id)
        if conv:
            messages = conv.messages
            if limit:
                messages = messages[-limit:]
            return messages
        return []

    def clear_messages(self, conversation_id: str) -> bool:
        """Clear messages from a conversation."""
        conv = self.get(conversation_id)
        if conv:
            conv.messages.clear()
            conv.updated_at = datetime.now()
            return True
        return False

    def export(self, conversation_id: str) -> dict[str, Any] | None:
        """Export a conversation to dict."""
        conv = self.get(conversation_id)
        if conv:
            return conv.to_dict_full()
        return None

    def export_all(self) -> list[dict[str, Any]]:
        """Export all conversations."""
        return [c.to_dict_full() for c in self._conversations.values()]

    def import_conversation(self, data: dict[str, Any]) -> Conversation | None:
        """Import a conversation from dict."""
        try:
            conv_id = data.get("id") or f"conv-{uuid.uuid4().hex[:8]}"

            conv = Conversation(
                id=conv_id,
                title=data.get("title", "Imported"),
                agent_id=data.get("agent_id"),
                agent_name=data.get("agent_name", ""),
                active=data.get("active", True),
                pinned=data.get("pinned", False),
                metadata=data.get("metadata", {}),
            )

            # Import messages
            for msg_data in data.get("messages", []):
                conv.messages.append(Message(
                    id=msg_data.get("id", f"msg-{uuid.uuid4().hex[:8]}"),
                    role=msg_data["role"],
                    content=msg_data["content"],
                    timestamp=datetime.fromisoformat(msg_data["timestamp"]) if msg_data.get("timestamp") else datetime.now(),
                    metadata=msg_data.get("metadata", {}),
                    tool_calls=msg_data.get("tool_calls"),
                    tool_results=msg_data.get("tool_results"),
                ))

            self._conversations[conv.id] = conv
            return conv

        except Exception as e:
            logger.error(f"Failed to import conversation: {e}")
            return None

    def _enforce_limit(self) -> None:
        """Remove old conversations if over limit."""
        if len(self._conversations) <= self.max_conversations:
            return

        # Get non-pinned conversations sorted by update time
        unpinned = sorted(
            [c for c in self._conversations.values() if not c.pinned],
            key=lambda c: c.updated_at,
        )

        # Remove oldest until under limit
        while len(self._conversations) > self.max_conversations and unpinned:
            oldest = unpinned.pop(0)
            del self._conversations[oldest.id]
            logger.info(f"Removed old conversation: {oldest.id}")

    @property
    def count(self) -> int:
        return len(self._conversations)

    @property
    def active_id(self) -> str | None:
        return self._active_id
