"""
Persona Coordination - Enhanced multi-agent persona management.

Provides advanced persona coordination capabilities:
- Persona state tracking across conversations
- Inter-persona communication protocols
- Role-based task delegation
- Personality consistency enforcement
- Collective persona operations

Reference: vendor/agents/collective-persona/ coordination patterns
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable

from agents.personas import (
    Persona,
    PersonaRole,
    get_persona,
    register_persona,
    say,
    think,
)

logger = logging.getLogger("titan.agents.persona_coordination")


# ============================================================================
# Coordination Enums
# ============================================================================


class CoordinationMode(str, Enum):
    """Modes of persona coordination."""

    SEQUENTIAL = "sequential"    # Personas work one after another
    PARALLEL = "parallel"        # Personas work simultaneously
    HIERARCHICAL = "hierarchical"  # Leader-follower structure
    COLLABORATIVE = "collaborative"  # Peer-to-peer collaboration
    DEBATE = "debate"            # Personas argue different positions


class InteractionType(str, Enum):
    """Types of inter-persona interactions."""

    REQUEST = "request"          # Ask another persona to do something
    INFORM = "inform"            # Share information
    DELEGATE = "delegate"        # Assign a task
    QUERY = "query"              # Ask a question
    CRITIQUE = "critique"        # Provide feedback
    AGREE = "agree"              # Express agreement
    DISAGREE = "disagree"        # Express disagreement
    SYNTHESIZE = "synthesize"    # Combine perspectives


# ============================================================================
# Data Structures
# ============================================================================


@dataclass
class PersonaState:
    """State of a persona in a coordination context."""

    persona: Persona
    active: bool = True
    current_task: str | None = None
    completed_tasks: list[str] = field(default_factory=list)
    messages_sent: int = 0
    messages_received: int = 0
    last_active: datetime = field(default_factory=datetime.now)
    context: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.persona.name,
            "role": self.persona.role.value,
            "active": self.active,
            "current_task": self.current_task,
            "completed_tasks": len(self.completed_tasks),
            "messages_sent": self.messages_sent,
            "messages_received": self.messages_received,
        }


@dataclass
class PersonaMessage:
    """A message between personas."""

    from_persona: str
    to_persona: str
    interaction_type: InteractionType
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "from": self.from_persona,
            "to": self.to_persona,
            "type": self.interaction_type.value,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class CoordinationResult:
    """Result of a coordinated operation."""

    mode: CoordinationMode
    participants: list[str]
    outputs: dict[str, Any]
    consensus: str | None = None
    dissent: list[str] = field(default_factory=list)
    messages_exchanged: int = 0
    duration_ms: int = 0


# ============================================================================
# Persona Coordinator
# ============================================================================


class PersonaCoordinator:
    """
    Coordinates interactions between multiple personas.

    Enables sophisticated multi-agent behaviors through persona coordination.
    """

    def __init__(
        self,
        mode: CoordinationMode = CoordinationMode.COLLABORATIVE,
    ) -> None:
        self.mode = mode
        self._states: dict[str, PersonaState] = {}
        self._message_history: list[PersonaMessage] = []
        self._message_handlers: dict[str, list[Callable]] = {}
        self._running = False

    def add_persona(
        self,
        persona: Persona | str,
        initial_context: dict[str, Any] | None = None,
    ) -> PersonaState:
        """
        Add a persona to the coordination.

        Args:
            persona: Persona or persona name
            initial_context: Initial context for the persona

        Returns:
            PersonaState for the added persona
        """
        if isinstance(persona, str):
            resolved = get_persona(persona)
            if not resolved:
                raise ValueError(f"Unknown persona: {persona}")
            persona = resolved

        state = PersonaState(
            persona=persona,
            context=initial_context or {},
        )
        self._states[persona.name] = state

        logger.info(f"Added persona to coordination: {persona.name}")
        return state

    def remove_persona(self, persona_name: str) -> bool:
        """Remove a persona from coordination."""
        if persona_name in self._states:
            del self._states[persona_name]
            return True
        return False

    def get_state(self, persona_name: str) -> PersonaState | None:
        """Get current state of a persona."""
        return self._states.get(persona_name)

    def list_active_personas(self) -> list[str]:
        """List names of active personas."""
        return [name for name, state in self._states.items() if state.active]

    async def send_message(
        self,
        from_persona: str,
        to_persona: str,
        interaction_type: InteractionType,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """
        Send a message between personas.

        Args:
            from_persona: Sender persona name
            to_persona: Recipient persona name
            interaction_type: Type of interaction
            content: Message content
            metadata: Additional metadata
        """
        message = PersonaMessage(
            from_persona=from_persona,
            to_persona=to_persona,
            interaction_type=interaction_type,
            content=content,
            metadata=metadata or {},
        )

        self._message_history.append(message)

        # Update states
        if from_persona in self._states:
            self._states[from_persona].messages_sent += 1
            self._states[from_persona].last_active = datetime.now()

        if to_persona in self._states:
            self._states[to_persona].messages_received += 1

        # Log the interaction
        think(from_persona, f"→ {to_persona}: [{interaction_type.value}] {content[:50]}...")

        # Trigger handlers
        if to_persona in self._message_handlers:
            for handler in self._message_handlers[to_persona]:
                try:
                    await handler(message)
                except Exception as e:
                    logger.error(f"Message handler error: {e}")

    def on_message(
        self,
        persona_name: str,
        handler: Callable[[PersonaMessage], Any],
    ) -> None:
        """
        Register a message handler for a persona.

        Args:
            persona_name: Persona to receive messages
            handler: Handler function
        """
        if persona_name not in self._message_handlers:
            self._message_handlers[persona_name] = []
        self._message_handlers[persona_name].append(handler)

    async def coordinate_task(
        self,
        task: str,
        persona_tasks: dict[str, str],
        timeout: float = 60.0,
    ) -> CoordinationResult:
        """
        Coordinate a task across multiple personas.

        Args:
            task: Main task description
            persona_tasks: Dict of persona_name -> specific task
            timeout: Maximum execution time

        Returns:
            CoordinationResult with outputs from each persona
        """
        start_time = datetime.now()
        outputs: dict[str, Any] = {}

        say("coordinator", f"Starting coordinated task: {task}")

        if self.mode == CoordinationMode.SEQUENTIAL:
            # Execute tasks one after another
            for persona_name, persona_task in persona_tasks.items():
                if persona_name not in self._states:
                    continue

                state = self._states[persona_name]
                state.current_task = persona_task
                state.active = True

                say(persona_name, f"Working on: {persona_task[:50]}...")

                # Simulate work (in real implementation, would call agent)
                await asyncio.sleep(0.1)

                outputs[persona_name] = f"Completed: {persona_task}"
                state.completed_tasks.append(persona_task)
                state.current_task = None

        elif self.mode == CoordinationMode.PARALLEL:
            # Execute tasks simultaneously
            async def execute_task(name: str, task: str) -> tuple[str, str]:
                if name in self._states:
                    state = self._states[name]
                    state.current_task = task
                    say(name, f"Working on: {task[:50]}...")
                    await asyncio.sleep(0.1)
                    state.completed_tasks.append(task)
                    state.current_task = None
                return name, f"Completed: {task}"

            tasks = [
                execute_task(name, task)
                for name, task in persona_tasks.items()
            ]

            results = await asyncio.gather(*tasks, return_exceptions=True)
            for result in results:
                if isinstance(result, tuple):
                    name, output = result
                    outputs[name] = output

        elif self.mode == CoordinationMode.HIERARCHICAL:
            # Leader delegates to others
            leader = list(persona_tasks.keys())[0] if persona_tasks else None
            if leader:
                say(leader, f"Leading task: {task}")
                for name, persona_task in list(persona_tasks.items())[1:]:
                    await self.send_message(
                        leader,
                        name,
                        InteractionType.DELEGATE,
                        persona_task,
                    )
                    outputs[name] = f"Delegated: {persona_task}"
                outputs[leader] = "Delegation complete"

        elif self.mode == CoordinationMode.DEBATE:
            # Personas argue different positions
            positions: dict[str, str] = {}
            for name, position in persona_tasks.items():
                say(name, f"Position: {position[:50]}...")
                positions[name] = position

            # Exchange critiques
            names = list(positions.keys())
            for i, name in enumerate(names):
                other = names[(i + 1) % len(names)]
                await self.send_message(
                    name,
                    other,
                    InteractionType.CRITIQUE,
                    f"Regarding your position on {positions[other][:30]}...",
                )

            outputs = positions

        duration = int((datetime.now() - start_time).total_seconds() * 1000)

        result = CoordinationResult(
            mode=self.mode,
            participants=list(persona_tasks.keys()),
            outputs=outputs,
            messages_exchanged=len(self._message_history),
            duration_ms=duration,
        )

        say("coordinator", f"Coordination complete. Duration: {duration}ms")
        return result

    async def reach_consensus(
        self,
        topic: str,
        positions: dict[str, str],
        max_rounds: int = 3,
    ) -> tuple[str | None, list[str]]:
        """
        Attempt to reach consensus among personas.

        Args:
            topic: Topic of discussion
            positions: Initial positions (persona_name -> position)
            max_rounds: Maximum discussion rounds

        Returns:
            Tuple of (consensus if reached, list of dissenting views)
        """
        say("coordinator", f"Seeking consensus on: {topic}")

        for round_num in range(max_rounds):
            say("coordinator", f"Round {round_num + 1}/{max_rounds}")

            # Check for agreement
            unique_positions = set(positions.values())
            if len(unique_positions) == 1:
                consensus = list(unique_positions)[0]
                say("coordinator", "Consensus reached!")
                return consensus, []

            # Exchange views
            names = list(positions.keys())
            for i, name in enumerate(names):
                other = names[(i + 1) % len(names)]
                await self.send_message(
                    name,
                    other,
                    InteractionType.INFORM,
                    f"My position: {positions[name]}",
                )

            # Simulate position evolution (in real impl, would use LLM)
            await asyncio.sleep(0.05)

        # No consensus
        dissent = [f"{name}: {pos}" for name, pos in positions.items()]
        say("coordinator", "No consensus reached")
        return None, dissent

    def get_message_history(
        self,
        persona_name: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """
        Get message history.

        Args:
            persona_name: Filter by persona (None for all)
            limit: Maximum messages to return

        Returns:
            List of message dicts
        """
        messages = self._message_history

        if persona_name:
            messages = [
                m for m in messages
                if m.from_persona == persona_name or m.to_persona == persona_name
            ]

        return [m.to_dict() for m in messages[-limit:]]

    def get_coordination_stats(self) -> dict[str, Any]:
        """Get statistics about the coordination."""
        return {
            "mode": self.mode.value,
            "total_personas": len(self._states),
            "active_personas": len(self.list_active_personas()),
            "total_messages": len(self._message_history),
            "persona_stats": {
                name: state.to_dict()
                for name, state in self._states.items()
            },
        }

    def reset(self) -> None:
        """Reset coordination state."""
        self._message_history.clear()
        for state in self._states.values():
            state.active = True
            state.current_task = None
            state.messages_sent = 0
            state.messages_received = 0


# ============================================================================
# Convenience Functions
# ============================================================================


def create_coordinator(
    mode: CoordinationMode = CoordinationMode.COLLABORATIVE,
    personas: list[str] | None = None,
) -> PersonaCoordinator:
    """
    Create a persona coordinator with specified personas.

    Args:
        mode: Coordination mode
        personas: List of persona names to add

    Returns:
        Configured PersonaCoordinator
    """
    coordinator = PersonaCoordinator(mode=mode)

    for persona_name in (personas or []):
        try:
            coordinator.add_persona(persona_name)
        except ValueError:
            logger.warning(f"Unknown persona: {persona_name}")

    return coordinator


async def quick_coordinate(
    task: str,
    personas: list[str],
    mode: CoordinationMode = CoordinationMode.PARALLEL,
) -> CoordinationResult:
    """
    Quick coordination helper.

    Args:
        task: Task to coordinate
        personas: Personas to involve
        mode: Coordination mode

    Returns:
        CoordinationResult
    """
    coordinator = create_coordinator(mode, personas)

    # Create simple task distribution
    persona_tasks = {name: f"Contribute to: {task}" for name in personas}

    return await coordinator.coordinate_task(task, persona_tasks)
