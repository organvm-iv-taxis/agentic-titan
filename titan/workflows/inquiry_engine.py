"""
Titan Workflows - Inquiry Engine

Core workflow execution engine for multi-perspective collaborative inquiry.
Orchestrates the execution of inquiry stages with multi-model routing,
context accumulation, and real-time progress updates.

Enhanced with:
- Context compaction via TokenOptimizer
- Sub-agent summarization for large contexts
- Budget-aware prompt selection
- Prompt effectiveness tracking
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from collections.abc import AsyncGenerator, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import TYPE_CHECKING, Any

from titan.core.config import get_config
from titan.workflows.cognitive_router import (
    CognitiveRouter,
    CognitiveTaskType,
    get_cognitive_router,
)
from titan.workflows.inquiry_config import (
    EXPANSIVE_INQUIRY_WORKFLOW,
    CognitiveStyle,
    InfluenceMode,
    InquiryStage,
    InquiryWorkflow,
    UserInterjection,
)
from titan.workflows.inquiry_dag import (
    ExecutionMode,
    InquiryDependencyGraph,
)
from titan.workflows.inquiry_prompts import get_prompt, get_prompt_with_budget_awareness

if TYPE_CHECKING:
    from hive.memory import HiveMind
    from titan.costs.budget import BudgetTracker
    from titan.prompts.metrics import PromptTracker
    from titan.prompts.token_optimizer import TokenOptimizer

logger = logging.getLogger("titan.workflows.inquiry_engine")


class InquiryStatus(StrEnum):
    """Status of an inquiry session."""

    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class StageResult:
    """Result of executing a single inquiry stage."""

    stage_name: str
    role: str
    content: str
    model_used: str
    timestamp: datetime
    tokens_used: int = 0
    duration_ms: int = 0
    stage_index: int = 0
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def success(self) -> bool:
        """Whether the stage completed successfully."""
        return self.error is None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "stage_name": self.stage_name,
            "role": self.role,
            "content": self.content,
            "model_used": self.model_used,
            "timestamp": self.timestamp.isoformat(),
            "tokens_used": self.tokens_used,
            "duration_ms": self.duration_ms,
            "stage_index": self.stage_index,
            "error": self.error,
            "metadata": self.metadata,
        }


@dataclass
class InquirySession:
    """
    An active inquiry session.

    Tracks the state of a multi-stage inquiry including the topic,
    workflow configuration, current progress, and accumulated results.
    """

    id: str
    topic: str
    workflow: InquiryWorkflow
    status: InquiryStatus = InquiryStatus.PENDING
    current_stage: int = 0
    results: list[StageResult] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    started_at: datetime | None = None
    completed_at: datetime | None = None
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    # Conversational interleaving
    interjections: list[UserInterjection] = field(default_factory=list)
    pause_requested: bool = False
    # Temporal tracking for re-inquiry
    parent_session_id: str | None = None
    chain_id: str | None = None
    version: int = 1

    @property
    def total_stages(self) -> int:
        """Total number of stages in the workflow."""
        return len(self.workflow.stages)

    @property
    def progress(self) -> float:
        """Progress as a percentage (0-100)."""
        if self.total_stages == 0:
            return 100.0
        return (len(self.results) / self.total_stages) * 100

    @property
    def is_complete(self) -> bool:
        """Whether all stages have been executed."""
        return len(self.results) >= self.total_stages

    def get_previous_context(
        self,
        token_optimizer: TokenOptimizer | None = None,
        max_tokens: int | None = None,
        use_summarization: bool = True,
    ) -> str:
        """
        Get accumulated context from previous stages.

        Args:
            token_optimizer: Optional optimizer for context compression
            max_tokens: Maximum tokens for context (triggers compression)
            use_summarization: Whether to summarize older stages

        Returns:
            Context string (JSON or summarized)
        """
        if not self.results:
            return ""

        # Build full context
        context = {}
        for result in self.results:
            context[result.stage_name] = {
                "role": result.role,
                "content": result.content,
                "stage_index": result.stage_index,
            }

        full_context = json.dumps(context, indent=2)

        # Compress if optimizer provided and context is large
        if token_optimizer and max_tokens:
            estimate = token_optimizer.estimate_tokens(full_context)
            if estimate.estimated_tokens > max_tokens:
                if use_summarization:
                    # Use stage result compression
                    compressed_results = token_optimizer.compress_stage_results(
                        [r.to_dict() for r in self.results],
                        max_tokens_per_stage=max_tokens // max(len(self.results), 1),
                    )
                    return str(compressed_results)
                else:
                    # Use general compression
                    compression = token_optimizer.compress_context(
                        full_context,
                        max_tokens=max_tokens,
                    )
                    return str(compression.compressed_text)

        return full_context

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "topic": self.topic,
            "workflow_name": self.workflow.name,
            "status": self.status.value,
            "current_stage": self.current_stage,
            "total_stages": self.total_stages,
            "progress": self.progress,
            "results": [r.to_dict() for r in self.results],
            "interjections": [i.to_dict() for i in self.interjections],
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "error": self.error,
            "metadata": self.metadata,
            "parent_session_id": self.parent_session_id,
            "chain_id": self.chain_id,
            "version": self.version,
        }

    def get_unprocessed_interjections(self) -> list[UserInterjection]:
        """Get interjections that haven't been processed yet."""
        return [i for i in self.interjections if not i.processed]

    def get_interjections_for_stage(self, stage_index: int) -> list[UserInterjection]:
        """Get interjections relevant to a specific stage."""
        return [
            i for i in self.interjections if i.injected_at_stage < stage_index and not i.processed
        ]


class InquiryEngine:
    """
    Core engine for executing multi-perspective inquiry workflows.

    Features:
    - Multi-model routing based on cognitive task type
    - Context accumulation between stages with compression
    - Progress streaming via callbacks
    - Integration with Hive Mind for shared memory
    - Budget-aware prompt selection
    - Prompt effectiveness tracking
    """

    def __init__(
        self,
        cognitive_router: CognitiveRouter | None = None,
        hive_mind: HiveMind | None = None,
        llm_caller: Callable[[str, str], Any] | None = None,
        default_model: str | None = None,
        token_optimizer: TokenOptimizer | None = None,
        prompt_tracker: PromptTracker | None = None,
        budget_tracker: BudgetTracker | None = None,
        max_context_tokens: int | None = None,
        quality_gates: list[Any] | None = None,
    ) -> None:
        """
        Initialize the inquiry engine.

        Args:
            cognitive_router: Router for selecting models per cognitive task
            hive_mind: Shared memory for agents (optional)
            llm_caller: Function to call LLM (async). Signature: (prompt, model) -> response
            default_model: Fallback model when routing unavailable
                (defaults to config if omitted)
            token_optimizer: Optimizer for context compression
            prompt_tracker: Tracker for prompt effectiveness metrics
            budget_tracker: Tracker for budget management
            max_context_tokens: Maximum tokens for accumulated context
                (defaults to config if omitted)
            quality_gates: List of quality gates to run after stages
        """
        if default_model is None or max_context_tokens is None:
            config = get_config()
            if default_model is None:
                default_model = config.llm.default_model
            if max_context_tokens is None:
                max_context_tokens = config.max_context_tokens
        assert default_model is not None
        assert max_context_tokens is not None

        self._cognitive_router = cognitive_router or get_cognitive_router()
        self._hive_mind = hive_mind
        self._llm_caller = llm_caller
        self._default_model = default_model

        # Token optimization
        self._token_optimizer = token_optimizer
        self._prompt_tracker = prompt_tracker
        self._budget_tracker = budget_tracker
        self._max_context_tokens = max_context_tokens
        self._quality_gates = quality_gates or []

        # Active sessions
        self._sessions: dict[str, InquirySession] = {}

        # Event handlers
        self._on_stage_started: list[Callable[[InquirySession, int], None]] = []
        self._on_stage_completed: list[Callable[[InquirySession, StageResult], None]] = []
        self._on_session_completed: list[Callable[[InquirySession], None]] = []

        logger.info("Inquiry engine initialized")

    # =========================================================================
    # Session Management
    # =========================================================================

    async def start_inquiry(
        self,
        topic: str,
        workflow: InquiryWorkflow | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> InquirySession:
        """
        Start a new inquiry session.

        Args:
            topic: The topic to explore
            workflow: Workflow to use (defaults to EXPANSIVE_INQUIRY_WORKFLOW)
            metadata: Optional metadata to attach

        Returns:
            New InquirySession
        """
        session_id = f"inq-{uuid.uuid4().hex[:12]}"

        session = InquirySession(
            id=session_id,
            topic=topic,
            workflow=workflow or EXPANSIVE_INQUIRY_WORKFLOW,
            status=InquiryStatus.PENDING,
            metadata=metadata or {},
        )

        self._sessions[session_id] = session
        logger.info(f"Started inquiry session {session_id} for topic: {topic[:50]}...")

        # Store in Hive Mind if available
        if self._hive_mind:
            await self._hive_mind.set(
                f"inquiry:{session_id}",
                session.to_dict(),
                ttl=3600 * 24,  # 24 hours
            )

        return session

    def get_session(self, session_id: str) -> InquirySession | None:
        """Get an inquiry session by ID."""
        return self._sessions.get(session_id)

    def list_sessions(
        self,
        status: InquiryStatus | None = None,
    ) -> list[InquirySession]:
        """List all sessions, optionally filtered by status."""
        sessions = list(self._sessions.values())
        if status:
            sessions = [s for s in sessions if s.status == status]
        return sessions

    # =========================================================================
    # Stage Execution
    # =========================================================================

    async def run_stage(
        self,
        session: InquirySession,
        stage_index: int | None = None,
    ) -> StageResult:
        """
        Execute a single stage of the inquiry.

        Args:
            session: The inquiry session
            stage_index: Index of stage to run (defaults to next stage)

        Returns:
            StageResult with the stage output
        """
        # Determine which stage to run
        if stage_index is None:
            stage_index = len(session.results)

        if stage_index >= len(session.workflow.stages):
            raise ValueError(f"Stage index {stage_index} out of range")

        stage = session.workflow.stages[stage_index]
        session.current_stage = stage_index

        # Update session status
        if session.status == InquiryStatus.PENDING:
            session.status = InquiryStatus.RUNNING
            session.started_at = datetime.now()

        # Notify handlers
        for started_handler in self._on_stage_started:
            try:
                started_handler(session, stage_index)
            except Exception as e:
                logger.warning(f"Stage started handler error: {e}")

        logger.info(f"Running stage {stage_index + 1}/{len(session.workflow.stages)}: {stage.name}")

        start_time = time.time()

        try:
            # Get budget info if available
            budget_remaining = None
            budget_total = None
            if self._budget_tracker:
                budget = await self._budget_tracker.get_budget(session.id)
                if budget:
                    budget_remaining = budget.remaining_usd
                    budget_total = budget.allocated_usd

            # Build the prompt with budget awareness
            prompt, used_concise = self._build_stage_prompt(
                session,
                stage,
                budget_remaining=budget_remaining,
                budget_total=budget_total,
            )

            # Route to appropriate model
            cognitive_type = self._style_to_cognitive_type(stage.cognitive_style)
            routing = await self._cognitive_router.route_for_task(
                cognitive_type,
                preferred_model=stage.preferred_model,
            )
            model = routing.model_id

            logger.debug(f"Using model {model} for {stage.name} (score: {routing.score})")

            # Call the LLM
            if self._llm_caller:
                response = await self._llm_caller(prompt, model)
                content = response if isinstance(response, str) else str(response)
                tokens_used = len(content.split()) * 2  # Rough estimate
            else:
                # Mock response for testing
                content = self._mock_stage_response(stage, session.topic)
                tokens_used = 0

            duration_ms = int((time.time() - start_time) * 1000)

            # Estimate prompt tokens
            prompt_tokens = 0
            if self._token_optimizer:
                prompt_tokens = self._token_optimizer.estimate_tokens(
                    prompt, model
                ).estimated_tokens

            result = StageResult(
                stage_name=stage.name,
                role=stage.role,
                content=content,
                model_used=model,
                timestamp=datetime.now(),
                tokens_used=tokens_used,
                duration_ms=duration_ms,
                stage_index=stage_index,
                metadata={
                    "routing_score": routing.score,
                    "routing_reasoning": routing.reasoning,
                    "cognitive_style": stage.cognitive_style.value,
                    "used_concise_prompt": used_concise,
                    "prompt_tokens": prompt_tokens,
                },
            )

            # Record epistemic signature metrics
            self._record_epistemic_metrics(session.id, stage.cognitive_style, content)

            # Run quality gates
            for gate in self._quality_gates:
                try:
                    # Avoid circular import by checking type name or just duck typing
                    if type(gate).__name__ == "DialecticGate":
                        gate_result = await gate.evaluate(session)
                        session.metadata["quality_check"] = gate_result.metadata
                        if not gate_result.passed:
                            logger.warning(f"Quality gate failed: {gate_result.issues}")
                except Exception as e:
                    logger.error(f"Quality gate execution failed: {e}")

            # Track prompt effectiveness if tracker available
            if self._prompt_tracker:
                # Estimate cost based on model
                cost_usd = 0.0
                if self._budget_tracker:
                    cost_usd = await self._budget_tracker.estimate_cost(
                        prompt_tokens,
                        tokens_used,
                        model,
                    )

                self._prompt_tracker.record(
                    stage_name=stage.name,
                    model=model,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=tokens_used,
                    quality_score=0.7,  # Default; would be from evaluator
                    latency_ms=duration_ms,
                    cost_usd=cost_usd,
                    prompt_variant="concise" if used_concise else "default",
                    adaptations_applied=["budget_aware"] if used_concise else [],
                )

        except Exception as e:
            logger.error(f"Stage {stage.name} failed: {e}")
            duration_ms = int((time.time() - start_time) * 1000)

            result = StageResult(
                stage_name=stage.name,
                role=stage.role,
                content="",
                model_used=self._default_model,
                timestamp=datetime.now(),
                duration_ms=duration_ms,
                stage_index=stage_index,
                error=str(e),
            )

        # Store result
        session.results.append(result)

        # Notify handlers
        for completion_handler in self._on_stage_completed:
            try:
                completion_handler(session, result)
            except Exception as e:
                logger.warning(f"Stage completed handler error: {e}")

        # Update Hive Mind
        if self._hive_mind:
            await self._hive_mind.set(
                f"inquiry:{session.id}",
                session.to_dict(),
                ttl=3600 * 24,
            )

        return result

    async def run_full_workflow(
        self,
        session: InquirySession,
    ) -> InquirySession:
        """
        Run all stages of the workflow sequentially.

        Args:
            session: The inquiry session to run

        Returns:
            The completed session
        """
        logger.info(f"Running full workflow for session {session.id}")

        try:
            while len(session.results) < len(session.workflow.stages):
                if session.status == InquiryStatus.CANCELLED:
                    logger.info(f"Session {session.id} was cancelled")
                    break

                await self.run_stage(session)

            # Mark complete
            if session.status == InquiryStatus.RUNNING:
                session.status = InquiryStatus.COMPLETED
                session.completed_at = datetime.now()

                # Notify handlers
                for handler in self._on_session_completed:
                    try:
                        handler(session)
                    except Exception as e:
                        logger.warning(f"Session completed handler error: {e}")

            logger.info(
                f"Workflow complete for session {session.id}. Ran {len(session.results)} stages."
            )

        except Exception as e:
            logger.error(f"Workflow failed for session {session.id}: {e}")
            session.status = InquiryStatus.FAILED
            session.error = str(e)
            session.completed_at = datetime.now()

        # Update Hive Mind
        if self._hive_mind:
            await self._hive_mind.set(
                f"inquiry:{session.id}",
                session.to_dict(),
                ttl=3600 * 24 * 7,  # Keep completed sessions for 7 days
            )

        return session

    async def run_dag_workflow(
        self,
        session: InquirySession,
        execution_mode: ExecutionMode = ExecutionMode.STAGED,
    ) -> InquirySession:
        """
        Run workflow stages using DAG-based execution ordering.

        This method respects stage dependencies and can execute independent
        stages in parallel when execution_mode is STAGED or PARALLEL.

        Args:
            session: The inquiry session to run
            execution_mode: How to execute stages:
                - SEQUENTIAL: One stage at a time, respecting dependency order
                - PARALLEL: All ready stages at once
                - STAGED: Level by level (default, recommended)

        Returns:
            The completed session
        """
        logger.info(f"Running DAG workflow for session {session.id} (mode: {execution_mode.value})")

        # Build dependency graph from workflow
        graph = InquiryDependencyGraph.from_workflow(session.workflow)

        # Store graph info in session metadata
        session.metadata["dag_info"] = {
            "execution_mode": execution_mode.value,
            "levels": graph.topological_sort(),
            "can_parallelize": graph.can_parallelize(),
        }

        try:
            if execution_mode == ExecutionMode.SEQUENTIAL:
                await self._run_dag_sequential(session, graph)
            elif execution_mode == ExecutionMode.PARALLEL:
                await self._run_dag_parallel(session, graph)
            else:  # STAGED
                await self._run_dag_staged(session, graph)

            # Mark complete
            if session.status == InquiryStatus.RUNNING:
                session.status = InquiryStatus.COMPLETED
                session.completed_at = datetime.now()

                # Notify handlers
                for handler in self._on_session_completed:
                    try:
                        handler(session)
                    except Exception as e:
                        logger.warning(f"Session completed handler error: {e}")

            logger.info(
                f"DAG workflow complete for session {session.id}. "
                f"Ran {len(session.results)} stages."
            )

        except Exception as e:
            logger.error(f"DAG workflow failed for session {session.id}: {e}")
            session.status = InquiryStatus.FAILED
            session.error = str(e)
            session.completed_at = datetime.now()

        # Update Hive Mind
        if self._hive_mind:
            await self._hive_mind.set(
                f"inquiry:{session.id}",
                session.to_dict(),
                ttl=3600 * 24 * 7,
            )

        return session

    async def _run_dag_sequential(
        self,
        session: InquirySession,
        graph: InquiryDependencyGraph,
    ) -> None:
        """Execute DAG stages one at a time in dependency order."""
        completed: set[int] = set()

        while len(completed) < len(graph):
            if session.status == InquiryStatus.CANCELLED:
                break

            ready = graph.get_ready_stages(completed)
            if not ready:
                logger.warning("No ready stages - possible cycle or completion")
                break

            # Execute first ready stage
            stage_idx = ready[0]
            await self.run_stage(session, stage_idx)
            graph.mark_completed(stage_idx)
            completed.add(stage_idx)

    async def _run_dag_parallel(
        self,
        session: InquirySession,
        graph: InquiryDependencyGraph,
    ) -> None:
        """Execute all ready stages in parallel."""
        completed: set[int] = set()

        while len(completed) < len(graph):
            if session.status == InquiryStatus.CANCELLED:
                break

            ready = graph.get_ready_stages(completed)
            if not ready:
                break

            logger.info(f"Executing {len(ready)} stages in parallel: {ready}")

            # Execute all ready stages concurrently
            tasks = [self.run_stage(session, idx) for idx in ready]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Mark completed and handle errors
            for idx, result in zip(ready, results):
                if isinstance(result, Exception):
                    logger.error(f"Stage {idx} failed: {result}")
                    graph.mark_failed(idx)
                else:
                    graph.mark_completed(idx)
                completed.add(idx)

    async def _run_dag_staged(
        self,
        session: InquirySession,
        graph: InquiryDependencyGraph,
    ) -> None:
        """Execute stages level by level (respecting dependencies)."""
        levels = graph.topological_sort()

        for level_idx, level_stages in enumerate(levels):
            if session.status == InquiryStatus.CANCELLED:
                break

            logger.info(f"Executing DAG level {level_idx + 1}/{len(levels)}: stages {level_stages}")

            if len(level_stages) == 1:
                # Single stage - execute directly
                await self.run_stage(session, level_stages[0])
                graph.mark_completed(level_stages[0])
            else:
                # Multiple stages - execute in parallel
                tasks = [self.run_stage(session, idx) for idx in level_stages]
                results = await asyncio.gather(*tasks, return_exceptions=True)

                for idx, result in zip(level_stages, results):
                    if isinstance(result, Exception):
                        logger.error(f"Stage {idx} failed: {result}")
                        graph.mark_failed(idx)
                    else:
                        graph.mark_completed(idx)

    def get_dag_context_for_stage(
        self,
        session: InquirySession,
        stage_idx: int,
    ) -> str:
        """
        Get context specifically from dependency stages (DAG-aware).

        Unlike sequential context which includes all previous stages,
        DAG context only includes results from stages this one depends on.

        Args:
            session: The inquiry session
            stage_idx: Index of the stage needing context

        Returns:
            Context string from dependency stages
        """
        graph = InquiryDependencyGraph.from_workflow(session.workflow)
        context_stage_indices = graph.get_context_stages(stage_idx)

        if not context_stage_indices:
            return ""

        # Build context from dependency stages only
        context = {}
        for result in session.results:
            if result.stage_index in context_stage_indices:
                context[result.stage_name] = {
                    "role": result.role,
                    "content": result.content,
                    "stage_index": result.stage_index,
                }

        return json.dumps(context, indent=2) if context else ""

    async def stream_workflow(
        self,
        session: InquirySession,
    ) -> AsyncGenerator[dict[str, Any], None]:
        """
        Run workflow with streaming progress updates.

        Yields dictionaries with event type and data for each stage.

        Args:
            session: The inquiry session

        Yields:
            Progress events
        """
        yield {
            "type": "session_started",
            "session_id": session.id,
            "topic": session.topic,
            "total_stages": session.total_stages,
        }

        try:
            while len(session.results) < session.total_stages:
                if session.status == InquiryStatus.CANCELLED:
                    yield {
                        "type": "session_cancelled",
                        "session_id": session.id,
                    }
                    break

                stage_index = len(session.results)
                stage = session.workflow.stages[stage_index]

                yield {
                    "type": "stage_started",
                    "session_id": session.id,
                    "stage_index": stage_index,
                    "stage_name": stage.name,
                    "role": stage.role,
                }

                result = await self.run_stage(session)

                yield {
                    "type": "stage_completed",
                    "session_id": session.id,
                    "stage_index": stage_index,
                    "result": result.to_dict(),
                }

            if session.status != InquiryStatus.CANCELLED:
                session.status = InquiryStatus.COMPLETED
                session.completed_at = datetime.now()

                yield {
                    "type": "session_completed",
                    "session_id": session.id,
                    "results_count": len(session.results),
                }

        except Exception as e:
            session.status = InquiryStatus.FAILED
            session.error = str(e)
            session.completed_at = datetime.now()

            yield {
                "type": "session_failed",
                "session_id": session.id,
                "error": str(e),
            }

    def cancel_session(self, session_id: str) -> bool:
        """Cancel a running session."""
        session = self._sessions.get(session_id)
        if session and session.status == InquiryStatus.RUNNING:
            session.status = InquiryStatus.CANCELLED
            session.completed_at = datetime.now()
            logger.info(f"Cancelled session {session_id}")
            return True
        return False

    # =========================================================================
    # Conversational Interleaving
    # =========================================================================

    def pause_session(self, session_id: str) -> bool:
        """
        Request pause of a running session at the next stage boundary.

        The session will complete the current stage and then pause
        before starting the next stage.

        Args:
            session_id: Session to pause

        Returns:
            True if pause was requested successfully
        """
        session = self._sessions.get(session_id)
        if not session:
            logger.warning(f"Session not found: {session_id}")
            return False

        if session.status not in (InquiryStatus.RUNNING, InquiryStatus.PENDING):
            logger.warning(f"Cannot pause session in state: {session.status}")
            return False

        session.pause_requested = True
        logger.info(f"Pause requested for session {session_id}")
        return True

    def inject_user_input(
        self,
        session_id: str,
        content: str,
        mode: InfluenceMode | str = InfluenceMode.CONTEXT,
    ) -> UserInterjection | None:
        """
        Inject user input into a paused or running session.

        Args:
            session_id: Session to inject into
            content: User's input text
            mode: How the input should influence the inquiry

        Returns:
            The created UserInterjection, or None if failed
        """
        session = self._sessions.get(session_id)
        if not session:
            logger.warning(f"Session not found: {session_id}")
            return None

        # Convert string mode to enum
        if isinstance(mode, str):
            mode = InfluenceMode(mode)

        # Create interjection
        interjection = UserInterjection(
            content=content,
            injected_at_stage=session.current_stage,
            influence_mode=mode,
        )

        session.interjections.append(interjection)
        logger.info(
            f"User interjection added to session {session_id} "
            f"at stage {session.current_stage} (mode: {mode.value})"
        )

        return interjection

    def resume_session(self, session_id: str) -> bool:
        """
        Resume a paused session.

        Args:
            session_id: Session to resume

        Returns:
            True if session was resumed
        """
        session = self._sessions.get(session_id)
        if not session:
            logger.warning(f"Session not found: {session_id}")
            return False

        if session.status != InquiryStatus.PAUSED:
            logger.warning(f"Cannot resume session in state: {session.status}")
            return False

        session.status = InquiryStatus.RUNNING
        session.pause_requested = False
        logger.info(f"Session {session_id} resumed")
        return True

    async def run_interleaved_workflow(
        self,
        session: InquirySession,
    ) -> AsyncGenerator[dict[str, Any], None]:
        """
        Run workflow with support for pausing and user interjections.

        This is similar to stream_workflow but checks for pause requests
        and processes interjections between stages.

        Args:
            session: The inquiry session

        Yields:
            Progress events including pause and interjection events
        """
        yield {
            "type": "session_started",
            "session_id": session.id,
            "topic": session.topic,
            "total_stages": session.total_stages,
        }

        try:
            while len(session.results) < session.total_stages:
                # Check for cancellation
                if session.status == InquiryStatus.CANCELLED:
                    yield {
                        "type": "session_cancelled",
                        "session_id": session.id,
                    }
                    break

                # Check for pause request
                if session.pause_requested:
                    session.status = InquiryStatus.PAUSED
                    session.pause_requested = False
                    yield {
                        "type": "session_paused",
                        "session_id": session.id,
                        "current_stage": session.current_stage,
                        "stages_completed": len(session.results),
                    }
                    # Wait for resume
                    while session.status == InquiryStatus.PAUSED:
                        await asyncio.sleep(0.1)
                    if session.status == InquiryStatus.CANCELLED:
                        yield {
                            "type": "session_cancelled",
                            "session_id": session.id,
                        }
                        return
                    yield {
                        "type": "session_resumed",
                        "session_id": session.id,
                    }

                # Process any pending interjections
                unprocessed = session.get_unprocessed_interjections()
                for interjection in unprocessed:
                    yield {
                        "type": "interjection_processing",
                        "session_id": session.id,
                        "interjection": interjection.to_dict(),
                    }
                    interjection.processed = True

                # Run next stage
                stage_index = len(session.results)
                stage = session.workflow.stages[stage_index]

                yield {
                    "type": "stage_started",
                    "session_id": session.id,
                    "stage_index": stage_index,
                    "stage_name": stage.name,
                    "role": stage.role,
                }

                result = await self.run_stage(session)

                yield {
                    "type": "stage_completed",
                    "session_id": session.id,
                    "stage_index": stage_index,
                    "result": result.to_dict(),
                }

            if session.status not in (InquiryStatus.CANCELLED, InquiryStatus.PAUSED):
                session.status = InquiryStatus.COMPLETED
                session.completed_at = datetime.now()

                yield {
                    "type": "session_completed",
                    "session_id": session.id,
                    "results_count": len(session.results),
                }

        except Exception as e:
            session.status = InquiryStatus.FAILED
            session.error = str(e)
            session.completed_at = datetime.now()

            yield {
                "type": "session_failed",
                "session_id": session.id,
                "error": str(e),
            }

    def _get_interjection_context(self, session: InquirySession) -> str:
        """
        Build context string from user interjections.

        Args:
            session: The inquiry session

        Returns:
            Context string incorporating user interjections
        """
        relevant = session.get_interjections_for_stage(session.current_stage + 1)
        if not relevant:
            return ""

        parts = ["User clarifications/directions:"]
        for i, interjection in enumerate(relevant, 1):
            mode_label = {
                InfluenceMode.CONTEXT: "Additional context",
                InfluenceMode.REDIRECT: "Direction change",
                InfluenceMode.CLARIFY: "Clarification",
            }.get(interjection.influence_mode, "Input")
            parts.append(f"{i}. [{mode_label}]: {interjection.content}")

        return "\n".join(parts)

    # =========================================================================
    # Event Handlers
    # =========================================================================

    def on_stage_started(
        self,
        handler: Callable[[InquirySession, int], None],
    ) -> None:
        """Register a handler for stage start events."""
        self._on_stage_started.append(handler)

    def on_stage_completed(
        self,
        handler: Callable[[InquirySession, StageResult], None],
    ) -> None:
        """Register a handler for stage completion events."""
        self._on_stage_completed.append(handler)

    def on_session_completed(
        self,
        handler: Callable[[InquirySession], None],
    ) -> None:
        """Register a handler for session completion events."""
        self._on_session_completed.append(handler)

    # =========================================================================
    # Internal Methods
    # =========================================================================

    def _build_stage_prompt(
        self,
        session: InquirySession,
        stage: InquiryStage,
        budget_remaining: float | None = None,
        budget_total: float | None = None,
    ) -> tuple[str, bool]:
        """
        Build the prompt for a stage with context compression and budget awareness.

        Args:
            session: The inquiry session
            stage: The stage to build prompt for
            budget_remaining: Remaining budget (for budget-aware selection)
            budget_total: Total budget

        Returns:
            Tuple of (prompt string, whether concise variant was used)
        """
        previous_context = ""
        if session.workflow.context_accumulation and session.results:
            # Check if using DAG-aware context
            if session.metadata.get("dag_info", {}).get("execution_mode") in ["staged", "parallel"]:
                previous_context = self.get_dag_context_for_stage(session, len(session.results))
            else:
                previous_context = session.get_previous_context(
                    token_optimizer=self._token_optimizer,
                    max_tokens=self._max_context_tokens,
                    use_summarization=True,
                )

        # Use budget-aware prompt selection if budget info available
        if budget_remaining is not None and budget_total is not None:
            prompt_with_budget = get_prompt_with_budget_awareness(
                template_key=stage.prompt_template,
                topic=session.topic,
                previous_context=previous_context,
                stage_number=len(session.results) + 1,
                total_stages=session.total_stages,
                budget_remaining=budget_remaining,
                budget_total=budget_total,
                token_optimizer=self._token_optimizer,
            )
            prompt_text, used_concise = prompt_with_budget
            return str(prompt_text), bool(used_concise)

        # Standard prompt generation
        prompt = get_prompt(
            template_key=stage.prompt_template,
            topic=session.topic,
            previous_context=previous_context,
            stage_number=len(session.results) + 1,
            total_stages=session.total_stages,
            token_optimizer=self._token_optimizer,
            max_context_tokens=self._max_context_tokens,
        )
        return prompt, False

    def _record_epistemic_metrics(
        self,
        session_id: str,
        style: CognitiveStyle,
        content: str,
    ) -> None:
        """Record epistemic signature metrics based on stage content and style."""
        from titan.metrics import get_metrics

        metrics = get_metrics()

        # Estimate "intensity" based on response length and style
        intensity = min(len(content.split()) / 500.0, 1.0)

        if style == CognitiveStyle.STRUCTURED_REASONING:
            metrics.set_inquiry_logic_density(session_id, intensity)
        elif style == CognitiveStyle.CREATIVE_SYNTHESIS:
            metrics.set_inquiry_mythic_depth(session_id, intensity)
        elif style == CognitiveStyle.CROSS_DOMAIN:
            metrics.set_inquiry_lateral_breadth(session_id, intensity)
        elif style == CognitiveStyle.META_ANALYSIS:
            metrics.set_inquiry_recursive_depth(session_id, intensity)
        elif style == CognitiveStyle.PATTERN_RECOGNITION:
            metrics.set_inquiry_pattern_strength(session_id, intensity)

    def _style_to_cognitive_type(
        self,
        style: CognitiveStyle,
    ) -> CognitiveTaskType:
        """Map cognitive style to cognitive task type for routing."""
        mapping = {
            CognitiveStyle.STRUCTURED_REASONING: CognitiveTaskType.STRUCTURED_REASONING,
            CognitiveStyle.CREATIVE_SYNTHESIS: CognitiveTaskType.CREATIVE_SYNTHESIS,
            CognitiveStyle.MATHEMATICAL_ANALYSIS: CognitiveTaskType.MATHEMATICAL_ANALYSIS,
            CognitiveStyle.CROSS_DOMAIN: CognitiveTaskType.CROSS_DOMAIN,
            CognitiveStyle.META_ANALYSIS: CognitiveTaskType.META_ANALYSIS,
            CognitiveStyle.PATTERN_RECOGNITION: CognitiveTaskType.PATTERN_RECOGNITION,
        }
        return mapping.get(style, CognitiveTaskType.STRUCTURED_REASONING)

    def _mock_stage_response(
        self,
        stage: InquiryStage,
        topic: str,
    ) -> str:
        """Generate mock response for testing without LLM."""
        return f"""## {stage.name} Analysis

**Topic:** {topic}

**Role:** {stage.role}

This is a mock response for the {stage.name} stage.
In production, this would contain the AI's actual analysis.

### Key Insights
- Insight 1 related to {topic}
- Insight 2 exploring different angles
- Insight 3 with deeper analysis

### {stage.description}

The {stage.role} has analyzed the topic through its unique cognitive lens,
revealing aspects that complement other perspectives.
"""


# =============================================================================
# Factory Functions
# =============================================================================

_default_engine: InquiryEngine | None = None


def get_inquiry_engine() -> InquiryEngine:
    """Get the default inquiry engine instance."""
    global _default_engine
    if _default_engine is None:
        # Import here to avoid circular dependency
        from titan.workflows.quality_gates import DialecticGate

        _default_engine = InquiryEngine(quality_gates=[DialecticGate()])
    return _default_engine


async def quick_inquiry(
    topic: str,
    workflow_name: str = "expansive",
) -> InquirySession:
    """
    Quick helper to run a full inquiry and return results.

    Args:
        topic: Topic to explore
        workflow_name: Name of workflow to use

    Returns:
        Completed InquirySession
    """
    from titan.workflows.inquiry_config import get_workflow

    engine = get_inquiry_engine()
    workflow = get_workflow(workflow_name) or EXPANSIVE_INQUIRY_WORKFLOW

    session = await engine.start_inquiry(topic, workflow)
    return await engine.run_full_workflow(session)
