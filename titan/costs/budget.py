"""
Titan Costs - Budget Tracking

Provides budget management and spend tracking.

Enhanced with:
- Integration with TokenOptimizer for accurate token estimation
- Model-specific pricing with automatic updates
- Cost projection based on remaining work
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import TYPE_CHECKING, Any
from uuid import UUID, uuid4

if TYPE_CHECKING:
    from titan.prompts.token_optimizer import TokenOptimizer

logger = logging.getLogger("titan.costs.budget")


class BudgetExceededError(Exception):
    """Raised when budget limit is exceeded."""

    def __init__(
        self,
        message: str,
        current_spend: float,
        limit: float,
        session_id: str | None = None,
    ) -> None:
        super().__init__(message)
        self.current_spend = current_spend
        self.limit = limit
        self.session_id = session_id


@dataclass
class BudgetConfig:
    """Configuration for budget tracking."""

    # Session limits
    session_limit_usd: float = 10.0
    agent_limit_usd: float = 2.0

    # Global limits
    daily_limit_usd: float = 100.0
    monthly_limit_usd: float = 1000.0

    # Alerts
    alert_threshold_percent: float = 80.0
    enforce_limits: bool = True

    # Cost estimation
    default_cost_per_1k_tokens: float = 0.01
    include_cached_tokens: bool = False


@dataclass
class Budget:
    """Budget allocation for a task or session."""

    id: UUID = field(default_factory=uuid4)
    session_id: str = ""
    agent_id: str | None = None
    allocated_usd: float = 0.0
    spent_usd: float = 0.0
    created_at: datetime = field(default_factory=datetime.utcnow)
    expires_at: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def remaining_usd(self) -> float:
        """Get remaining budget."""
        return max(0.0, self.allocated_usd - self.spent_usd)

    @property
    def utilization_percent(self) -> float:
        """Get budget utilization percentage."""
        if self.allocated_usd <= 0:
            return 0.0
        return (self.spent_usd / self.allocated_usd) * 100

    @property
    def is_exhausted(self) -> bool:
        """Check if budget is exhausted."""
        return self.remaining_usd <= 0

    @property
    def is_expired(self) -> bool:
        """Check if budget has expired."""
        if self.expires_at is None:
            return False
        return datetime.utcnow() > self.expires_at

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": str(self.id),
            "session_id": self.session_id,
            "agent_id": self.agent_id,
            "allocated_usd": self.allocated_usd,
            "spent_usd": self.spent_usd,
            "remaining_usd": self.remaining_usd,
            "utilization_percent": self.utilization_percent,
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "metadata": self.metadata,
        }


@dataclass
class BudgetAllocation:
    """Allocation of budget for a specific task."""

    task_id: str
    allocated_usd: float
    model_tier: str
    reasoning: str
    constraints: list[str] = field(default_factory=list)


@dataclass
class SpendRecord:
    """Record of spending."""

    id: UUID = field(default_factory=uuid4)
    session_id: str = ""
    agent_id: str | None = None
    amount_usd: float = 0.0
    model: str = ""
    provider: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = field(default_factory=dict)


class BudgetTracker:
    """
    Tracks spending and enforces budget limits.

    Features:
    - Session and agent-level tracking
    - Daily/monthly aggregate limits
    - Alert thresholds
    - Redis persistence for distributed tracking
    - Integration with TokenOptimizer for accurate estimation
    - Cost projection for remaining work
    """

    def __init__(
        self,
        config: BudgetConfig | None = None,
        token_optimizer: "TokenOptimizer | None" = None,
    ) -> None:
        self.config = config or BudgetConfig()
        self._token_optimizer = token_optimizer
        self._budgets: dict[str, Budget] = {}  # session_id -> Budget
        self._agent_budgets: dict[str, Budget] = {}  # agent_id -> Budget
        self._spend_history: list[SpendRecord] = []
        self._daily_spend: float = 0.0
        self._monthly_spend: float = 0.0
        self._daily_reset: datetime = datetime.utcnow()
        self._monthly_reset: datetime = datetime.utcnow()
        self._lock = asyncio.Lock()
        self._alert_callbacks: list[Any] = []

        # Extended pricing table with more models
        self._pricing = {
            # Anthropic (input, output per 1K tokens)
            "claude-3-opus": (0.015, 0.075),
            "claude-opus-4": (0.015, 0.075),
            "claude-3-5-sonnet": (0.003, 0.015),
            "claude-sonnet-4": (0.003, 0.015),
            "claude-3-sonnet": (0.003, 0.015),
            "claude-3-haiku": (0.00025, 0.00125),
            "claude-3-5-haiku": (0.001, 0.005),
            # OpenAI
            "gpt-4-turbo": (0.01, 0.03),
            "gpt-4o": (0.005, 0.015),
            "gpt-4o-mini": (0.00015, 0.0006),
            "gpt-3.5-turbo": (0.0005, 0.0015),
            "o1": (0.015, 0.06),
            "o1-mini": (0.003, 0.012),
            # Groq (approximately)
            "llama-3-70b": (0.0007, 0.0008),
            "llama-3-8b": (0.0001, 0.0002),
            "mixtral-8x7b": (0.0005, 0.0005),
            # Google
            "gemini-1.5-pro": (0.00125, 0.005),
            "gemini-1.5-flash": (0.000075, 0.0003),
        }

    async def create_session_budget(
        self,
        session_id: str,
        limit_usd: float | None = None,
        duration_hours: float | None = None,
    ) -> Budget:
        """
        Create a budget for a session.

        Args:
            session_id: Session identifier
            limit_usd: Budget limit (uses config default if None)
            duration_hours: Budget duration (None = no expiry)

        Returns:
            Created Budget
        """
        async with self._lock:
            budget = Budget(
                session_id=session_id,
                allocated_usd=limit_usd or self.config.session_limit_usd,
                expires_at=datetime.utcnow() + timedelta(hours=duration_hours)
                if duration_hours else None,
            )
            self._budgets[session_id] = budget
            logger.info(f"Created session budget: {session_id} = ${budget.allocated_usd}")
            return budget

    async def create_agent_budget(
        self,
        agent_id: str,
        session_id: str,
        limit_usd: float | None = None,
    ) -> Budget:
        """
        Create a budget for an agent within a session.

        Args:
            agent_id: Agent identifier
            session_id: Parent session ID
            limit_usd: Budget limit (uses config default if None)

        Returns:
            Created Budget
        """
        async with self._lock:
            # Check session budget first
            session_budget = self._budgets.get(session_id)
            if session_budget:
                # Agent budget can't exceed session remaining
                max_limit = min(
                    limit_usd or self.config.agent_limit_usd,
                    session_budget.remaining_usd,
                )
            else:
                max_limit = limit_usd or self.config.agent_limit_usd

            budget = Budget(
                session_id=session_id,
                agent_id=agent_id,
                allocated_usd=max_limit,
            )
            self._agent_budgets[agent_id] = budget
            logger.info(f"Created agent budget: {agent_id} = ${budget.allocated_usd}")
            return budget

    async def record_spend(
        self,
        session_id: str,
        amount_usd: float,
        agent_id: str | None = None,
        model: str = "",
        provider: str = "",
        input_tokens: int = 0,
        output_tokens: int = 0,
        cached_tokens: int = 0,
    ) -> bool:
        """
        Record spending against a budget.

        Args:
            session_id: Session ID
            amount_usd: Amount spent
            agent_id: Optional agent ID
            model: Model used
            provider: Provider used
            input_tokens: Input token count
            output_tokens: Output token count
            cached_tokens: Cached token count

        Returns:
            True if spend was recorded, False if limit exceeded
        """
        async with self._lock:
            # Reset daily/monthly if needed
            await self._check_period_resets()

            # Create spend record
            record = SpendRecord(
                session_id=session_id,
                agent_id=agent_id,
                amount_usd=amount_usd,
                model=model,
                provider=provider,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cached_tokens=cached_tokens,
            )

            # Check global limits
            if self.config.enforce_limits:
                if self._daily_spend + amount_usd > self.config.daily_limit_usd:
                    raise BudgetExceededError(
                        "Daily budget limit exceeded",
                        self._daily_spend,
                        self.config.daily_limit_usd,
                        session_id,
                    )
                if self._monthly_spend + amount_usd > self.config.monthly_limit_usd:
                    raise BudgetExceededError(
                        "Monthly budget limit exceeded",
                        self._monthly_spend,
                        self.config.monthly_limit_usd,
                        session_id,
                    )

            # Update session budget
            session_budget = self._budgets.get(session_id)
            if session_budget:
                if self.config.enforce_limits and session_budget.remaining_usd < amount_usd:
                    raise BudgetExceededError(
                        "Session budget limit exceeded",
                        session_budget.spent_usd,
                        session_budget.allocated_usd,
                        session_id,
                    )
                session_budget.spent_usd += amount_usd

                # Check alert threshold
                if session_budget.utilization_percent >= self.config.alert_threshold_percent:
                    await self._trigger_alert(session_budget)

            # Update agent budget
            if agent_id:
                agent_budget = self._agent_budgets.get(agent_id)
                if agent_budget:
                    if self.config.enforce_limits and agent_budget.remaining_usd < amount_usd:
                        raise BudgetExceededError(
                            "Agent budget limit exceeded",
                            agent_budget.spent_usd,
                            agent_budget.allocated_usd,
                            session_id,
                        )
                    agent_budget.spent_usd += amount_usd

            # Update global tracking
            self._daily_spend += amount_usd
            self._monthly_spend += amount_usd
            self._spend_history.append(record)

            logger.debug(
                f"Recorded spend: ${amount_usd:.4f} for {session_id}"
                f" (daily: ${self._daily_spend:.2f}, monthly: ${self._monthly_spend:.2f})"
            )

            return True

    async def get_budget(self, session_id: str) -> Budget | None:
        """Get session budget."""
        return self._budgets.get(session_id)

    async def get_agent_budget(self, agent_id: str) -> Budget | None:
        """Get agent budget."""
        return self._agent_budgets.get(agent_id)

    async def get_remaining(self, session_id: str) -> float:
        """Get remaining budget for session."""
        budget = self._budgets.get(session_id)
        return budget.remaining_usd if budget else 0.0

    async def get_spend_summary(
        self,
        session_id: str | None = None,
        agent_id: str | None = None,
    ) -> dict[str, Any]:
        """Get spending summary."""
        records = self._spend_history
        if session_id:
            records = [r for r in records if r.session_id == session_id]
        if agent_id:
            records = [r for r in records if r.agent_id == agent_id]

        total_spend = sum(r.amount_usd for r in records)
        total_input = sum(r.input_tokens for r in records)
        total_output = sum(r.output_tokens for r in records)
        total_cached = sum(r.cached_tokens for r in records)

        # Group by model
        by_model: dict[str, float] = {}
        for r in records:
            by_model[r.model] = by_model.get(r.model, 0) + r.amount_usd

        return {
            "total_spend_usd": total_spend,
            "total_input_tokens": total_input,
            "total_output_tokens": total_output,
            "total_cached_tokens": total_cached,
            "request_count": len(records),
            "by_model": by_model,
            "daily_spend_usd": self._daily_spend,
            "monthly_spend_usd": self._monthly_spend,
        }

    async def estimate_cost(
        self,
        input_tokens: int,
        output_tokens: int,
        model: str,
    ) -> float:
        """
        Estimate cost for a request.

        Args:
            input_tokens: Expected input tokens
            output_tokens: Expected output tokens
            model: Model to use

        Returns:
            Estimated cost in USD
        """
        # Find matching pricing (try exact match, then partial)
        model_lower = model.lower()
        input_price, output_price = None, None

        # Exact match first
        if model_lower in self._pricing:
            input_price, output_price = self._pricing[model_lower]
        else:
            # Partial match
            for price_model, prices in self._pricing.items():
                if price_model in model_lower or model_lower in price_model:
                    input_price, output_price = prices
                    break

        # Default pricing if no match
        if input_price is None:
            input_price = self.config.default_cost_per_1k_tokens
            output_price = self.config.default_cost_per_1k_tokens

        cost = (input_tokens / 1000 * input_price) + (output_tokens / 1000 * output_price)
        return cost

    async def estimate_cost_for_text(
        self,
        input_text: str,
        expected_output_tokens: int,
        model: str,
    ) -> float:
        """
        Estimate cost for text using token optimizer.

        Args:
            input_text: Input text to estimate
            expected_output_tokens: Expected output tokens
            model: Model to use

        Returns:
            Estimated cost in USD
        """
        if self._token_optimizer:
            estimate = self._token_optimizer.estimate_tokens(input_text, model)
            input_tokens = estimate.estimated_tokens
        else:
            # Rough estimate: ~4 chars per token
            input_tokens = len(input_text) // 4

        return await self.estimate_cost(input_tokens, expected_output_tokens, model)

    async def project_session_cost(
        self,
        session_id: str,
        remaining_stages: int,
        model: str = "claude-3-5-sonnet",
        concise_mode: bool = False,
    ) -> dict[str, float]:
        """
        Project total cost for remaining session work.

        Args:
            session_id: Session ID
            remaining_stages: Number of stages remaining
            model: Expected model to use
            concise_mode: Whether concise prompts will be used

        Returns:
            Dict with projected costs
        """
        # Average tokens per stage (based on typical usage)
        if concise_mode:
            avg_input_tokens = 800
            avg_output_tokens = 300
        else:
            avg_input_tokens = 2000
            avg_output_tokens = 600

        # Get current spend
        budget = self._budgets.get(session_id)
        current_spend = budget.spent_usd if budget else 0.0

        # Project remaining cost
        per_stage_cost = await self.estimate_cost(
            avg_input_tokens,
            avg_output_tokens,
            model,
        )
        projected_remaining = per_stage_cost * remaining_stages
        projected_total = current_spend + projected_remaining

        return {
            "current_spend": current_spend,
            "projected_remaining": projected_remaining,
            "projected_total": projected_total,
            "per_stage_estimate": per_stage_cost,
            "remaining_stages": remaining_stages,
        }

    def set_token_optimizer(self, optimizer: "TokenOptimizer") -> None:
        """Set the token optimizer for accurate estimation."""
        self._token_optimizer = optimizer

    def update_pricing(self, model: str, input_price: float, output_price: float) -> None:
        """Update pricing for a model."""
        self._pricing[model.lower()] = (input_price, output_price)
        logger.info(f"Updated pricing for {model}: input=${input_price}/1K, output=${output_price}/1K")

    async def can_afford(
        self,
        session_id: str,
        estimated_cost: float,
        agent_id: str | None = None,
    ) -> tuple[bool, str]:
        """
        Check if a request can be afforded.

        Returns:
            Tuple of (can_afford, reason)
        """
        # Check global limits
        if self._daily_spend + estimated_cost > self.config.daily_limit_usd:
            return False, "Daily budget limit would be exceeded"
        if self._monthly_spend + estimated_cost > self.config.monthly_limit_usd:
            return False, "Monthly budget limit would be exceeded"

        # Check session budget
        session_budget = self._budgets.get(session_id)
        if session_budget and session_budget.remaining_usd < estimated_cost:
            return False, "Session budget would be exceeded"

        # Check agent budget
        if agent_id:
            agent_budget = self._agent_budgets.get(agent_id)
            if agent_budget and agent_budget.remaining_usd < estimated_cost:
                return False, "Agent budget would be exceeded"

        return True, "Within budget"

    async def _check_period_resets(self) -> None:
        """Check and reset daily/monthly counters if needed."""
        now = datetime.utcnow()

        # Daily reset
        if now.date() > self._daily_reset.date():
            self._daily_spend = 0.0
            self._daily_reset = now
            logger.info("Daily spend counter reset")

        # Monthly reset
        if now.month != self._monthly_reset.month or now.year != self._monthly_reset.year:
            self._monthly_spend = 0.0
            self._monthly_reset = now
            logger.info("Monthly spend counter reset")

    async def _trigger_alert(self, budget: Budget) -> None:
        """Trigger budget alert."""
        logger.warning(
            f"Budget alert: {budget.session_id} at {budget.utilization_percent:.1f}% "
            f"(${budget.spent_usd:.2f} / ${budget.allocated_usd:.2f})"
        )
        for callback in self._alert_callbacks:
            try:
                await callback(budget)
            except Exception as e:
                logger.error(f"Alert callback error: {e}")

    def add_alert_callback(self, callback: Any) -> None:
        """Add callback for budget alerts."""
        self._alert_callbacks.append(callback)


# Singleton instance
_default_tracker: BudgetTracker | None = None


def get_budget_tracker(config: BudgetConfig | None = None) -> BudgetTracker:
    """Get the default budget tracker."""
    global _default_tracker
    if _default_tracker is None:
        _default_tracker = BudgetTracker(config)
    return _default_tracker
