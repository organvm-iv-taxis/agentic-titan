"""
Hive Mind - Task Analyzer

LLM-powered task analysis for intelligent topology selection.
Goes beyond keyword matching to understand task semantics.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from hive.topology import TaskProfile, TopologyType

if TYPE_CHECKING:
    from adapters.router import LLMRouter

logger = logging.getLogger("titan.hive.analyzer")


# Structured prompt for LLM analysis
ANALYSIS_PROMPT = """Analyze the following task and determine the optimal agent topology.

Task: {task}

Available topologies:
- SWARM: All agents communicate with all others. Best for brainstorming, consensus, exploration.
- HIERARCHY: Tree structure with parent-child relationships. Best for command chains, delegation.
- PIPELINE: Sequential stages. Best for workflows, ETL, review chains.
- MESH: Resilient grid with multiple paths. Best for fault-tolerant, distributed tasks.
- RING: Circular token passing. Best for voting, round-robin processing.
- STAR: Central hub with spokes. Best for coordination, orchestration.

Respond with a JSON object containing:
{{
    "recommended_topology": "<SWARM|HIERARCHY|PIPELINE|MESH|RING|STAR>",
    "confidence": <0.0-1.0>,
    "reasoning": "<brief explanation>",
    "profile": {{
        "requires_consensus": <true|false>,
        "has_sequential_stages": <true|false>,
        "needs_fault_tolerance": <true|false>,
        "has_clear_leader": <true|false>,
        "is_voting_based": <true|false>,
        "parallel_subtasks": <number>,
        "complexity": "<low|medium|high>",
        "estimated_agents": <number>
    }}
}}

JSON response:"""


@dataclass
class AnalysisResult:
    """Result of task analysis."""

    recommended_topology: TopologyType
    confidence: float
    reasoning: str
    profile: TaskProfile
    raw_response: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "recommended_topology": self.recommended_topology.value,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "profile": {
                "requires_consensus": self.profile.requires_consensus,
                "has_sequential_stages": self.profile.has_sequential_stages,
                "needs_fault_tolerance": self.profile.needs_fault_tolerance,
                "has_clear_leader": self.profile.has_clear_leader,
                "is_voting_based": self.profile.is_voting_based,
                "parallel_subtasks": self.profile.parallel_subtasks,
                "complexity": self.profile.complexity,
                "estimated_agents": self.profile.estimated_agents,
            },
        }


class TaskAnalyzer:
    """
    LLM-powered task analyzer.

    Analyzes task descriptions to determine optimal topology,
    resource requirements, and agent composition.
    """

    def __init__(
        self,
        llm_router: LLMRouter | None = None,
        use_llm: bool = True,
    ) -> None:
        """
        Initialize analyzer.

        Args:
            llm_router: LLM router for intelligent analysis
            use_llm: Whether to use LLM (falls back to keyword matching if False)
        """
        self._llm_router = llm_router
        self._use_llm = use_llm and llm_router is not None
        self._analysis_cache: dict[str, AnalysisResult] = {}

    async def analyze(
        self,
        task: str,
        context: dict[str, Any] | None = None,
        use_cache: bool = True,
    ) -> AnalysisResult:
        """
        Analyze a task to determine optimal topology.

        Args:
            task: Task description
            context: Additional context (agent capabilities, constraints)
            use_cache: Whether to use cached results

        Returns:
            Analysis result with topology recommendation
        """
        # Check cache
        cache_key = task.strip().lower()
        if use_cache and cache_key in self._analysis_cache:
            logger.debug(f"Using cached analysis for: {task[:50]}...")
            return self._analysis_cache[cache_key]

        # Try LLM analysis
        if self._use_llm and self._llm_router:
            try:
                result = await self._analyze_with_llm(task, context)
                self._analysis_cache[cache_key] = result
                return result
            except Exception as e:
                logger.warning(f"LLM analysis failed, falling back to keywords: {e}")

        # Fallback to keyword-based analysis
        result = self._analyze_with_keywords(task)
        self._analysis_cache[cache_key] = result
        return result

    async def _analyze_with_llm(
        self,
        task: str,
        context: dict[str, Any] | None,
    ) -> AnalysisResult:
        """Analyze task using LLM."""
        prompt = ANALYSIS_PROMPT.format(task=task)

        if context:
            prompt = (
                f"Context:\n"
                f"- Available agents: {context.get('agent_count', 'unknown')}\n"
                f"- Capabilities: {context.get('capabilities', [])}\n"
                f"- Constraints: {context.get('constraints', 'none')}\n\n"
                f"{prompt}"
            )

        # Call LLM
        response = await self._llm_router.complete(
            prompt=prompt,
            max_tokens=500,
            temperature=0.1,  # Low temperature for deterministic analysis
        )

        # Parse JSON response
        try:
            # Extract JSON from response
            response_text = response.content
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0]
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0]

            data = json.loads(response_text.strip())
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse LLM response as JSON: {e}")
            raise

        # Parse topology
        topology_str = data.get("recommended_topology", "SWARM").upper()
        try:
            topology = TopologyType(topology_str.lower())
        except ValueError:
            topology = TopologyType.SWARM

        # Parse profile
        profile_data = data.get("profile", {})
        profile = TaskProfile(
            requires_consensus=profile_data.get("requires_consensus", False),
            has_sequential_stages=profile_data.get("has_sequential_stages", False),
            needs_fault_tolerance=profile_data.get("needs_fault_tolerance", False),
            has_clear_leader=profile_data.get("has_clear_leader", False),
            is_voting_based=profile_data.get("is_voting_based", False),
            parallel_subtasks=profile_data.get("parallel_subtasks", 0),
            complexity=profile_data.get("complexity", "medium"),
            estimated_agents=profile_data.get("estimated_agents", 2),
        )

        return AnalysisResult(
            recommended_topology=topology,
            confidence=data.get("confidence", 0.8),
            reasoning=data.get("reasoning", "LLM analysis"),
            profile=profile,
            raw_response=data,
        )

    def _analyze_with_keywords(self, task: str) -> AnalysisResult:
        """Analyze task using keyword matching (fallback)."""
        profile = TaskProfile.from_task(task)

        # Determine topology from profile
        if profile.is_voting_based:
            topology = TopologyType.RING
            reasoning = "Task involves voting or ranking"
        elif profile.requires_consensus:
            topology = TopologyType.SWARM
            reasoning = "Task requires consensus or brainstorming"
        elif profile.has_sequential_stages:
            topology = TopologyType.PIPELINE
            reasoning = "Task has sequential stages or workflow"
        elif profile.needs_fault_tolerance:
            topology = TopologyType.MESH
            reasoning = "Task needs fault tolerance"
        elif profile.has_clear_leader:
            topology = TopologyType.STAR if profile.estimated_agents <= 5 else TopologyType.HIERARCHY
            reasoning = "Task has a clear coordinator/leader"
        else:
            topology = TopologyType.SWARM
            reasoning = "Default to swarm (most flexible)"

        return AnalysisResult(
            recommended_topology=topology,
            confidence=0.6,  # Lower confidence for keyword matching
            reasoning=reasoning,
            profile=profile,
        )

    async def suggest_agents(
        self,
        task: str,
        available_capabilities: list[str],
    ) -> list[dict[str, Any]]:
        """
        Suggest agent composition for a task.

        Args:
            task: Task description
            available_capabilities: List of available agent capabilities

        Returns:
            List of suggested agents with roles
        """
        # Analyze task first
        analysis = await self.analyze(task)

        suggestions = []

        # Based on topology and profile, suggest agents
        if analysis.recommended_topology == TopologyType.PIPELINE:
            # Suggest stage-based agents
            stages = ["research", "analysis", "synthesis", "review"]
            for i, stage in enumerate(stages):
                matching_caps = [c for c in available_capabilities if stage in c.lower()]
                suggestions.append({
                    "role": f"stage-{i}",
                    "stage": i,
                    "suggested_capabilities": matching_caps or [available_capabilities[0]] if available_capabilities else [],
                    "description": f"{stage.capitalize()} stage",
                })

        elif analysis.recommended_topology == TopologyType.STAR:
            # Suggest hub + spokes
            suggestions.append({
                "role": "hub",
                "suggested_capabilities": ["planning", "coordination"],
                "description": "Central coordinator",
            })
            for i in range(min(3, len(available_capabilities))):
                suggestions.append({
                    "role": "spoke",
                    "suggested_capabilities": [available_capabilities[i]],
                    "description": f"Worker {i+1}",
                })

        elif analysis.recommended_topology == TopologyType.HIERARCHY:
            # Suggest tree structure
            suggestions.append({
                "role": "root",
                "suggested_capabilities": ["planning", "orchestration"],
                "description": "Root coordinator",
            })
            suggestions.append({
                "role": "manager",
                "suggested_capabilities": ["coordination"],
                "description": "Middle manager",
            })
            for cap in available_capabilities[:3]:
                suggestions.append({
                    "role": "worker",
                    "suggested_capabilities": [cap],
                    "description": f"Worker ({cap})",
                })

        else:
            # Default: peer agents with different capabilities
            for cap in available_capabilities[:5]:
                suggestions.append({
                    "role": "peer",
                    "suggested_capabilities": [cap],
                    "description": f"Peer agent ({cap})",
                })

        return suggestions

    def clear_cache(self) -> None:
        """Clear analysis cache."""
        self._analysis_cache.clear()
        logger.debug("Analysis cache cleared")


# Convenience function
async def analyze_task(
    task: str,
    llm_router: LLMRouter | None = None,
) -> AnalysisResult:
    """
    Analyze a task and get topology recommendation.

    Args:
        task: Task description
        llm_router: Optional LLM router for intelligent analysis

    Returns:
        Analysis result
    """
    analyzer = TaskAnalyzer(llm_router=llm_router)
    return await analyzer.analyze(task)
