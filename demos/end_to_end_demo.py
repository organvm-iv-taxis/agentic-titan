#!/usr/bin/env python3
"""
Agentic Titan - End-to-End Demo

Showcases all Titan capabilities in a unified demonstration:
1. Swarm topology for brainstorming
2. Pipeline topology for research-code-review workflow
3. Hierarchy topology for expert review
4. Voting mechanisms for consensus
5. Retrospective and learning
6. Episode recording and retrieval

Usage:
    python demos/end_to_end_demo.py
"""

from __future__ import annotations

import asyncio
import time
from datetime import datetime
from typing import Any

# Rich console for beautiful output
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich import box
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

# Titan imports
from hive.topology import TopologyEngine, TopologyType, TaskProfile
from hive.decision.voting import VotingSession, VotingStrategy, Vote, VotingResult
from hive.learning import EpisodicLearner, Episode, EpisodeOutcome

console = Console() if RICH_AVAILABLE else None


def print_header(text: str) -> None:
    """Print a section header."""
    if console:
        console.print(Panel(text, style="bold blue", box=box.DOUBLE))
    else:
        print("\n" + "=" * 60)
        print(text)
        print("=" * 60)


def print_phase(phase: int, emoji: str, title: str, description: str) -> None:
    """Print a phase header."""
    if console:
        console.print(f"\n[bold cyan]{emoji}  PHASE {phase}: {title}[/bold cyan]")
        console.print(f"   [dim]{description}[/dim]\n")
    else:
        print(f"\n{emoji} PHASE {phase}: {title}")
        print(f"   {description}\n")


def print_result(message: str, **kwargs: Any) -> None:
    """Print a result line."""
    if console:
        formatted = message
        for key, value in kwargs.items():
            formatted += f" [yellow]{key}[/yellow]=[green]{value}[/green]"
        console.print(f"   {formatted}")
    else:
        formatted = message
        for key, value in kwargs.items():
            formatted += f" {key}={value}"
        print(f"   {formatted}")


def print_agent_action(agent: str, action: str) -> None:
    """Print an agent action."""
    colors = {
        "Researcher": "bright_magenta",
        "Coder": "bright_green",
        "Reviewer": "bright_yellow",
    }
    color = colors.get(agent, "white")
    if console:
        console.print(f"   [{color}][{agent}][/{color}] {action}")
    else:
        print(f"   [{agent}] {action}")


class MockAgent:
    """Mock agent for demo purposes."""

    def __init__(self, agent_id: str, name: str, archetype: str, capabilities: list[str]):
        self.agent_id = agent_id
        self.name = name
        self.archetype = archetype
        self.capabilities = capabilities
        self.weight = 1.0

    async def work(self, task: str) -> dict[str, Any]:
        """Simulate agent work."""
        await asyncio.sleep(0.1)  # Simulate work

        if self.archetype == "researcher":
            return {
                "questions": [
                    "What are the key challenges in quantum computing?",
                    "How do current frameworks compare?",
                    "What are the scalability considerations?",
                    "What are the error correction approaches?",
                    "What are the hardware requirements?",
                ],
                "sources": ["arXiv", "Nature", "IEEE", "Google AI Blog"],
                "summary": "Comprehensive research on quantum computing frameworks completed.",
            }
        elif self.archetype == "coder":
            return {
                "files_created": ["quantum_circuit.py", "simulator.py"],
                "lines_written": 142,
                "tests_written": 8,
                "code_output": "# Quantum circuit implementation\nclass QuantumCircuit:\n    pass",
            }
        elif self.archetype == "reviewer":
            return {
                "issues_found": [
                    {"severity": "critical", "message": "Missing error handling in qubit_measure()"},
                    {"severity": "critical", "message": "Potential race condition in parallel execution"},
                ],
                "suggestions": [
                    {"message": "Consider adding type hints for better IDE support"},
                    {"message": "Add docstrings to public methods"},
                ],
                "approved": False,
                "confidence": 0.85,
            }
        return {"status": "completed"}


async def phase_1_swarm(engine: TopologyEngine, agents: list[MockAgent]) -> dict[str, Any]:
    """Phase 1: Swarm topology for brainstorming."""
    print_phase(1, "SWARM", "SWARM", "All agents discuss approach together")

    start = time.time()

    # Create swarm topology
    topology = engine.create_topology(TopologyType.SWARM)

    # Add all agents
    for agent in agents:
        topology.add_agent(
            agent_id=agent.agent_id,
            name=agent.name,
            capabilities=agent.capabilities,
        )

    print_result("Topology created:", type="swarm", agents=len(agents))

    # Simulate brainstorming
    framework_options = ["cirq", "qiskit", "pennylane", "pyquil"]
    votes = []

    for agent in agents:
        # Each agent "thinks" and proposes
        print_agent_action(agent.name, "Analyzing framework options...")
        await asyncio.sleep(0.05)

        # Simulate vote based on agent type
        if agent.archetype == "researcher":
            choice = "cirq"
            confidence = 0.85
            reasoning = "Best documentation and Google backing"
        elif agent.archetype == "coder":
            choice = "cirq"
            confidence = 0.72
            reasoning = "Clean API, good performance"
        else:
            choice = "qiskit"
            confidence = 0.65
            reasoning = "More mature ecosystem"

        votes.append({
            "agent": agent.name,
            "choice": choice,
            "confidence": confidence,
            "reasoning": reasoning,
        })

    # Tally votes
    vote_counts: dict[str, float] = {}
    for v in votes:
        vote_counts[v["choice"]] = vote_counts.get(v["choice"], 0) + v["confidence"]

    winner = max(vote_counts, key=lambda k: vote_counts[k])
    strength = vote_counts[winner] / sum(vote_counts.values())

    duration_ms = (time.time() - start) * 1000

    print_result(f"-> Decision: {winner}", strength=f"{strength:.2f}")
    print_result(f"Switch completed in {duration_ms:.1f}ms")

    return {
        "topology": "swarm",
        "decision": winner,
        "strength": strength,
        "duration_ms": duration_ms,
        "votes": votes,
    }


async def phase_2_pipeline(engine: TopologyEngine, agents: list[MockAgent]) -> dict[str, Any]:
    """Phase 2: Pipeline topology for sequential workflow."""
    print_phase(2, "PIPELINE", "PIPELINE", "Sequential research -> code -> review workflow")

    start = time.time()

    # Switch to pipeline
    old_type = engine.current_topology.topology_type.value if engine.current_topology else "none"
    topology = engine.create_topology(TopologyType.PIPELINE)

    # Add agents as pipeline stages
    researcher = next(a for a in agents if a.archetype == "researcher")
    coder = next(a for a in agents if a.archetype == "coder")
    reviewer = next(a for a in agents if a.archetype == "reviewer")

    topology.add_agent(researcher.agent_id, researcher.name, researcher.capabilities, stage=0)
    topology.add_agent(coder.agent_id, coder.name, coder.capabilities, stage=1)
    topology.add_agent(reviewer.agent_id, reviewer.name, reviewer.capabilities, stage=2)

    switch_ms = (time.time() - start) * 1000
    print_result(f"-> Switched from {old_type} to pipeline in {switch_ms:.1f}ms")

    results = {}

    # Stage 1: Research
    print_agent_action("Researcher", "Gathering information on quantum frameworks...")
    research_result = await researcher.work("Research quantum computing frameworks")
    results["research"] = research_result
    print_agent_action("Researcher", f"Generated {len(research_result['questions'])} research questions")
    print_agent_action("Researcher", f"Found {len(research_result['sources'])} authoritative sources")

    # Stage 2: Code
    print_agent_action("Coder", "Implementing quantum circuit based on research...")
    code_result = await coder.work("Implement quantum circuit")
    results["code"] = code_result
    print_agent_action("Coder", f"Generated {code_result['lines_written']} lines of code")
    print_agent_action("Coder", f"Created {len(code_result['files_created'])} files")

    # Stage 3: Review
    print_agent_action("Reviewer", "Reviewing implementation for quality...")
    review_result = await reviewer.work("Review quantum implementation")
    results["review"] = review_result
    critical = len([i for i in review_result["issues_found"] if i["severity"] == "critical"])
    print_agent_action("Reviewer", f"Found {critical} critical issues")

    duration_ms = (time.time() - start) * 1000
    print_result(f"Pipeline completed in {duration_ms:.1f}ms")

    return {
        "topology": "pipeline",
        "stages_completed": 3,
        "results": results,
        "duration_ms": duration_ms,
    }


async def phase_3_hierarchy(engine: TopologyEngine, agents: list[MockAgent]) -> dict[str, Any]:
    """Phase 3: Hierarchy topology for expert review."""
    print_phase(3, "HIERARCHY", "HIERARCHY", "Expert reviews with authority structure")

    start = time.time()

    # Switch to hierarchy with reviewer as root
    old_type = engine.current_topology.topology_type.value if engine.current_topology else "none"
    topology = engine.create_topology(TopologyType.HIERARCHY)

    reviewer = next(a for a in agents if a.archetype == "reviewer")
    coder = next(a for a in agents if a.archetype == "coder")
    researcher = next(a for a in agents if a.archetype == "researcher")

    # Reviewer is the root (lead)
    topology.add_agent(reviewer.agent_id, reviewer.name, reviewer.capabilities)
    # Others report to reviewer
    topology.add_agent(coder.agent_id, coder.name, coder.capabilities, parent_id=reviewer.agent_id)
    topology.add_agent(researcher.agent_id, researcher.name, researcher.capabilities, parent_id=reviewer.agent_id)

    switch_ms = (time.time() - start) * 1000
    print_result(f"-> Switched from {old_type} to hierarchy in {switch_ms:.1f}ms")
    print_result(f"-> Root: {reviewer.name} (expert reviewer)")

    # Expert review process
    print_agent_action("Reviewer", "Conducting expert review as team lead...")
    await asyncio.sleep(0.1)

    review_findings = {
        "architecture": {
            "score": 8,
            "notes": "Clean separation of concerns, good abstraction layers"
        },
        "correctness": {
            "score": 7,
            "notes": "Minor issues with edge cases in measurement"
        },
        "performance": {
            "score": 9,
            "notes": "Efficient use of numpy for matrix operations"
        },
        "maintainability": {
            "score": 8,
            "notes": "Good code organization, could use more comments"
        },
    }

    overall_score = sum(f["score"] for f in review_findings.values()) / len(review_findings)

    print_agent_action("Reviewer", f"Architecture score: {review_findings['architecture']['score']}/10")
    print_agent_action("Reviewer", f"Correctness score: {review_findings['correctness']['score']}/10")
    print_agent_action("Reviewer", f"Overall score: {overall_score:.1f}/10")

    # Subordinates acknowledge
    print_agent_action("Coder", "Acknowledged feedback, will address critical issues")
    print_agent_action("Researcher", "Will gather additional references for edge cases")

    duration_ms = (time.time() - start) * 1000
    print_result(f"Hierarchy review completed in {duration_ms:.1f}ms")

    return {
        "topology": "hierarchy",
        "root": reviewer.name,
        "findings": review_findings,
        "overall_score": overall_score,
        "duration_ms": duration_ms,
    }


async def phase_4_voting(agents: list[MockAgent]) -> VotingResult:
    """Phase 4: Multi-expert weighted voting on code approval."""
    print_phase(4, "VOTING", "CONSENSUS", "Multi-expert weighted voting on approval")

    # Create voting session
    session = VotingSession(
        question="Should we approve the quantum circuit implementation for production?",
        choices=["APPROVE", "REQUEST_CHANGES", "REJECT"],
        strategy=VotingStrategy.WEIGHTED,
        min_votes=len(agents),
    )

    # Agents cast votes with weights based on expertise
    weights = {
        "researcher": 1.0,
        "coder": 1.2,  # Slightly higher weight for code decisions
        "reviewer": 1.5,  # Highest weight as the expert
    }

    for agent in agents:
        weight = weights.get(agent.archetype, 1.0)

        # Simulate agent decision
        if agent.archetype == "reviewer":
            choice = "REQUEST_CHANGES"
            confidence = 0.85
            reasoning = "Critical issues must be addressed before approval"
        elif agent.archetype == "coder":
            choice = "APPROVE"
            confidence = 0.70
            reasoning = "Core functionality is solid, issues are minor"
        else:
            choice = "APPROVE"
            confidence = 0.75
            reasoning = "Research requirements are met"

        vote = Vote(
            agent_id=agent.agent_id,
            agent_name=agent.name,
            choice=choice,
            confidence=confidence,
            weight=weight,
            reasoning=reasoning,
        )
        session.cast_vote(vote)

        print_agent_action(agent.name, f"Voted {choice} (confidence={confidence:.2f}, weight={weight:.1f})")

    # Tally results
    result = session.tally()

    if console:
        # Create results table
        table = Table(title="Voting Results", box=box.ROUNDED)
        table.add_column("Choice", style="cyan")
        table.add_column("Weighted Votes", justify="right", style="green")

        for choice, count in result.vote_counts.items():
            table.add_row(choice, f"{count:.2f}")

        console.print(table)
    else:
        print("\n   Voting Results:")
        for choice, count in result.vote_counts.items():
            print(f"   {choice}: {count:.2f}")

    print_result(f"-> Winner: {result.winner}")
    print_result(f"-> Consensus: {result.consensus_reached}",
                 strength=f"{result.consensus_strength:.0%}")

    return result


async def phase_5_retrospective(engine: TopologyEngine, agents: list[MockAgent]) -> dict[str, Any]:
    """Phase 5: Swarm retrospective for lessons learned."""
    print_phase(5, "RETRO", "RETROSPECTIVE", "Swarm discussion for lessons learned")

    start = time.time()

    # Switch back to swarm for open discussion
    old_type = engine.current_topology.topology_type.value if engine.current_topology else "none"
    topology = engine.create_topology(TopologyType.SWARM)

    for agent in agents:
        topology.add_agent(agent.agent_id, agent.name, agent.capabilities)

    switch_ms = (time.time() - start) * 1000
    print_result(f"-> Switched from {old_type} to swarm in {switch_ms:.1f}ms")

    # Collect insights from each agent
    lessons = []

    print_agent_action("Researcher", "Pipeline was efficient for research-code-review flow")
    lessons.append("Pipeline topology efficient for sequential workflows")

    print_agent_action("Coder", "Hierarchy helped clarify decision authority")
    lessons.append("Hierarchy clarifies accountability for final decisions")

    print_agent_action("Reviewer", "Weighted voting gave appropriate influence to experts")
    lessons.append("Weighted voting ensures expertise is respected")

    # Consensus on learnings
    print_result(f"-> Team agreed on {len(lessons)} key lessons")

    duration_ms = (time.time() - start) * 1000

    return {
        "topology": "swarm",
        "lessons": lessons,
        "duration_ms": duration_ms,
    }


async def phase_6_learning(learner: EpisodicLearner, demo_results: dict[str, Any]) -> Episode:
    """Phase 6: Record the episode for future learning."""
    print_phase(6, "LEARN", "LEARNING", "Recording episode for future reference")

    # Create task profile
    profile = TaskProfile(
        requires_consensus=True,
        has_sequential_stages=True,
        needs_fault_tolerance=False,
        has_clear_leader=True,
        is_voting_based=True,
        complexity="medium",
        estimated_agents=3,
    )

    # Start episode
    episode = learner.start_episode(
        task_description="Implement quantum computing framework with multi-agent review",
        selected_topology=TopologyType.PIPELINE,  # Primary topology used
        task_profile=profile,
        agent_count=3,
    )

    print_result(f"-> Episode started: {episode.episode_id[:12]}...")

    # Calculate total duration
    total_duration = 0.0
    for key, val in demo_results.items():
        if isinstance(val, dict) and "duration_ms" in val:
            total_duration += val["duration_ms"]

    # Record outcome
    outcome = EpisodeOutcome(
        success=True,
        completion_time_ms=total_duration,
        agent_utilization=0.85,
        communication_overhead=12.0,  # Messages per agent
        topology_switches=4,  # swarm -> pipeline -> hierarchy -> swarm
        error_rate=0.0,
        user_feedback=0.8,  # Positive feedback
        notes="Multi-topology workflow successful for complex implementation task",
    )

    learner.end_episode(outcome)

    print_result(f"-> Episode recorded with score: {outcome.score:.2f}")
    print_result("-> Lessons: pipeline efficient for research-code-review")

    return episode


async def phase_7_retrieval(learner: EpisodicLearner) -> dict[str, Any]:
    """Phase 7: Retrieve learning for future similar tasks."""
    print_phase(7, "QUERY", "RETRIEVAL", "Using learned knowledge for future decisions")

    # Query for similar tasks
    similar = learner.find_similar_episodes(
        task_description="Build a machine learning pipeline with code review",
        limit=3,
    )

    print_result(f"-> Found {len(similar)} similar past episodes")

    # Get recommendation for new task
    new_profile = TaskProfile(
        requires_consensus=True,
        has_sequential_stages=True,
        needs_fault_tolerance=False,
        has_clear_leader=True,
        is_voting_based=False,
        complexity="medium",
        estimated_agents=3,
    )

    recommended, confidence = learner.get_recommendation(new_profile)

    print_result(f"-> Recommended topology: {recommended.value}", confidence=f"{confidence:.2f}")

    # Show statistics
    stats = learner.get_statistics()

    if console:
        table = Table(title="Learning Statistics", box=box.ROUNDED)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", justify="right", style="green")

        table.add_row("Total Episodes", str(stats.get("total_episodes", 0)))
        table.add_row("Avg Score", f"{stats.get('average_score', 0):.2f}")
        table.add_row("Success Rate", f"{stats.get('success_rate', 0):.0%}")

        console.print(table)
    else:
        print("\n   Learning Statistics:")
        print(f"   Total Episodes: {stats.get('total_episodes', 0)}")
        print(f"   Avg Score: {stats.get('average_score', 0):.2f}")
        print(f"   Success Rate: {stats.get('success_rate', 0):.0%}")

    return {
        "similar_episodes": len(similar),
        "recommended_topology": recommended.value,
        "confidence": confidence,
        "statistics": stats,
    }


async def run_demo() -> dict[str, Any]:
    """Run the complete end-to-end demo."""
    print_header("Agentic Titan - End-to-End Demo")

    if console:
        console.print("[dim]Showcasing topology switching, voting, and episodic learning[/dim]\n")
    else:
        print("Showcasing topology switching, voting, and episodic learning\n")

    # Initialize components
    engine = TopologyEngine()
    learner = EpisodicLearner(learning_rate=0.1)

    # Create mock agents
    agents = [
        MockAgent("agent_001", "Researcher", "researcher", ["web_search", "summarization"]),
        MockAgent("agent_002", "Coder", "coder", ["code_generation", "execution"]),
        MockAgent("agent_003", "Reviewer", "reviewer", ["code_review", "quality_assessment"]),
    ]

    results: dict[str, Any] = {}

    # Run all phases
    results["phase_1"] = await phase_1_swarm(engine, agents)
    results["phase_2"] = await phase_2_pipeline(engine, agents)
    results["phase_3"] = await phase_3_hierarchy(engine, agents)
    results["phase_4"] = await phase_4_voting(agents)
    results["phase_5"] = await phase_5_retrospective(engine, agents)
    results["phase_6"] = await phase_6_learning(learner, results)
    results["phase_7"] = await phase_7_retrieval(learner)

    # Summary
    if console:
        console.print("\n")
        console.print(Panel(
            "[bold green]Demo completed successfully![/bold green]\n\n"
            "Demonstrated:\n"
            "  * Dynamic topology switching (Swarm -> Pipeline -> Hierarchy -> Swarm)\n"
            "  * Sequential agent workflows with pipeline topology\n"
            "  * Authority-based review with hierarchy topology\n"
            "  * Weighted voting for consensus decisions\n"
            "  * Episodic learning for topology optimization\n"
            "  * Knowledge retrieval for future tasks",
            title="Demo Summary",
            border_style="green",
        ))
    else:
        print("\n" + "=" * 60)
        print("Demo completed successfully!")
        print("=" * 60)

    return results


def main() -> None:
    """Main entry point."""
    try:
        asyncio.run(run_demo())
    except KeyboardInterrupt:
        if console:
            console.print("\n[yellow]Demo interrupted by user[/yellow]")
        else:
            print("\nDemo interrupted by user")


if __name__ == "__main__":
    main()
