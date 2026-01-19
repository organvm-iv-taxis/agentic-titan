"""
Agentic Titan CLI - Command-line interface for the agent swarm.

Commands:
- titan run <spec>     - Run an agent from a spec file
- titan swarm <task>   - Start a swarm for a task
- titan status         - Show swarm status
- titan list           - List available agents
- titan health         - Health check all services

Inspired by: kimi-cli mode-switching patterns
"""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from agents.personas import say, ORCHESTRATOR, report_success, report_error
from hive.memory import HiveMind, MemoryConfig
from hive.topology import TopologyEngine, TopologyType
from titan.spec import AgentSpec, SpecRegistry, get_spec_registry
from adapters.router import LLMRouter, RoutingStrategy, get_router

# Initialize
app = typer.Typer(
    name="titan",
    help="Agentic Titan - Polymorphic Agent Swarm",
    add_completion=False,
)
console = Console()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("titan.cli")


# ============================================================================
# Helper Functions
# ============================================================================


def print_banner() -> None:
    """Print the Titan banner."""
    banner = """
    ╔═══════════════════════════════════════════════════════════╗
    ║                                                           ║
    ║     █████╗  ██████╗ ███████╗███╗   ██╗████████╗██╗ ██████╗║
    ║    ██╔══██╗██╔════╝ ██╔════╝████╗  ██║╚══██╔══╝██║██╔════╝║
    ║    ███████║██║  ███╗█████╗  ██╔██╗ ██║   ██║   ██║██║     ║
    ║    ██╔══██║██║   ██║██╔══╝  ██║╚██╗██║   ██║   ██║██║     ║
    ║    ██║  ██║╚██████╔╝███████╗██║ ╚████║   ██║   ██║╚██████╗║
    ║    ╚═╝  ╚═╝ ╚═════╝ ╚══════╝╚═╝  ╚═══╝   ╚═╝   ╚═╝ ╚═════╝║
    ║                                                           ║
    ║          ████████╗██╗████████╗ █████╗ ███╗   ██╗          ║
    ║          ╚══██╔══╝██║╚══██╔══╝██╔══██╗████╗  ██║          ║
    ║             ██║   ██║   ██║   ███████║██╔██╗ ██║          ║
    ║             ██║   ██║   ██║   ██╔══██║██║╚██╗██║          ║
    ║             ██║   ██║   ██║   ██║  ██║██║ ╚████║          ║
    ║             ╚═╝   ╚═╝   ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═══╝          ║
    ║                                                           ║
    ║        Polymorphic Agent Swarm Architecture v0.1.0        ║
    ╚═══════════════════════════════════════════════════════════╝
    """
    console.print(banner, style="cyan")


async def check_infrastructure() -> dict[str, bool]:
    """Check infrastructure services."""
    import httpx

    checks = {
        "redis": False,
        "chromadb": False,
        "nats": False,
        "ollama": False,
    }

    async with httpx.AsyncClient() as client:
        # Redis
        try:
            import redis.asyncio as redis_lib

            r = redis_lib.from_url("redis://localhost:6379")
            await r.ping()
            checks["redis"] = True
            await r.close()
        except Exception:
            pass

        # ChromaDB
        try:
            resp = await client.get("http://localhost:8000/api/v1/heartbeat", timeout=2)
            checks["chromadb"] = resp.status_code == 200
        except Exception:
            pass

        # NATS
        try:
            resp = await client.get("http://localhost:8222/healthz", timeout=2)
            checks["nats"] = resp.status_code == 200
        except Exception:
            pass

        # Ollama
        try:
            resp = await client.get("http://localhost:11434/api/tags", timeout=2)
            checks["ollama"] = resp.status_code == 200
        except Exception:
            pass

    return checks


# ============================================================================
# Commands
# ============================================================================


@app.command()
def run(
    spec_path: str = typer.Argument(..., help="Path to agent spec file"),
    prompt: str = typer.Option(None, "--prompt", "-p", help="Task prompt"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
) -> None:
    """Run an agent from a spec file."""
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    say(ORCHESTRATOR, f"Loading spec from {spec_path}")

    async def run_agent() -> None:
        spec = AgentSpec.from_file(spec_path)
        say(ORCHESTRATOR, f"Loaded agent: {spec.name}")

        console.print(Panel(
            f"[green]Agent loaded successfully![/green]\n\n"
            f"Name: {spec.name}\n"
            f"Capabilities: {', '.join(spec.capabilities)}\n"
            f"LLM: {spec.llm.get('preferred', 'default')}",
            title="Agent Spec",
        ))

        if not prompt:
            say(ORCHESTRATOR, "No prompt provided. Use --prompt to run a task.")
            return

        say(ORCHESTRATOR, f"Task: {prompt}")

        # Create agent based on spec type
        from agents.archetypes import (
            ResearcherAgent,
            CoderAgent,
            ReviewerAgent,
            OrchestratorAgent,
        )

        # Map spec names to agent classes
        agent_map = {
            "researcher": ResearcherAgent,
            "coder": CoderAgent,
            "reviewer": ReviewerAgent,
            "orchestrator": OrchestratorAgent,
        }

        agent_class = agent_map.get(spec.name.lower())
        if not agent_class:
            report_error(ORCHESTRATOR, Exception(f"Unknown agent type: {spec.name}"), "")
            return

        # Initialize Hive Mind for agent
        hive = HiveMind()
        await hive.initialize()

        try:
            # Create agent with appropriate kwargs
            if spec.name.lower() == "researcher":
                agent = agent_class(topic=prompt, hive_mind=hive)
            elif spec.name.lower() == "coder":
                agent = agent_class(task_description=prompt, hive_mind=hive)
            elif spec.name.lower() == "reviewer":
                agent = agent_class(content=prompt, hive_mind=hive)
            elif spec.name.lower() == "orchestrator":
                agent = agent_class(task=prompt, hive_mind=hive)
            else:
                agent = agent_class(hive_mind=hive)

            # Run agent
            say(ORCHESTRATOR, "Starting agent...")
            result = await agent.run(prompt)

            # Display result
            console.print(Panel(
                f"[green]Agent completed![/green]\n\n"
                f"Status: {result.state.value}\n"
                f"Turns: {result.turns_taken}\n"
                f"Duration: {result.execution_time_ms / 1000:.2f}s",
                title="Result",
            ))

            if result.result:
                output_str = str(result.result)
                console.print(Panel(
                    output_str[:2000] + ("..." if len(output_str) > 2000 else ""),
                    title="Output",
                ))

        finally:
            await hive.shutdown()

    try:
        asyncio.run(run_agent())
    except Exception as e:
        report_error(ORCHESTRATOR, e, "Failed to run agent")
        raise typer.Exit(1)


@app.command()
def swarm(
    task: str = typer.Argument(..., help="Task for the swarm"),
    topology: str = typer.Option("auto", "--topology", "-t", help="Topology type"),
    agents: int = typer.Option(3, "--agents", "-a", help="Number of agents"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
) -> None:
    """Start an agent swarm for a task."""
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    print_banner()

    async def run_swarm() -> None:
        say(ORCHESTRATOR, "Initializing swarm...")

        # Initialize components
        hive = HiveMind()
        await hive.initialize()

        engine = TopologyEngine(hive)

        # Select topology
        if topology == "auto":
            suggestion = engine.suggest_topology(task)
            selected = suggestion["recommended"]
            say(ORCHESTRATOR, f"Auto-selected topology: {selected}")
            console.print(f"Reasons: {', '.join(suggestion['reasons'])}")
        else:
            selected = topology

        # Create topology
        topo = engine.create_topology(selected)
        say(ORCHESTRATOR, f"Created {selected} topology")

        # Check LLM availability
        router = get_router()
        await router.initialize()
        providers = router.list_available_providers()

        console.print(Panel(
            f"Task: {task}\n"
            f"Topology: {selected}\n"
            f"Agents: {agents}\n"
            f"LLM Providers: {', '.join(p.value for p in providers)}",
            title="Swarm Configuration",
        ))

        # TODO: Create and run agents
        say(ORCHESTRATOR, "Swarm ready (implementation pending)")

        await hive.shutdown()

    asyncio.run(run_swarm())


@app.command()
def status() -> None:
    """Show swarm status."""
    print_banner()

    async def show_status() -> None:
        checks = await check_infrastructure()

        table = Table(title="Infrastructure Status")
        table.add_column("Service", style="cyan")
        table.add_column("Status", justify="center")
        table.add_column("URL")

        services = [
            ("Redis", checks["redis"], "localhost:6379"),
            ("ChromaDB", checks["chromadb"], "localhost:8000"),
            ("NATS", checks["nats"], "localhost:4222"),
            ("Ollama", checks["ollama"], "localhost:11434"),
        ]

        for name, available, url in services:
            status = "[green]●[/green]" if available else "[red]○[/red]"
            table.add_row(name, status, url)

        console.print(table)

        # LLM providers
        router = get_router()
        await router.initialize()

        table2 = Table(title="LLM Providers")
        table2.add_column("Provider", style="cyan")
        table2.add_column("Status", justify="center")
        table2.add_column("Models")

        for info in router.list_providers():
            status = "[green]●[/green]" if info.available else "[red]○[/red]"
            models = ", ".join(info.models[:3]) + ("..." if len(info.models) > 3 else "")
            table2.add_row(info.provider.value, status, models or "-")

        console.print(table2)

    asyncio.run(show_status())


@app.command("list")
def list_agents(
    directory: str = typer.Option("./specs", "--dir", "-d", help="Specs directory"),
) -> None:
    """List available agent specs."""
    registry = get_spec_registry()

    specs_dir = Path(directory)
    if specs_dir.exists():
        registry.load_directory(specs_dir)

    if not registry:
        console.print("[yellow]No agent specs found[/yellow]")
        console.print(f"Create specs in {directory}/*.titan.yaml")
        return

    table = Table(title="Available Agents")
    table.add_column("ID", style="cyan")
    table.add_column("Name")
    table.add_column("Capabilities")
    table.add_column("LLM")

    for spec in registry.list():
        caps = ", ".join(spec.capabilities[:3])
        if len(spec.capabilities) > 3:
            caps += "..."
        table.add_row(
            spec.id,
            spec.name,
            caps,
            spec.llm.get("preferred", "default"),
        )

    console.print(table)


@app.command()
def health() -> None:
    """Health check all services."""
    async def run_health() -> None:
        say(ORCHESTRATOR, "Running health checks...")

        # Infrastructure
        infra = await check_infrastructure()

        # Hive Mind
        hive = HiveMind()
        try:
            await hive.initialize()
            hive_health = await hive.health_check()
            await hive.shutdown()
        except Exception as e:
            hive_health = {"error": str(e)}

        # LLM Router
        router = get_router()
        try:
            await router.initialize()
            llm_health = await router.health_check()
        except Exception as e:
            llm_health = {"error": str(e)}

        all_healthy = all(infra.values()) and all(
            v for k, v in llm_health.items() if k != "error"
        )

        if all_healthy:
            report_success(ORCHESTRATOR, "All systems operational")
        else:
            report_error(ORCHESTRATOR, "Some systems unhealthy", "Health check")
            console.print(f"Infrastructure: {infra}")
            console.print(f"Hive Mind: {hive_health}")
            console.print(f"LLM: {llm_health}")

    asyncio.run(run_health())


@app.command()
def init(
    directory: str = typer.Argument(".", help="Directory to initialize"),
) -> None:
    """Initialize a new Titan project."""
    dir_path = Path(directory)
    dir_path.mkdir(parents=True, exist_ok=True)

    # Create directories
    (dir_path / "specs").mkdir(exist_ok=True)
    (dir_path / "agents").mkdir(exist_ok=True)

    # Create example spec
    example_spec = '''apiVersion: titan/v1
kind: Agent
metadata:
  name: researcher
  labels:
    tier: cognitive
    domain: knowledge
spec:
  capabilities:
    - web_search
    - summarization
    - research

  personality:
    traits: [thorough, curious, skeptical]
    communication_style: academic

  llm:
    preferred: claude-sonnet
    min_context: 16000
    tools_required: false

  tools:
    - name: web_search
      protocol: native
      module: titan.tools.search

  memory:
    short_term: 10
    long_term: hive_mind

  maxTurns: 20
  timeoutMs: 300000
'''

    spec_file = dir_path / "specs" / "researcher.titan.yaml"
    if not spec_file.exists():
        spec_file.write_text(example_spec)

    console.print(f"[green]Initialized Titan project in {directory}[/green]")
    console.print(f"Created: specs/researcher.titan.yaml")
    console.print("\nNext steps:")
    console.print("  1. Start infrastructure: docker compose up -d redis chromadb")
    console.print("  2. Check status: titan status")
    console.print("  3. Run an agent: titan run specs/researcher.titan.yaml")


@app.command()
def topology(
    task: str = typer.Argument(..., help="Task to analyze"),
) -> None:
    """Suggest a topology for a task."""
    engine = TopologyEngine()
    suggestion = engine.suggest_topology(task)

    console.print(Panel(
        f"[bold]Recommended: {suggestion['recommended']}[/bold]\n\n"
        f"Reasons:\n" + "\n".join(f"  • {r}" for r in suggestion["reasons"]) + "\n\n"
        f"Task Profile:\n" + "\n".join(
            f"  • {k}: {v}" for k, v in suggestion["profile"].items()
        ),
        title="Topology Suggestion",
    ))


@app.command()
def runtime(
    action: str = typer.Argument("status", help="Action: status, suggest, spawn"),
    task: str = typer.Option(None, "--task", "-t", help="Task for suggestion"),
    spec: str = typer.Option(None, "--spec", "-s", help="Agent spec file for spawn"),
    runtime_type: str = typer.Option(None, "--runtime", "-r", help="Force runtime type"),
) -> None:
    """Manage runtime environments."""

    async def run_runtime() -> None:
        from runtime import RuntimeSelector, RuntimeConstraints, RuntimeType

        selector = RuntimeSelector()
        await selector.initialize()

        try:
            if action == "status":
                # Show runtime health
                health = await selector.health_check()
                console.print(Panel(
                    f"[bold]Strategy:[/bold] {health['strategy']}\n"
                    f"[bold]Initialized:[/bold] {health['initialized']}\n\n"
                    f"[bold]Runtimes:[/bold]",
                    title="Runtime Status",
                ))

                for rt_name, rt_health in health["runtimes"].items():
                    status = "[green]✓[/green]" if rt_health.get("initialized") else "[red]✗[/red]"
                    console.print(f"  {status} {rt_name}: {rt_health}")

            elif action == "suggest":
                # Suggest runtime for constraints
                constraints = RuntimeConstraints()
                if task:
                    # Analyze task for constraints
                    task_lower = task.lower()
                    if "gpu" in task_lower or "model" in task_lower:
                        constraints.requires_gpu = True
                    if "scale" in task_lower or "parallel" in task_lower:
                        constraints.expected_instances = 5
                        constraints.auto_scale = True
                    if "isolated" in task_lower or "sandbox" in task_lower:
                        constraints.needs_isolation = True

                suggestion = selector.suggest(constraints)
                console.print(Panel(
                    f"[bold]Recommended:[/bold] {suggestion['recommended']}\n"
                    f"[bold]Score:[/bold] {suggestion['score']:.1f}\n\n"
                    f"[bold]Reasons:[/bold]\n" +
                    "\n".join(f"  • {r}" for r in suggestion["reasons"]) + "\n\n"
                    f"[bold]Alternatives:[/bold]\n" +
                    "\n".join(
                        f"  • {a['type']}: {a['score']:.1f}"
                        for a in suggestion["alternatives"]
                    ),
                    title="Runtime Suggestion",
                ))

            elif action == "spawn":
                if not spec:
                    report_error(ORCHESTRATOR, Exception("--spec required"), "")
                    return

                # Load spec and spawn
                agent_spec = AgentSpec.from_file(spec)
                rt = RuntimeType(runtime_type) if runtime_type else None

                say(ORCHESTRATOR, f"Spawning {agent_spec.name} on {rt or 'auto-selected'} runtime")
                process = await selector.spawn(
                    agent_id=agent_spec.name,
                    agent_spec=agent_spec.to_dict(),
                    prompt=task,
                    runtime_type=rt,
                )

                console.print(Panel(
                    f"[bold]Process ID:[/bold] {process.process_id}\n"
                    f"[bold]Agent:[/bold] {process.agent_id}\n"
                    f"[bold]Runtime:[/bold] {process.runtime_type.value}\n"
                    f"[bold]State:[/bold] {process.state.value}",
                    title="Agent Spawned",
                ))

            else:
                console.print(f"[red]Unknown action: {action}[/red]")
                console.print("Available: status, suggest, spawn")

        finally:
            await selector.shutdown()

    asyncio.run(run_runtime())


@app.command()
def analyze(
    task: str = typer.Argument(..., help="Task description to analyze"),
    use_llm: bool = typer.Option(True, "--llm/--no-llm", help="Use LLM for analysis"),
) -> None:
    """Analyze a task with LLM and suggest topology."""

    async def run_analysis() -> None:
        from hive.analyzer import TaskAnalyzer
        from hive.learning import get_episodic_learner
        from hive.events import get_event_bus

        say(ORCHESTRATOR, f"Analyzing task: {task[:50]}...")

        # Set up components
        llm_router = None
        if use_llm:
            try:
                llm_router = get_router()
                await llm_router.initialize()
            except Exception as e:
                console.print(f"[yellow]LLM not available, using keyword analysis: {e}[/yellow]")
                llm_router = None

        analyzer = TaskAnalyzer(llm_router=llm_router, use_llm=use_llm and llm_router is not None)
        learner = get_episodic_learner()
        event_bus = get_event_bus()

        # Create topology engine with all components
        engine = TopologyEngine(
            task_analyzer=analyzer,
            episodic_learner=learner,
            event_bus=event_bus,
        )

        # Analyze
        selected, analysis = await engine.analyze_and_select(task, use_llm=use_llm)

        # Display results
        console.print(Panel(
            f"[bold green]Recommended Topology: {analysis['recommended_topology'].upper()}[/bold green]\n\n"
            f"[bold]Confidence:[/bold] {analysis.get('confidence', 0.5):.0%}\n"
            f"[bold]Reasoning:[/bold] {analysis.get('reasoning', 'N/A')}\n\n"
            f"[bold]Task Profile:[/bold]\n" +
            "\n".join(f"  • {k}: {v}" for k, v in analysis.get('profile', {}).items()),
            title="🔬 Task Analysis",
        ))

        # Show learning stats if available
        stats = learner.get_statistics()
        if stats and stats["total_episodes"] > 0:
            console.print(f"\n[dim]Learning: {stats['total_episodes']} episodes recorded[/dim]")

    asyncio.run(run_analysis())


@app.command()
def learning(
    action: str = typer.Argument("stats", help="Action: stats, clear, export"),
    output: str = typer.Option(None, "--output", "-o", help="Output file for export"),
) -> None:
    """Manage episodic learning system."""
    from hive.learning import get_episodic_learner
    import json

    learner = get_episodic_learner()

    if action == "stats":
        stats = learner.get_statistics()

        # Build stats display
        content = (
            f"[bold]Total Episodes:[/bold] {stats['total_episodes']}\n"
            f"[bold]Completed:[/bold] {stats['completed_episodes']}\n"
            f"[bold]Unique Profiles:[/bold] {stats['unique_profiles']}\n"
            f"[bold]Learning Rate:[/bold] {stats['learning_rate']}\n\n"
            f"[bold]Topology Performance:[/bold]\n"
        )

        for topology, data in stats.get("topology_stats", {}).items():
            content += (
                f"  [cyan]{topology}[/cyan]: "
                f"{data['count']} uses, "
                f"avg score: {data['avg_score']:.2f}, "
                f"success: {data['success_rate']:.0%}\n"
            )

        if not stats.get("topology_stats"):
            content += "  [dim]No episodes recorded yet[/dim]\n"

        console.print(Panel(content, title="📚 Episodic Learning Statistics"))

    elif action == "clear":
        confirm = typer.confirm("Clear all learning data?")
        if confirm:
            learner._episodes.clear()
            learner._preferences.clear()
            learner._save()
            console.print("[green]Learning data cleared[/green]")

    elif action == "export":
        if not output:
            output = ".titan/learning_export.json"

        data = {
            "statistics": learner.get_statistics(),
            "episodes": [e.to_dict() for e in learner._episodes],
            "preferences": {
                k: {t.value: p.to_dict() for t, p in v.items()}
                for k, v in learner._preferences.items()
            },
        }

        Path(output).parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w") as f:
            json.dump(data, f, indent=2)

        console.print(f"[green]Exported learning data to {output}[/green]")

    else:
        console.print(f"[red]Unknown action: {action}[/red]")
        console.print("Available: stats, clear, export")


@app.command()
def events(
    action: str = typer.Argument("history", help="Action: history, clear"),
    event_type: str = typer.Option(None, "--type", "-t", help="Filter by event type"),
    limit: int = typer.Option(20, "--limit", "-n", help="Number of events to show"),
) -> None:
    """View event history."""
    from hive.events import get_event_bus, EventType

    event_bus = get_event_bus()

    if action == "history":
        # Filter by type if specified
        filter_type = None
        if event_type:
            try:
                filter_type = EventType(event_type)
            except ValueError:
                console.print(f"[red]Unknown event type: {event_type}[/red]")
                console.print(f"Available: {', '.join(e.value for e in EventType)}")
                return

        events_list = event_bus.get_history(event_type=filter_type, limit=limit)

        if not events_list:
            console.print("[dim]No events recorded[/dim]")
            return

        table = Table(title=f"Event History (last {len(events_list)})")
        table.add_column("Time", style="dim")
        table.add_column("Type", style="cyan")
        table.add_column("Source")
        table.add_column("Payload", max_width=50)

        for event in events_list:
            payload_str = str(event.payload)[:47] + "..." if len(str(event.payload)) > 50 else str(event.payload)
            table.add_row(
                event.timestamp.strftime("%H:%M:%S"),
                event.event_type.value,
                event.source_id or "-",
                payload_str,
            )

        console.print(table)

    elif action == "clear":
        event_bus.clear_history()
        console.print("[green]Event history cleared[/green]")

    else:
        console.print(f"[red]Unknown action: {action}[/red]")
        console.print("Available: history, clear")


# ============================================================================
# Phase 4: Scale & Polish Commands
# ============================================================================


@app.command()
def stress(
    scenario: str = typer.Argument("swarm", help="Scenario: swarm, pipeline, hierarchy, chaos, scale"),
    agents: int = typer.Option(50, "--agents", "-a", help="Number of agents"),
    duration: int = typer.Option(60, "--duration", "-d", help="Duration in seconds"),
    max_concurrent: int = typer.Option(20, "--concurrent", "-c", help="Max concurrent agents"),
    failure_rate: float = typer.Option(0.0, "--failure-rate", "-f", help="Failure injection rate (0-1)"),
    output: str = typer.Option(None, "--output", "-o", help="Output file for results"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
) -> None:
    """Run stress tests against the agent swarm."""
    from titan.stress import StressTestRunner, StressTestConfig
    from titan.stress.scenarios import list_scenarios

    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Show available scenarios
    if scenario == "list":
        table = Table(title="Available Stress Test Scenarios")
        table.add_column("Name", style="cyan")
        table.add_column("Description")

        for s in list_scenarios():
            table.add_row(s["name"], s["description"])

        console.print(table)
        return

    print_banner()
    say(ORCHESTRATOR, f"Starting stress test: {scenario}")

    async def run_stress() -> None:
        # Initialize HiveMind if available
        hive = None
        try:
            hive = HiveMind()
            await hive.initialize()
        except Exception as e:
            console.print(f"[yellow]HiveMind not available: {e}[/yellow]")

        # Configure test
        config = StressTestConfig(
            scenario_name=scenario,
            target_agents=agents,
            duration_seconds=duration,
            max_concurrent=max_concurrent,
            failure_rate=failure_rate,
            verbose=verbose,
        )

        console.print(Panel(
            f"[bold]Scenario:[/bold] {scenario}\n"
            f"[bold]Target Agents:[/bold] {agents}\n"
            f"[bold]Duration:[/bold] {duration}s\n"
            f"[bold]Max Concurrent:[/bold] {max_concurrent}\n"
            f"[bold]Failure Rate:[/bold] {failure_rate:.0%}",
            title="Stress Test Configuration",
        ))

        # Run test
        runner = StressTestRunner(config, hive_mind=hive)
        result = await runner.run()

        # Display results
        console.print("\n" + result.metrics.summary())

        if output:
            result.save(output)
            console.print(f"\n[green]Results saved to {output}[/green]")

        if hive:
            await hive.shutdown()

    try:
        asyncio.run(run_stress())
    except KeyboardInterrupt:
        say(ORCHESTRATOR, "Stress test interrupted")


@app.command()
def dashboard(
    action: str = typer.Argument("serve", help="Action: serve"),
    port: int = typer.Option(8080, "--port", "-p", help="Port to listen on"),
    host: str = typer.Option("127.0.0.1", "--host", "-h", help="Host to bind to"),
    reload: bool = typer.Option(False, "--reload", "-r", help="Enable auto-reload"),
) -> None:
    """Start the web dashboard."""
    from dashboard.app import run_dashboard

    if action == "serve":
        print_banner()
        say(ORCHESTRATOR, f"Starting dashboard on http://{host}:{port}")
        console.print("\nEndpoints:")
        console.print(f"  Dashboard: http://{host}:{port}")
        console.print(f"  API: http://{host}:{port}/api")
        console.print(f"  WebSocket: ws://{host}:{port}/ws")
        console.print("\nPress Ctrl+C to stop\n")

        run_dashboard(host=host, port=port, reload=reload)

    else:
        console.print(f"[red]Unknown action: {action}[/red]")
        console.print("Available: serve")


@app.command()
def metrics(
    action: str = typer.Argument("serve", help="Action: serve, show"),
    port: int = typer.Option(9100, "--port", "-p", help="Metrics port"),
    host: str = typer.Option("0.0.0.0", "--host", "-h", help="Host to bind to"),
) -> None:
    """Manage Prometheus metrics."""
    from titan.metrics import (
        start_metrics_server,
        get_metrics_text,
        get_metrics,
        PROMETHEUS_AVAILABLE,
    )

    if not PROMETHEUS_AVAILABLE:
        console.print("[red]prometheus_client not installed[/red]")
        console.print("Install with: pip install prometheus-client")
        return

    if action == "serve":
        say(ORCHESTRATOR, f"Starting metrics server on http://{host}:{port}/metrics")
        console.print("\nScrape configuration for Prometheus:")
        console.print(f"  - job_name: 'titan'\n    static_configs:\n      - targets: ['{host}:{port}']")
        console.print("\nPress Ctrl+C to stop\n")

        try:
            start_metrics_server(port=port, host=host)
            # Keep the server running
            import time
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            say(ORCHESTRATOR, "Metrics server stopped")

    elif action == "show":
        # Show current metrics
        metrics_text = get_metrics_text()
        console.print(Panel(metrics_text[:5000], title="Current Metrics"))

    else:
        console.print(f"[red]Unknown action: {action}[/red]")
        console.print("Available: serve, show")


@app.command()
def observe(
    action: str = typer.Argument("start", help="Action: start, stop, status"),
) -> None:
    """Start observability stack (Prometheus + Grafana)."""
    import subprocess
    import os

    deploy_dir = Path(__file__).parent.parent / "deploy"

    if action == "start":
        say(ORCHESTRATOR, "Starting observability stack...")
        try:
            result = subprocess.run(
                ["docker", "compose", "--profile", "monitoring", "up", "-d"],
                cwd=deploy_dir,
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                console.print("[green]Observability stack started![/green]")
                console.print("\nServices:")
                console.print("  Prometheus: http://localhost:9090")
                console.print("  Grafana: http://localhost:3000 (admin/titan)")
            else:
                console.print(f"[red]Failed to start: {result.stderr}[/red]")
        except FileNotFoundError:
            console.print("[red]Docker not found. Please install Docker.[/red]")

    elif action == "stop":
        say(ORCHESTRATOR, "Stopping observability stack...")
        try:
            subprocess.run(
                ["docker", "compose", "--profile", "monitoring", "down"],
                cwd=deploy_dir,
                capture_output=True,
            )
            console.print("[green]Observability stack stopped[/green]")
        except FileNotFoundError:
            console.print("[red]Docker not found[/red]")

    elif action == "status":
        try:
            result = subprocess.run(
                ["docker", "compose", "ps", "--format", "table {{.Name}}\t{{.Status}}\t{{.Ports}}"],
                cwd=deploy_dir,
                capture_output=True,
                text=True,
            )
            console.print(result.stdout)
        except FileNotFoundError:
            console.print("[red]Docker not found[/red]")

    else:
        console.print(f"[red]Unknown action: {action}[/red]")
        console.print("Available: start, stop, status")


# ============================================================================
# MCP Server
# ============================================================================


@app.command()
def mcp(
    action: str = typer.Argument("run", help="Action: run, test"),
) -> None:
    """
    Run the Titan MCP Server.

    This exposes agents via the Model Context Protocol (JSON-RPC over stdio).
    Claude Code and other MCP clients can spawn and manage agents.

    Actions:
        run  - Start the MCP server on stdio
        test - Run a quick self-test
    """
    from mcp.server import run_server, create_server

    if action == "run":
        say(ORCHESTRATOR, "Starting Titan MCP Server...")
        console.print("[cyan]MCP Server ready on stdio[/cyan]")
        console.print("Connect via Claude Code or other MCP clients")
        console.print("")
        console.print("Available tools:")
        console.print("  • spawn_agent - Create agents (researcher, coder, reviewer, orchestrator)")
        console.print("  • agent_status - Check agent progress")
        console.print("  • agent_result - Get completed results")
        console.print("  • list_agents - List active sessions")
        console.print("  • cancel_agent - Cancel running agents")
        console.print("")

        # Run the server
        run_server()

    elif action == "test":
        say(ORCHESTRATOR, "Testing MCP Server...")

        async def run_test() -> None:
            import json
            server = create_server()

            # Test initialize
            from mcp.server import MCPRequest
            init_req = MCPRequest(
                jsonrpc="2.0",
                id=1,
                method="initialize",
                params={"protocolVersion": "2024-11-05"},
            )
            resp = await server.handle_request(init_req)
            console.print(f"[green]✓[/green] Initialize: {resp.result['serverInfo']['name']}")

            # Test tools/list
            tools_req = MCPRequest(
                jsonrpc="2.0",
                id=2,
                method="tools/list",
                params={},
            )
            resp = await server.handle_request(tools_req)
            tool_names = [t["name"] for t in resp.result["tools"]]
            console.print(f"[green]✓[/green] Tools: {', '.join(tool_names)}")

            # Test resources/list
            res_req = MCPRequest(
                jsonrpc="2.0",
                id=3,
                method="resources/list",
                params={},
            )
            resp = await server.handle_request(res_req)
            res_names = [r["name"] for r in resp.result["resources"]]
            console.print(f"[green]✓[/green] Resources: {', '.join(res_names)}")

            # Test spawn_agent (simulated, won't actually run LLM)
            spawn_req = MCPRequest(
                jsonrpc="2.0",
                id=4,
                method="tools/call",
                params={
                    "name": "spawn_agent",
                    "arguments": {
                        "agent_type": "simple",
                        "task": "Test task",
                    },
                },
            )
            resp = await server.handle_request(spawn_req)
            content = json.loads(resp.result["content"][0]["text"])
            console.print(f"[green]✓[/green] Spawn: session {content['session_id']}")

            # Test list_agents
            list_req = MCPRequest(
                jsonrpc="2.0",
                id=5,
                method="tools/call",
                params={
                    "name": "list_agents",
                    "arguments": {},
                },
            )
            resp = await server.handle_request(list_req)
            agents = json.loads(resp.result["content"][0]["text"])
            console.print(f"[green]✓[/green] List: {len(agents)} active agents")

            console.print("")
            console.print("[green]All MCP tests passed![/green]")

        asyncio.run(run_test())

    else:
        console.print(f"[red]Unknown action: {action}[/red]")
        console.print("Available: run, test")


# ============================================================================
# Entry Point
# ============================================================================


def main() -> None:
    """CLI entry point."""
    app()


if __name__ == "__main__":
    main()
