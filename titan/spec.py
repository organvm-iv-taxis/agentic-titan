"""
Agent Spec DSL - YAML-based agent definition language.

Enables portable agent definitions that work across:
- Different LLM providers
- Multiple runtimes (local, container, serverless)
- Various tool protocols (MCP, native)

Inspired by: Anthropic's skills repository YAML DSL
"""

# mypy: disable-error-code="misc,untyped-decorator"

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, cast

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

from agents.framework.errors import SpecValidationError

logger = logging.getLogger("titan.spec")


# ============================================================================
# Spec Models
# ============================================================================


class ToolProtocol(StrEnum):
    """Supported tool protocols."""

    MCP = "mcp"
    NATIVE = "native"
    HTTP = "http"


class RuntimeType(StrEnum):
    """Supported runtime types."""

    LOCAL = "local"
    CONTAINER = "container"
    SERVERLESS = "serverless"


class AgentTier(StrEnum):
    """Agent capability tiers."""

    COGNITIVE = "cognitive"  # Complex reasoning
    OPERATIONAL = "operational"  # Task execution
    SPECIALIZED = "specialized"  # Domain-specific


class LLMPreference(BaseModel):
    """LLM configuration preferences."""

    preferred: str = "claude-sonnet"
    fallback: list[str] = Field(default_factory=list)
    min_context: int = 8000
    tools_required: bool = False


class ToolSpec(BaseModel):
    """Tool specification."""

    name: str
    protocol: ToolProtocol = ToolProtocol.NATIVE
    server: str | None = None  # For MCP tools
    module: str | None = None  # For native tools
    config: dict[str, Any] = Field(default_factory=dict)


class MemorySpec(BaseModel):
    """Memory configuration."""

    short_term: int = 10  # Messages to keep
    long_term: str = "hive_mind"  # Memory backend


class RuntimeSpec(BaseModel):
    """Runtime configuration."""

    local: dict[str, Any] | None = None
    container: dict[str, Any] | None = None
    serverless: dict[str, Any] | None = None


class PersonalitySpec(BaseModel):
    """Agent personality traits."""

    traits: list[str] = Field(default_factory=list)
    communication_style: str = "neutral"


class AgentMetadata(BaseModel):
    """Agent metadata."""

    name: str
    labels: dict[str, str] = Field(default_factory=dict)
    annotations: dict[str, str] = Field(default_factory=dict)


class AgentSpecModel(BaseModel):
    """
    Complete agent specification.

    This is the Pydantic model for validating agent YAML specs.
    """

    api_version: str = Field(alias="apiVersion", default="titan/v1")
    kind: str = "Agent"
    metadata: AgentMetadata
    spec: AgentSpecInner

    model_config = ConfigDict(populate_by_name=True)


class AgentSpecInner(BaseModel):
    """Inner spec containing agent definition."""

    capabilities: list[str] = Field(default_factory=list)
    personality: PersonalitySpec = Field(default_factory=PersonalitySpec)
    llm: LLMPreference = Field(default_factory=LLMPreference)
    tools: list[ToolSpec] = Field(default_factory=list)
    memory: MemorySpec = Field(default_factory=MemorySpec)
    runtimes: RuntimeSpec = Field(default_factory=RuntimeSpec)
    system_prompt: str | None = Field(alias="systemPrompt", default=None)
    max_turns: int = Field(alias="maxTurns", default=20)
    timeout_ms: int = Field(alias="timeoutMs", default=300000)

    @field_validator("capabilities")
    @classmethod
    def validate_capabilities(cls, v: list[str]) -> list[str]:
        valid_caps = {
            "web_search",
            "document_analysis",
            "code_generation",
            "code_review",
            "summarization",
            "data_analysis",
            "planning",
            "execution",
            "research",
            "writing",
        }
        for cap in v:
            if cap not in valid_caps:
                logger.warning(f"Unknown capability: {cap}")
        return v

    model_config = ConfigDict(populate_by_name=True)


# ============================================================================
# Spec Parser
# ============================================================================


@dataclass
class AgentSpec:
    """
    Parsed agent specification.

    This is the runtime representation of an agent spec.
    """

    id: str
    name: str
    labels: dict[str, str]
    capabilities: list[str]
    personality: dict[str, Any]
    llm: dict[str, Any]
    tools: list[dict[str, Any]]
    memory: dict[str, Any]
    runtimes: dict[str, Any]
    system_prompt: str | None
    max_turns: int
    timeout_ms: int
    source_path: str | None = None

    @classmethod
    def from_yaml(cls, yaml_content: str, source_path: str | None = None) -> AgentSpec:
        """
        Parse an agent spec from YAML content.

        Args:
            yaml_content: YAML string
            source_path: Optional source file path

        Returns:
            Parsed AgentSpec

        Raises:
            SpecValidationError: If spec is invalid
        """
        try:
            data = yaml.safe_load(yaml_content)
        except yaml.YAMLError as e:
            raise SpecValidationError(
                source_path or "unknown",
                [f"YAML parse error: {e}"],
            ) from e

        try:
            model = AgentSpecModel(**data)
        except Exception as e:
            raise SpecValidationError(
                source_path or "unknown",
                [str(e)],
            ) from e

        # Generate ID from name
        agent_id = model.metadata.name.lower().replace(" ", "-")

        return cls(
            id=agent_id,
            name=model.metadata.name,
            labels=model.metadata.labels,
            capabilities=model.spec.capabilities,
            personality=model.spec.personality.model_dump(),
            llm=model.spec.llm.model_dump(),
            tools=[t.model_dump() for t in model.spec.tools],
            memory=model.spec.memory.model_dump(),
            runtimes=model.spec.runtimes.model_dump(),
            system_prompt=model.spec.system_prompt,
            max_turns=model.spec.max_turns,
            timeout_ms=model.spec.timeout_ms,
            source_path=source_path,
        )

    @classmethod
    def from_file(cls, path: str | Path) -> AgentSpec:
        """
        Load an agent spec from a file.

        Args:
            path: Path to YAML file

        Returns:
            Parsed AgentSpec
        """
        path = Path(path)
        if not path.exists():
            raise SpecValidationError(str(path), ["File not found"])

        return cls.from_yaml(path.read_text(), str(path))

    def to_yaml(self) -> str:
        """Serialize spec back to YAML."""
        data = {
            "apiVersion": "titan/v1",
            "kind": "Agent",
            "metadata": {
                "name": self.name,
                "labels": self.labels,
            },
            "spec": {
                "capabilities": self.capabilities,
                "personality": self.personality,
                "llm": self.llm,
                "tools": self.tools,
                "memory": self.memory,
                "runtimes": self.runtimes,
                "systemPrompt": self.system_prompt,
                "maxTurns": self.max_turns,
                "timeoutMs": self.timeout_ms,
            },
        }
        return cast(str, yaml.dump(data, default_flow_style=False, sort_keys=False))

    def to_dict(self) -> dict[str, Any]:
        """Serialize spec to dictionary."""
        return {
            "apiVersion": "titan/v1",
            "kind": "Agent",
            "metadata": {
                "name": self.name,
                "labels": self.labels,
            },
            "spec": {
                "capabilities": self.capabilities,
                "personality": self.personality,
                "llm": self.llm,
                "tools": self.tools,
                "memory": self.memory,
                "runtimes": self.runtimes,
                "systemPrompt": self.system_prompt,
                "maxTurns": self.max_turns,
                "timeoutMs": self.timeout_ms,
            },
        }


# ============================================================================
# Spec Registry
# ============================================================================


class SpecRegistry:
    """
    Registry for agent specifications.

    Loads and manages agent specs from files and directories.
    """

    def __init__(self) -> None:
        self._specs: dict[str, AgentSpec] = {}

    def register(self, spec: AgentSpec) -> None:
        """Register an agent spec."""
        self._specs[spec.id] = spec
        logger.info(f"Registered spec: {spec.id}")

    def get(self, agent_id: str) -> AgentSpec | None:
        """Get a spec by ID."""
        return self._specs.get(agent_id)

    def list_specs(self) -> list[AgentSpec]:
        """List all registered specs."""
        return list(self._specs.values())

    def load_file(self, path: str | Path) -> AgentSpec:
        """Load and register a spec from a file."""
        spec = AgentSpec.from_file(path)
        self.register(spec)
        return spec

    def load_directory(self, directory: str | Path) -> list[AgentSpec]:
        """
        Load all specs from a directory.

        Looks for *.titan.yaml and *.agent.yaml files.
        """
        directory = Path(directory)
        if not directory.is_dir():
            raise SpecValidationError(str(directory), ["Not a directory"])

        specs: list[AgentSpec] = []
        patterns = ["*.titan.yaml", "*.agent.yaml", "*.titan.yml", "*.agent.yml"]

        for pattern in patterns:
            for path in directory.glob(pattern):
                try:
                    spec = self.load_file(path)
                    specs.append(spec)
                except SpecValidationError as e:
                    logger.warning(f"Failed to load {path}: {e}")

        logger.info(f"Loaded {len(specs)} specs from {directory}")
        return specs

    def find_by_capability(self, capability: str) -> list[AgentSpec]:
        """Find specs with a specific capability."""
        return [s for s in self._specs.values() if capability in s.capabilities]

    def find_by_label(self, key: str, value: str) -> list[AgentSpec]:
        """Find specs with a specific label."""
        return [s for s in self._specs.values() if s.labels.get(key) == value]

    def list(self) -> list[AgentSpec]:
        """Backward-compatible alias for listing registered specs."""
        return self.list_specs()

    def __len__(self) -> int:
        return len(self._specs)

    def __contains__(self, agent_id: str) -> bool:
        return agent_id in self._specs


# Singleton registry
_default_registry: SpecRegistry | None = None


def get_spec_registry() -> SpecRegistry:
    """Get the default spec registry."""
    global _default_registry
    if _default_registry is None:
        _default_registry = SpecRegistry()
    return _default_registry
