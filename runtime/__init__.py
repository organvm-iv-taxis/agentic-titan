"""
Runtime Fabric - Multi-environment agent deployment.

Supports:
- Local: Direct Python process
- Sandboxed: OS-level isolation (Seatbelt/Landlock)
- Container: Docker/K3s isolated environments
- Serverless: OpenFaaS for burst scaling

The Runtime Selector automatically chooses the best runtime based on:
- GPU requirements
- Scale requirements
- Cost optimization
- Fault tolerance needs
- Security requirements
"""

from runtime.base import (
    Runtime,
    RuntimeType,
    RuntimeConfig,
    RuntimeConstraints,
    AgentProcess,
    ProcessState,
)
from runtime.selector import RuntimeSelector, SelectionStrategy
from runtime.local import LocalRuntime
from runtime.docker import DockerRuntime
from runtime.openfaas import OpenFaaSRuntime
from runtime.sandbox import (
    SandboxedRuntime,
    SandboxConfig,
    SandboxType,
    create_sandboxed_runtime,
)

__all__ = [
    # Base
    "Runtime",
    "RuntimeType",
    "RuntimeConfig",
    "RuntimeConstraints",
    "AgentProcess",
    "ProcessState",
    # Selector
    "RuntimeSelector",
    "SelectionStrategy",
    # Implementations
    "LocalRuntime",
    "DockerRuntime",
    "OpenFaaSRuntime",
    # Sandboxed
    "SandboxedRuntime",
    "SandboxConfig",
    "SandboxType",
    "create_sandboxed_runtime",
]
