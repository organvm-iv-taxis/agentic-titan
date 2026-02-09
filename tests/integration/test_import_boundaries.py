"""Import-boundary fitness tests for core modules."""

# mypy: disable-error-code="untyped-decorator"

from __future__ import annotations

import importlib
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

# Optional dependency import roots from pyproject optional-dependencies.
OPTIONAL_TIER_IMPORT_ROOTS = [
    "accelerate",
    "aiofiles",
    "aiohttp",
    "aiosqlite",
    "alembic",
    "asyncpg",
    "celery",
    "chromadb",
    "datasets",
    "deepeval",
    "docx",
    "fastapi",
    "jinja2",
    "langfuse",
    "multipart",
    "nats",
    "ollama",
    "openpyxl",
    "passlib",
    "peft",
    "pptx",
    "prometheus_client",
    "psutil",
    "pygls",
    "pypdf",
    "ray",
    "sentence_transformers",
    "slowapi",
    "sqlalchemy",
    "trl",
    "uvicorn",
    "wandb",
]

# Required import roots from base dependencies.
CORE_REQUIRED_IMPORT_ROOTS = [
    "anthropic",
    "httpx",
    "openai",
    "pydantic",
    "pydantic_settings",
    "redis",
    "rich",
    "typer",
    "yaml",
]

# Core modules must remain import-safe with optional dependencies blocked.
CORE_MODULES = [
    "titan",
    "titan.spec",
    "titan.core",
    "titan.core.config",
    "titan.core.project_context",
    "titan.workflows",
    "titan.workflows.cognitive_router",
    "titan.workflows.inquiry_config",
    "titan.workflows.inquiry_dag",
    "titan.workflows.quality_gates",
    "titan.workflows.inquiry_prompts",
    "titan.workflows.inquiry_export",
    "titan.workflows.inquiry_engine",
    "titan.orchestration.termination",
    "titan.orchestration.watchdog",
    "titan.safety.gates",
]

# High-signal forbidden roots for selected core modules.
FORBIDDEN_AT_IMPORT_BY_MODULE: dict[str, list[str]] = {
    "titan.core.config": ["fastapi", "celery", "ray", "sqlalchemy"],
    "titan.workflows.cognitive_router": ["fastapi", "celery", "ray", "sqlalchemy"],
    "titan.workflows.inquiry_config": ["fastapi", "sqlalchemy", "celery"],
    "titan.workflows.inquiry_dag": ["fastapi", "celery", "ray", "sqlalchemy"],
    "titan.workflows.inquiry_engine": ["fastapi", "sqlalchemy", "celery", "ray"],
    "titan.workflows.quality_gates": ["fastapi", "sqlalchemy", "celery", "ray"],
}


def _run_python_with_optional_block(
    code: str,
    blocked_roots: list[str] | None = None,
) -> subprocess.CompletedProcess[str]:
    roots = blocked_roots or OPTIONAL_TIER_IMPORT_ROOTS
    script = textwrap.dedent(
        f"""
        import importlib
        import importlib.abc
        import sys

        blocked_roots = set({roots!r})

        class OptionalTierBlocker(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname.split(".", 1)[0] in blocked_roots:
                    raise ImportError(f"blocked optional dependency: {{fullname}}")
                return None

        sys.meta_path.insert(0, OptionalTierBlocker())
        {code}
        """
    )
    project_root = Path(__file__).resolve().parents[2]
    return subprocess.run(
        [sys.executable, "-c", script],
        cwd=project_root,
        capture_output=True,
        text=True,
    )


def _run_import_with_optional_block(
    module_name: str,
    blocked_roots: list[str] | None = None,
) -> subprocess.CompletedProcess[str]:
    import_code = f"importlib.import_module({module_name!r})"
    return _run_python_with_optional_block(import_code, blocked_roots=blocked_roots)


@pytest.mark.parametrize("root_module", CORE_REQUIRED_IMPORT_ROOTS)
def test_core_required_roots_are_importable(root_module: str) -> None:
    """Base dependency roots must be importable in core environments."""
    importlib.import_module(root_module)


def test_optional_blocker_sanity_check() -> None:
    """Sanity check that the blocker reliably rejects blocked roots."""
    proc = _run_python_with_optional_block(
        "importlib.import_module('fastapi')",
        blocked_roots=["fastapi"],
    )
    assert proc.returncode != 0, "sanity check failed: blocked fastapi import succeeded"
    assert "blocked optional dependency: fastapi" in proc.stderr


@pytest.mark.parametrize("module_name", CORE_MODULES)
def test_core_modules_do_not_require_optional_deps_at_import(module_name: str) -> None:
    """Core modules should import without optional-tier dependencies."""
    proc = _run_import_with_optional_block(module_name)
    assert proc.returncode == 0, (
        f"{module_name} imported optional-tier dependencies at import time.\n"
        f"stdout:\n{proc.stdout}\n"
        f"stderr:\n{proc.stderr}"
    )


@pytest.mark.parametrize("module_name,forbidden_roots", FORBIDDEN_AT_IMPORT_BY_MODULE.items())
def test_module_specific_forbidden_roots(module_name: str, forbidden_roots: list[str]) -> None:
    """Selected modules should remain free of critical optional imports."""
    proc = _run_import_with_optional_block(module_name, blocked_roots=forbidden_roots)
    assert proc.returncode == 0, (
        f"{module_name} hit module-specific optional imports.\n"
        f"blocked_roots={forbidden_roots}\n"
        f"stdout:\n{proc.stdout}\n"
        f"stderr:\n{proc.stderr}"
    )
