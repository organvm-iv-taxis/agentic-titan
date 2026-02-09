"""
Titan Workflows - Multi-Perspective Collaborative Inquiry System.

This package intentionally uses lazy exports so importing ``titan.workflows``
does not force heavy runtime configuration imports.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS: dict[str, tuple[str, str]] = {
    "InquiryStage": ("titan.workflows.inquiry_config", "InquiryStage"),
    "InquiryWorkflow": ("titan.workflows.inquiry_config", "InquiryWorkflow"),
    "EXPANSIVE_INQUIRY_WORKFLOW": (
        "titan.workflows.inquiry_config",
        "EXPANSIVE_INQUIRY_WORKFLOW",
    ),
    "DEFAULT_WORKFLOWS": ("titan.workflows.inquiry_config", "DEFAULT_WORKFLOWS"),
    "StageResult": ("titan.workflows.inquiry_engine", "StageResult"),
    "InquirySession": ("titan.workflows.inquiry_engine", "InquirySession"),
    "InquiryEngine": ("titan.workflows.inquiry_engine", "InquiryEngine"),
    "InquiryStatus": ("titan.workflows.inquiry_engine", "InquiryStatus"),
    "CognitiveTaskType": ("titan.workflows.cognitive_router", "CognitiveTaskType"),
    "CognitiveRouter": ("titan.workflows.cognitive_router", "CognitiveRouter"),
    "COGNITIVE_MODEL_MAP": ("titan.workflows.cognitive_router", "COGNITIVE_MODEL_MAP"),
    "export_stage_to_markdown": (
        "titan.workflows.inquiry_export",
        "export_stage_to_markdown",
    ),
    "export_session_to_markdown": (
        "titan.workflows.inquiry_export",
        "export_session_to_markdown",
    ),
    "STAGE_PROMPTS": ("titan.workflows.inquiry_prompts", "STAGE_PROMPTS"),
}

__all__ = list(_EXPORTS.keys())


def __getattr__(name: str) -> Any:
    """Resolve package exports lazily to keep import surface lightweight."""
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attr_name = _EXPORTS[name]
    module = import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Expose lazy exports in interactive environments."""
    return sorted(set(globals()) | set(__all__))
