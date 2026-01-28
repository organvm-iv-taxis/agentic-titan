"""
Titan Prompts - Auto-Prompting, Token Conservation & Prompt Engineering

This package provides advanced prompt engineering capabilities:

- **Token Optimizer**: Context compression, token estimation, budget-aware selection
- **Auto Prompter**: Dynamic prompt adaptation based on complexity and model tier
- **Metrics**: Prompt effectiveness tracking and optimization recommendations
- **Examples**: Few-shot example bank for prompt injection

Based on research from:
- Anthropic's Context Engineering and Building Effective Agents
- Industry reports on token optimization (90% cost reduction potential)
- Best practices for multi-agent systems
"""

from titan.prompts.token_optimizer import (
    CompressionResult,
    CompressionStrategy,
    TokenEstimate,
    TokenOptimizer,
    get_token_optimizer,
)
from titan.prompts.auto_prompt import (
    AdaptedPrompt,
    AutoPrompter,
    ModelTier,
    PromptConfig,
    TaskComplexity,
    get_auto_prompter,
)
from titan.prompts.metrics import (
    AggregatedMetrics,
    MetricAggregation,
    OptimizationRecommendation,
    PromptMetrics,
    PromptTracker,
    get_prompt_tracker,
)
from titan.prompts.examples import (
    Example,
    ExampleBank,
    STAGE_EXAMPLES,
    get_example_bank,
)

__all__ = [
    # Token Optimizer
    "CompressionResult",
    "CompressionStrategy",
    "TokenEstimate",
    "TokenOptimizer",
    "get_token_optimizer",
    # Auto Prompter
    "AdaptedPrompt",
    "AutoPrompter",
    "ModelTier",
    "PromptConfig",
    "TaskComplexity",
    "get_auto_prompter",
    # Metrics
    "AggregatedMetrics",
    "MetricAggregation",
    "OptimizationRecommendation",
    "PromptMetrics",
    "PromptTracker",
    "get_prompt_tracker",
    # Examples
    "Example",
    "ExampleBank",
    "STAGE_EXAMPLES",
    "get_example_bank",
]
