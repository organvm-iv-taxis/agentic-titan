"""
Titan Prompts - Few-Shot Example Bank

Provides diverse, high-quality examples for few-shot injection into prompts.

Based on research:
- Anthropic: "Examples are pictures worth a thousand words"
- Use diverse examples over edge case lists
- 2-3 examples per task type is optimal
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Example:
    """A single few-shot example."""

    input: str
    output: str
    task_type: str
    quality_score: float = 1.0  # 0-1, for example selection
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


# =============================================================================
# Inquiry Stage Examples
# =============================================================================

SCOPE_CLARIFICATION_EXAMPLES: list[dict[str, str]] = [
    {
        "input": "What is consciousness?",
        "output": """**Core Restatement**: How does subjective experience emerge from physical processes, and what criteria could distinguish conscious from non-conscious systems?

**Key Dimensions**:
1. The "hard problem" - why there is something it is like to be a conscious entity
2. Neural correlates - what brain processes are necessary and sufficient
3. Measurement challenges - how to detect consciousness in non-communicative systems
4. Boundaries - where consciousness begins and ends (animals, AI, groups)

**Boundary Conditions**: Out of scope: spiritual or religious definitions, altered states specific to substances, legal definitions of consciousness.

**Exploration Potential**: This question sits at the intersection of neuroscience, philosophy, and AI - understanding it could transform medicine, ethics, and technology.""",
    },
    {
        "input": "Why do startups fail?",
        "output": """**Core Restatement**: What systemic and behavioral factors most strongly predict startup failure, and how do these factors interact across different stages and industries?

**Key Dimensions**:
1. Product-market fit - building something people actually want
2. Team dynamics - founder conflicts, hiring decisions, culture
3. Capital efficiency - burn rate, fundraising timing, runway management
4. Market timing - entering too early or too late
5. Execution vs strategy - which matters more at each stage

**Boundary Conditions**: Focusing on technology/venture-backed startups. Excluding family businesses, franchises, and lifestyle businesses.

**Exploration Potential**: Understanding failure patterns could improve capital allocation, founder education, and investment decisions across the ecosystem.""",
    },
]

LOGICAL_BRANCHING_EXAMPLES: list[dict[str, str]] = [
    {
        "input": "Topic: Remote work productivity",
        "output": """**Primary Inquiry Lines**:
1. How does communication frequency correlate with output quality in distributed teams?
2. What role does synchronous vs asynchronous work play in different task types?
3. How do home environments impact cognitive performance?
4. What management practices predict high performance in remote settings?
5. How does remote work affect knowledge transfer and organizational learning?

**Logical Tree**:
1. Communication & Output
   - Why: Does more communication always mean better outcomes? → Not necessarily; interruption costs
   - How: What communication patterns optimize signal-to-noise? → Structured async updates + focused sync
   - What if: Teams eliminated real-time chat? → Slower coordination but deeper work

2. Sync vs Async Work
   - Why: When is real-time collaboration essential? → High-uncertainty creative tasks
   - How: Can complex decisions be made asynchronously? → Yes, with proper documentation norms
   - What if: All meetings became optional? → Self-selection reveals true necessity

**Dependencies**: Understanding communication patterns (#1) is prerequisite to optimizing sync/async balance (#2)

**Logical Tensions**: Individual productivity may conflict with team coordination needs

**Synthesis Points**: All lines converge on the meta-question of measuring knowledge work output""",
    },
]

INTUITIVE_BRANCHING_EXAMPLES: list[dict[str, str]] = [
    {
        "input": "Topic: Artificial Intelligence",
        "output": """**Mythic Framings**:
1. **The Golem** - AI as humanity's created servant that may exceed its master's control
2. **Prometheus** - AI as fire stolen from the gods, bringing both light and destruction
3. **The Oracle** - AI as all-knowing entity we consult but cannot fully comprehend
4. **The Mirror** - AI as reflection showing us who we really are
5. **The Child** - AI as offspring inheriting our values (and flaws)

**Archetypal Patterns**:
- Creator/Creation: The eternal tension between maker and made
- Transformation: The irreversible crossing of a threshold
- The Shadow: Technology revealing humanity's hidden nature
- Rebirth: Death of old human capabilities, birth of new hybrid forms

**Symbolic Stories**:
*The Apprentice's Broom*: We command AI to carry water, faster and faster, until the house floods. The sorcerer (human wisdom) must return before the automation drowns us.

**Hidden Dimensions**: AI anxiety may mask deeper fears about human meaning and mortality

**Emotional Resonance**: We approach AI with the same mix of wonder and dread our ancestors felt at lightning - power beyond understanding""",
    },
]

LATERAL_EXPLORATION_EXAMPLES: list[dict[str, str]] = [
    {
        "input": "Topic: Urban transportation",
        "output": """**Connected Domains**:
1. **Ant colonies** - Distributed, emergent traffic optimization without central control
2. **Blood circulation** - Efficiency of arterial/capillary hierarchies for flow
3. **Internet routing** - Packet switching vs circuit switching parallels
4. **Jazz improvisation** - Coordination without central conductor
5. **Epidemiology** - Disease spread patterns mirror traffic congestion dynamics

**Bridge Analogies**:
- Ant pheromone trails → dynamic road pricing signals
- Blood vessel branching → optimal transit network topology
- TCP/IP congestion control → adaptive traffic signal timing
- Jazz call-and-response → autonomous vehicle negotiation
- Contact tracing → traffic origin-destination studies

**Hybrid Questions**:
- Could cities use "packet switching" - dynamic routing of individual vehicles rather than fixed lanes?
- What if traffic signals worked like immune responses - learning from past "infections" of congestion?
- How would ant colony algorithms redesign highway interchanges?

**Structural Patterns**: All domains feature tension between individual optimization and system efficiency

**Innovation Potential**: Biological systems have solved routing problems over millions of years - their solutions may outperform our engineered approaches""",
    },
]

RECURSIVE_DESIGN_EXAMPLES: list[dict[str, str]] = [
    {
        "input": "Topic: Climate change adaptation",
        "output": """**Stage Analysis**:
- Scope stage revealed the topic is really about coordination failure under uncertainty
- Logic stage exposed tensions between individual and collective rationality
- Mythos stage connected to deep narratives of stewardship vs exploitation
- Lateral stage found parallels to evolutionary adaptation and immune systems

**Feedback Design**:
A recursive inquiry system for climate adaptation should:
1. Track which framings (logical, mythic, lateral) generate most actionable insights
2. Identify when the inquiry is circling vs deepening
3. Surface contradictions between stages as productive tensions
4. Measure distance from original framing - are we exploring or drifting?

**Evolution Mechanisms**:
1. **Pruning**: If logical analysis keeps returning to "collective action problem," stop re-deriving it
2. **Seeding**: Plant promising lateral connections (immune systems, evolution) into future inquiries
3. **Mutation**: Deliberately invert assumptions ("what if we can't cooperate?") to test robustness

**Pruning Criteria**: Remove inquiry paths that (a) repeat previous conclusions, (b) cannot connect to other paths, or (c) require information we cannot obtain

**Meta-Insights**: The inquiry process itself mirrors the climate challenge - individual stages optimize locally while requiring coordination for global insight""",
    },
]

PATTERN_RECOGNITION_EXAMPLES: list[dict[str, str]] = [
    {
        "input": "Topic: Future of education",
        "output": """**Identified Motifs**:
- Personalization vs standardization (appears in scope, logic, lateral stages)
- Technology as amplifier of existing dynamics (found in mythos, lateral)
- Tension between credentialing and learning (scope, logic)
- Social dimension of learning often undervalued (mythos, lateral)
- Measurement shapes behavior (logic, recursive)

**Meta-Patterns**:
1. **The Compression Pattern**: Education is trying to compress more knowledge into less time, but compression has fundamental limits (like data compression)

2. **The Coherence-Flexibility Tradeoff**: More personalized education means less shared knowledge base, creating cultural fragmentation

3. **The Tool Paradox**: Tools that make learning easier may reduce the struggle that creates deep understanding

**Cross-Modal Connections**:
- Logic saw personalization as optimization problem
- Mythos saw it as loss of shared narrative
- Lateral connected it to biological adaptation (specialization vs resilience)

**Predictive Implications**: Education will likely bifurcate into (1) standardized, credential-focused tracks and (2) personalized, mastery-focused alternatives - with increasing tension between them

**Unified Theory**: Education is fundamentally about creating capable, adapted individuals who share enough common ground to coordinate. The challenge is maintaining both adaptability AND coherence as knowledge expands and society diversifies.""",
    },
]

# =============================================================================
# Combined Example Bank
# =============================================================================

STAGE_EXAMPLES: dict[str, list[dict[str, str]]] = {
    "scope_clarification": SCOPE_CLARIFICATION_EXAMPLES,
    "logical_branching": LOGICAL_BRANCHING_EXAMPLES,
    "intuitive_branching": INTUITIVE_BRANCHING_EXAMPLES,
    "lateral_exploration": LATERAL_EXPLORATION_EXAMPLES,
    "recursive_design": RECURSIVE_DESIGN_EXAMPLES,
    "pattern_recognition": PATTERN_RECOGNITION_EXAMPLES,
}

# =============================================================================
# General Task Examples
# =============================================================================

CODE_REVIEW_EXAMPLES: list[dict[str, str]] = [
    {
        "input": "Review this function for issues:\ndef process(data): return [x*2 for x in data if x > 0]",
        "output": """**Issues Found**:
1. **Missing type hints**: Function lacks parameter and return type annotations
2. **No input validation**: Will fail on non-iterable input
3. **Silent filtering**: Negative values are silently discarded without logging
4. **Generic name**: 'process' doesn't describe what transformation occurs

**Suggested Improvement**:
```python
def double_positive_values(data: list[int | float]) -> list[int | float]:
    \"\"\"Double all positive values in the input list.\"\"\"
    if not isinstance(data, (list, tuple)):
        raise TypeError(f"Expected list/tuple, got {type(data)}")
    return [x * 2 for x in data if x > 0]
```""",
    },
]

ANALYSIS_EXAMPLES: list[dict[str, str]] = [
    {
        "input": "Analyze the trade-offs between microservices and monolithic architectures",
        "output": """**Microservices Advantages**:
- Independent deployment and scaling per service
- Technology diversity (right tool per domain)
- Team autonomy and clear ownership boundaries
- Fault isolation prevents cascading failures

**Microservices Disadvantages**:
- Distributed system complexity (networking, consistency)
- Operational overhead (monitoring, deployment, debugging)
- Cross-service transactions are difficult
- Initial development slower due to infrastructure needs

**Monolith Advantages**:
- Simpler development, deployment, debugging
- Transaction consistency is straightforward
- Lower operational overhead
- Better for small teams and early-stage products

**Decision Framework**:
- Start monolithic unless you have proven scale requirements
- Split to microservices when: team coordination costs exceed service boundaries, or specific components have vastly different scaling needs
- "Modular monolith" can capture many benefits of both""",
    },
]

# =============================================================================
# Example Bank Management
# =============================================================================


class ExampleBank:
    """
    Manages few-shot examples for prompt injection.

    Features:
    - Stage-specific examples for inquiry workflows
    - General task type examples
    - Quality-based example selection
    - Custom example registration
    """

    def __init__(self) -> None:
        """Initialize with default examples."""
        self._examples: dict[str, list[dict[str, str]]] = {}

        # Load default examples
        self._examples.update(STAGE_EXAMPLES)
        self._examples["code_review"] = CODE_REVIEW_EXAMPLES
        self._examples["analysis"] = ANALYSIS_EXAMPLES

    def get_examples(
        self,
        task_type: str,
        max_examples: int = 2,
    ) -> list[dict[str, str]]:
        """
        Get examples for a task type.

        Args:
            task_type: Type of task
            max_examples: Maximum number to return

        Returns:
            List of example dictionaries
        """
        examples = self._examples.get(task_type, [])
        return examples[:max_examples]

    def add_example(
        self,
        task_type: str,
        input_text: str,
        output_text: str,
    ) -> None:
        """
        Add a custom example.

        Args:
            task_type: Task type for the example
            input_text: Example input
            output_text: Example output
        """
        if task_type not in self._examples:
            self._examples[task_type] = []

        self._examples[task_type].append({
            "input": input_text,
            "output": output_text,
        })

    def list_task_types(self) -> list[str]:
        """List all available task types."""
        return list(self._examples.keys())

    def get_example_count(self, task_type: str) -> int:
        """Get number of examples for a task type."""
        return len(self._examples.get(task_type, []))

    def format_examples_for_prompt(
        self,
        task_type: str,
        max_examples: int = 2,
    ) -> str:
        """
        Format examples as XML for prompt injection.

        Args:
            task_type: Task type
            max_examples: Maximum examples to include

        Returns:
            Formatted XML string
        """
        examples = self.get_examples(task_type, max_examples)

        if not examples:
            return ""

        formatted = "<examples>\n"
        for i, example in enumerate(examples, 1):
            formatted += f"<example_{i}>\n"
            formatted += f"<input>\n{example['input']}\n</input>\n"
            formatted += f"<output>\n{example['output']}\n</output>\n"
            formatted += f"</example_{i}>\n"
        formatted += "</examples>"

        return formatted


# Default instance
_default_bank: ExampleBank | None = None


def get_example_bank() -> ExampleBank:
    """Get the default example bank instance."""
    global _default_bank
    if _default_bank is None:
        _default_bank = ExampleBank()
    return _default_bank
