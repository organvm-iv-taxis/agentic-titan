"""
Titan Workflows - Visualization Generator

Generates visualization specifications from inquiry stage outputs.
Produces configurations for Chart.js, D3.js, and other libraries.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from titan.workflows.inquiry_engine import InquirySession

logger = logging.getLogger("titan.workflows.visualization")


class VisualizationType(StrEnum):
    """Types of visualizations."""

    RADAR = "radar"  # Radar/spider chart for multi-dimensional comparison
    BAR = "bar"  # Bar chart for categorical data
    SANKEY = "sankey"  # Sankey diagram for flow/connections
    FORCE = "force"  # Force-directed graph
    TREEMAP = "treemap"  # Hierarchical treemap
    TIMELINE = "timeline"  # Temporal timeline
    HEATMAP = "heatmap"  # Heat map for correlations
    SUNBURST = "sunburst"  # Hierarchical sunburst
    PIE = "pie"  # Pie/donut chart
    LINE = "line"  # Line chart for trends


class VisualizationLibrary(StrEnum):
    """Supported visualization libraries."""

    CHARTJS = "chartjs"
    D3 = "d3"
    PLOTLY = "plotly"
    MERMAID = "mermaid"


@dataclass
class VisualizationSpec:
    """
    Specification for a single visualization.

    Contains the type, library, configuration, and data needed
    to render a visualization.
    """

    viz_type: VisualizationType
    library: VisualizationLibrary
    title: str
    description: str
    config: dict[str, Any]  # Library-specific config
    data: dict[str, Any]  # Data for the visualization
    width: int = 600
    height: int = 400
    responsive: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "viz_type": self.viz_type.value,
            "library": self.library.value,
            "title": self.title,
            "description": self.description,
            "config": self.config,
            "data": self.data,
            "width": self.width,
            "height": self.height,
            "responsive": self.responsive,
            "metadata": self.metadata,
        }

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)

    def to_chartjs_config(self) -> dict[str, Any]:
        """Generate Chart.js-compatible configuration."""
        if self.library != VisualizationLibrary.CHARTJS:
            raise ValueError("Not a Chart.js visualization")

        return {
            "type": self._chartjs_type_map.get(self.viz_type, "bar"),
            "data": self.data,
            "options": {
                "responsive": self.responsive,
                "plugins": {
                    "title": {
                        "display": True,
                        "text": self.title,
                    },
                },
                **self.config,
            },
        }

    _chartjs_type_map = {
        VisualizationType.RADAR: "radar",
        VisualizationType.BAR: "bar",
        VisualizationType.PIE: "pie",
        VisualizationType.LINE: "line",
    }


@dataclass
class VisualizationSuite:
    """
    A collection of visualizations for an inquiry session.

    Contains multiple visualization specs that together provide
    visual insight into the inquiry results.
    """

    session_id: str
    topic: str
    visualizations: list[VisualizationSpec]
    recommended_layout: str = "grid"  # grid, tabs, scroll
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "session_id": self.session_id,
            "topic": self.topic,
            "visualizations": [v.to_dict() for v in self.visualizations],
            "recommended_layout": self.recommended_layout,
            "metadata": self.metadata,
        }

    def get_by_type(self, viz_type: VisualizationType) -> list[VisualizationSpec]:
        """Get visualizations by type."""
        return [v for v in self.visualizations if v.viz_type == viz_type]


class VisualizationGenerator:
    """
    Generates visualization specifications from inquiry results.

    Analyzes stage outputs to determine appropriate visualizations
    and generates library-specific configurations.
    """

    # Colors for Chart.js visualizations
    CHART_COLORS = [
        "rgba(54, 162, 235, 0.8)",  # Blue
        "rgba(255, 99, 132, 0.8)",  # Red
        "rgba(75, 192, 192, 0.8)",  # Teal
        "rgba(255, 206, 86, 0.8)",  # Yellow
        "rgba(153, 102, 255, 0.8)",  # Purple
        "rgba(255, 159, 64, 0.8)",  # Orange
    ]

    def __init__(self) -> None:
        """Initialize the visualization generator."""
        pass

    async def generate_suite(
        self,
        session: InquirySession,
    ) -> VisualizationSuite:
        """
        Generate a complete visualization suite for an inquiry session.

        Args:
            session: The inquiry session to visualize

        Returns:
            VisualizationSuite with appropriate visualizations
        """
        logger.info(f"Generating visualizations for session {session.id}")

        visualizations = []

        # Always generate stage progress radar
        if len(session.results) >= 2:
            radar = self._generate_stage_radar(session)
            visualizations.append(radar)

        # Generate cognitive style distribution
        style_chart = self._generate_style_distribution(session)
        visualizations.append(style_chart)

        # Generate stage timeline
        if len(session.results) >= 2:
            timeline = self._generate_timeline(session)
            visualizations.append(timeline)

        # Attempt to parse stage content for additional visualizations
        for result in session.results:
            content_viz = self._parse_content_for_viz(result.content, result.stage_name)
            if content_viz:
                visualizations.append(content_viz)

        # Generate knowledge graph if we have enough data
        if len(session.results) >= 3:
            graph = self._generate_concept_graph(session)
            visualizations.append(graph)

        return VisualizationSuite(
            session_id=session.id,
            topic=session.topic,
            visualizations=visualizations,
            recommended_layout="tabs" if len(visualizations) > 4 else "grid",
            metadata={
                "stage_count": len(session.results),
                "workflow": session.workflow.name,
            },
        )

    def _generate_stage_radar(self, session: InquirySession) -> VisualizationSpec:
        """Generate radar chart showing stage characteristics."""
        labels = [r.stage_name for r in session.results]

        # Generate scores based on content length and other heuristics
        datasets = [
            {
                "label": "Content Depth",
                "data": [min(1.0, len(r.content) / 2000) for r in session.results],
                "backgroundColor": "rgba(54, 162, 235, 0.2)",
                "borderColor": "rgba(54, 162, 235, 1)",
                "pointBackgroundColor": "rgba(54, 162, 235, 1)",
            }
        ]

        return VisualizationSpec(
            viz_type=VisualizationType.RADAR,
            library=VisualizationLibrary.CHARTJS,
            title="Stage Analysis Depth",
            description="Relative depth of analysis across inquiry stages",
            config={
                "elements": {
                    "line": {"borderWidth": 3},
                },
                "scales": {
                    "r": {
                        "angleLines": {"display": True},
                        "suggestedMin": 0,
                        "suggestedMax": 1,
                    },
                },
            },
            data={
                "labels": labels,
                "datasets": datasets,
            },
        )

    def _generate_style_distribution(self, session: InquirySession) -> VisualizationSpec:
        """Generate pie chart of cognitive styles used."""
        style_counts: dict[str, int] = {}
        for stage in session.workflow.stages:
            style = stage.cognitive_style.value
            style_counts[style] = style_counts.get(style, 0) + 1

        labels = list(style_counts.keys())
        values = list(style_counts.values())

        return VisualizationSpec(
            viz_type=VisualizationType.PIE,
            library=VisualizationLibrary.CHARTJS,
            title="Cognitive Style Distribution",
            description="Distribution of cognitive styles across inquiry stages",
            config={
                "plugins": {
                    "legend": {"position": "right"},
                },
            },
            data={
                "labels": labels,
                "datasets": [
                    {
                        "data": values,
                        "backgroundColor": self.CHART_COLORS[: len(labels)],
                    }
                ],
            },
        )

    def _generate_timeline(self, session: InquirySession) -> VisualizationSpec:
        """Generate timeline showing stage execution."""
        labels = [r.stage_name for r in session.results]
        durations = [r.duration_ms / 1000 for r in session.results]  # Convert to seconds

        return VisualizationSpec(
            viz_type=VisualizationType.BAR,
            library=VisualizationLibrary.CHARTJS,
            title="Stage Execution Timeline",
            description="Duration of each inquiry stage in seconds",
            config={
                "indexAxis": "y",
                "scales": {
                    "x": {
                        "title": {
                            "display": True,
                            "text": "Duration (seconds)",
                        },
                    },
                },
            },
            data={
                "labels": labels,
                "datasets": [
                    {
                        "label": "Duration (s)",
                        "data": durations,
                        "backgroundColor": self.CHART_COLORS[0],
                    }
                ],
            },
        )

    def _parse_content_for_viz(
        self,
        content: str,
        stage_name: str,
    ) -> VisualizationSpec | None:
        """
        Attempt to parse stage content for visualization data.

        Looks for structured data like lists, comparisons, or metrics.
        """
        # Look for numbered lists
        list_pattern = r"^\d+\.\s+(.+)$"
        list_items = re.findall(list_pattern, content, re.MULTILINE)

        if len(list_items) >= 3:
            return VisualizationSpec(
                viz_type=VisualizationType.BAR,
                library=VisualizationLibrary.CHARTJS,
                title=f"{stage_name}: Key Points",
                description=f"Key points extracted from {stage_name}",
                config={
                    "indexAxis": "y",
                },
                data={
                    "labels": [item[:50] for item in list_items[:8]],
                    "datasets": [
                        {
                            "label": "Relevance",
                            "data": list(range(len(list_items[:8]), 0, -1)),
                            "backgroundColor": self.CHART_COLORS[1],
                        }
                    ],
                },
                metadata={"extracted_from": stage_name},
            )

        # Look for percentage mentions
        percent_pattern = r"(\d+(?:\.\d+)?)\s*%"
        percentages = re.findall(percent_pattern, content)

        if len(percentages) >= 2:
            labels = [f"Item {i + 1}" for i in range(len(percentages[:6]))]
            values = [float(p) for p in percentages[:6]]

            return VisualizationSpec(
                viz_type=VisualizationType.PIE,
                library=VisualizationLibrary.CHARTJS,
                title=f"{stage_name}: Percentages",
                description=f"Percentage values found in {stage_name}",
                config={},
                data={
                    "labels": labels,
                    "datasets": [
                        {
                            "data": values,
                            "backgroundColor": self.CHART_COLORS[: len(values)],
                        }
                    ],
                },
                metadata={"extracted_from": stage_name},
            )

        return None

    def _generate_concept_graph(self, session: InquirySession) -> VisualizationSpec:
        """Generate D3 force-directed concept graph."""
        nodes = []
        links = []

        # Add topic as central node
        nodes.append(
            {
                "id": "topic",
                "label": session.topic[:30],
                "group": "topic",
                "size": 30,
            }
        )

        # Add stages as nodes
        for i, result in enumerate(session.results):
            node_id = f"stage_{i}"
            nodes.append(
                {
                    "id": node_id,
                    "label": result.stage_name,
                    "group": "stage",
                    "size": 20,
                }
            )

            # Link to topic
            links.append(
                {
                    "source": "topic",
                    "target": node_id,
                    "value": 2,
                }
            )

            # Link to previous stage
            if i > 0:
                links.append(
                    {
                        "source": f"stage_{i - 1}",
                        "target": node_id,
                        "value": 1,
                    }
                )

        return VisualizationSpec(
            viz_type=VisualizationType.FORCE,
            library=VisualizationLibrary.D3,
            title="Inquiry Concept Graph",
            description="Force-directed graph showing relationships between inquiry stages",
            config={
                "linkDistance": 100,
                "charge": -300,
                "gravity": 0.1,
            },
            data={
                "nodes": nodes,
                "links": links,
            },
        )

    def parse_visualization_output(
        self,
        content: str,
        source_stage: str,
    ) -> list[VisualizationSpec]:
        """
        Parse stage output for embedded visualization specifications.

        Looks for structured data patterns that can be visualized.

        Args:
            content: Stage output content
            source_stage: Name of the source stage

        Returns:
            List of extracted VisualizationSpecs
        """
        specs = []

        # Try to extract structured data
        viz = self._parse_content_for_viz(content, source_stage)
        if viz:
            specs.append(viz)

        # Look for JSON blocks that might be visualization data
        json_pattern = r"```json\s*(.*?)\s*```"
        json_blocks = re.findall(json_pattern, content, re.DOTALL)

        for block in json_blocks:
            try:
                data = json.loads(block)
                if isinstance(data, dict) and "labels" in data:
                    specs.append(
                        VisualizationSpec(
                            viz_type=VisualizationType.BAR,
                            library=VisualizationLibrary.CHARTJS,
                            title=f"Data from {source_stage}",
                            description="Embedded visualization data",
                            config={},
                            data={
                                "labels": data.get("labels", []),
                                "datasets": [
                                    {
                                        "label": "Values",
                                        "data": data.get("values", data.get("data", [])),
                                        "backgroundColor": self.CHART_COLORS[0],
                                    }
                                ],
                            },
                            metadata={"extracted_from": source_stage, "source": "json_block"},
                        )
                    )
            except json.JSONDecodeError:
                continue

        return specs


# =============================================================================
# Factory Functions
# =============================================================================

_generator: VisualizationGenerator | None = None


def get_visualization_generator() -> VisualizationGenerator:
    """Get the visualization generator singleton."""
    global _generator
    if _generator is None:
        _generator = VisualizationGenerator()
    return _generator


async def generate_visualizations(
    session: InquirySession,
) -> VisualizationSuite:
    """
    Convenience function to generate visualizations for a session.

    Args:
        session: Inquiry session to visualize

    Returns:
        VisualizationSuite
    """
    generator = get_visualization_generator()
    return await generator.generate_suite(session)
