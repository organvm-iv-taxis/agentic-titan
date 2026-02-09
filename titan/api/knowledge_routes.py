"""
Titan API - Knowledge Routes

Endpoints for knowledge graph exploration, search, and statistics.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Query

from titan.api.typing_helpers import BaseModel, Field, typed_get

logger = logging.getLogger("titan.api.knowledge")

router = APIRouter(prefix="/knowledge", tags=["knowledge"])


# =============================================================================
# Request/Response Models
# =============================================================================


class SearchResult(BaseModel):
    """A knowledge search result."""

    key: str = Field(..., description="The item key/identifier")
    content_preview: str = Field(..., description="Preview of the content")
    score: float = Field(..., description="Relevance score 0-1")
    source: str = Field(..., description="Source of the knowledge")
    tags: list[str] = Field(default_factory=list, description="Associated tags")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class SearchResponse(BaseModel):
    """Response from knowledge search."""

    query: str = Field(..., description="The search query")
    results: list[SearchResult] = Field(..., description="Search results")
    total_results: int = Field(..., description="Total matching results")
    processing_time_ms: int = Field(..., description="Search processing time")


class GraphNode(BaseModel):
    """A node in the knowledge graph."""

    id: str = Field(..., description="Node identifier")
    label: str = Field(..., description="Display label")
    type: str = Field(..., description="Node type (concept, entity, inquiry, etc.)")
    properties: dict[str, Any] = Field(default_factory=dict, description="Node properties")


class GraphEdge(BaseModel):
    """An edge in the knowledge graph."""

    source: str = Field(..., description="Source node ID")
    target: str = Field(..., description="Target node ID")
    relationship: str = Field(..., description="Relationship type")
    weight: float = Field(default=1.0, description="Edge weight")


class GraphResponse(BaseModel):
    """Response containing a knowledge subgraph."""

    nodes: list[GraphNode] = Field(..., description="Graph nodes")
    edges: list[GraphEdge] = Field(..., description="Graph edges")
    center_node: str | None = Field(None, description="The central node ID if graph is centered")
    depth: int = Field(..., description="Graph traversal depth")


class KnowledgeStatsResponse(BaseModel):
    """Knowledge base statistics."""

    total_entries: int = Field(..., description="Total knowledge entries")
    total_inquiries: int = Field(..., description="Total inquiry sessions stored")
    total_concepts: int = Field(..., description="Total concepts/entities")
    storage_backends: dict[str, bool] = Field(..., description="Backend availability")
    cache_stats: dict[str, Any] = Field(default_factory=dict, description="Cache statistics")
    last_updated: str = Field(..., description="Last update timestamp")


# =============================================================================
# Endpoints
# =============================================================================


@typed_get(router, "/search", response_model=SearchResponse)
async def search_knowledge(
    query: str = Query(..., min_length=1, description="Search query"),
    limit: int = Query(default=10, ge=1, le=100, description="Maximum results"),
    source: str | None = Query(default=None, description="Filter by source"),
    tag: str | None = Query(default=None, description="Filter by tag"),
) -> SearchResponse:
    """
    Search the knowledge base.

    Performs semantic search across stored knowledge including
    inquiry results, learned patterns, and indexed content.
    """
    import time

    start_time = time.time()

    try:
        from hive.memory import HiveMind

        hive = HiveMind()
        await hive.initialize()

        try:
            # Perform search
            raw_results = await hive.search(query, limit=limit * 2)  # Get extra for filtering

            # Filter and format results
            results: list[SearchResult] = []
            for r in raw_results:
                # Apply filters
                if source and r.get("source") != source:
                    continue
                if tag and tag not in r.get("tags", []):
                    continue

                if len(results) >= limit:
                    break

                results.append(
                    SearchResult(
                        key=r.get("key", ""),
                        content_preview=str(r.get("value", ""))[:200],
                        score=r.get("score", 0.0),
                        source=r.get("source", "unknown"),
                        tags=r.get("tags", []),
                        metadata=r.get("metadata", {}),
                    )
                )

            processing_time = int((time.time() - start_time) * 1000)

            return SearchResponse(
                query=query,
                results=results,
                total_results=len(raw_results),
                processing_time_ms=processing_time,
            )

        finally:
            await hive.shutdown()

    except ImportError:
        logger.warning("HiveMind not available")
        return SearchResponse(
            query=query,
            results=[],
            total_results=0,
            processing_time_ms=0,
        )
    except Exception as e:
        logger.error(f"Knowledge search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@typed_get(router, "/graph", response_model=GraphResponse)
async def get_knowledge_graph(
    center: str | None = Query(default=None, description="Center node ID"),
    depth: int = Query(default=2, ge=1, le=5, description="Traversal depth"),
    node_type: str | None = Query(default=None, description="Filter by node type"),
    limit: int = Query(default=50, ge=1, le=200, description="Maximum nodes"),
) -> GraphResponse:
    """
    Get a knowledge graph or subgraph.

    Returns nodes and edges representing knowledge relationships.
    Can be centered on a specific node or return a general overview.
    """
    try:
        nodes: list[GraphNode] = []
        edges: list[GraphEdge] = []

        # Try to get inquiry sessions as nodes
        try:
            from titan.workflows.inquiry_engine import get_inquiry_engine

            engine = get_inquiry_engine()
            sessions = engine.list_sessions()

            for session in sessions[:limit]:
                # Add session node
                nodes.append(
                    GraphNode(
                        id=session.id,
                        label=session.topic[:30],
                        type="inquiry",
                        properties={
                            "status": session.status.value,
                            "workflow": session.workflow.name,
                            "stages": session.total_stages,
                        },
                    )
                )

                # Add stage nodes and edges
                if depth >= 2:
                    for i, result in enumerate(session.results):
                        stage_id = f"{session.id}_stage_{i}"
                        nodes.append(
                            GraphNode(
                                id=stage_id,
                                label=result.stage_name,
                                type="stage",
                                properties={
                                    "role": result.role,
                                    "model": result.model_used,
                                },
                            )
                        )
                        edges.append(
                            GraphEdge(
                                source=session.id,
                                target=stage_id,
                                relationship="has_stage",
                            )
                        )

                        # Link sequential stages
                        if i > 0:
                            prev_stage_id = f"{session.id}_stage_{i - 1}"
                            edges.append(
                                GraphEdge(
                                    source=prev_stage_id,
                                    target=stage_id,
                                    relationship="followed_by",
                                )
                            )

        except Exception as e:
            logger.warning(f"Could not load inquiry sessions for graph: {e}")

        # Try to get topology information
        try:
            from hive.topology import get_topology_engine

            topo_engine = get_topology_engine()
            if hasattr(topo_engine, "_active_topology") and topo_engine._active_topology:
                topo = topo_engine._active_topology
                topo_type = topo.topology_type.value if hasattr(topo, "topology_type") else "custom"
                topo_node_id = f"topology_{topo_type}"

                nodes.append(
                    GraphNode(
                        id=topo_node_id,
                        label="Active Topology",
                        type="topology",
                        properties={
                            "type": (
                                topo.topology_type.value
                                if hasattr(topo, "topology_type")
                                else "unknown"
                            ),
                        },
                    )
                )

        except Exception as e:
            logger.debug(f"Could not load topology for graph: {e}")

        # Filter by type if specified
        if node_type:
            nodes = [n for n in nodes if n.type == node_type]
            node_ids = {n.id for n in nodes}
            edges = [e for e in edges if e.source in node_ids and e.target in node_ids]

        # Filter by center if specified
        if center:
            # Keep only nodes connected to center within depth
            connected = {center}
            for _ in range(depth):
                new_connected = set()
                for edge in edges:
                    if edge.source in connected:
                        new_connected.add(edge.target)
                    if edge.target in connected:
                        new_connected.add(edge.source)
                connected.update(new_connected)

            nodes = [n for n in nodes if n.id in connected]
            edges = [e for e in edges if e.source in connected and e.target in connected]

        return GraphResponse(
            nodes=nodes[:limit],
            edges=edges,
            center_node=center,
            depth=depth,
        )

    except Exception as e:
        logger.error(f"Knowledge graph retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@typed_get(router, "/stats", response_model=KnowledgeStatsResponse)
async def get_knowledge_stats() -> KnowledgeStatsResponse:
    """
    Get knowledge base statistics.

    Returns counts, storage backend status, and cache statistics.
    """
    from datetime import datetime

    try:
        stats = KnowledgeStatsResponse(
            total_entries=0,
            total_inquiries=0,
            total_concepts=0,
            storage_backends={},
            cache_stats={},
            last_updated=datetime.now().isoformat(),
        )

        # Check inquiry sessions
        try:
            from titan.workflows.inquiry_engine import get_inquiry_engine

            engine = get_inquiry_engine()
            sessions = engine.list_sessions()
            stats.total_inquiries = len(sessions)
            stats.total_entries += len(sessions)
        except Exception as e:
            logger.debug(f"Could not count inquiry sessions: {e}")

        # Check HiveMind backends
        try:
            from hive.memory import HiveMind

            hive = HiveMind()
            await hive.initialize()

            try:
                health = await hive.health_check()
                stats.storage_backends = {
                    "redis": health.get("redis", False),
                    "chromadb": health.get("chroma", False),
                }
            finally:
                await hive.shutdown()

        except Exception as e:
            logger.debug(f"Could not check HiveMind backends: {e}")
            stats.storage_backends = {"redis": False, "chromadb": False}

        # Check learning system
        try:
            from hive.learning import get_episodic_learner

            learner = get_episodic_learner()
            learning_stats = learner.get_statistics()
            stats.total_entries += learning_stats.get("total_episodes", 0)
            stats.cache_stats["episodes"] = learning_stats.get("total_episodes", 0)
        except Exception as e:
            logger.debug(f"Could not get learning stats: {e}")

        return stats

    except Exception as e:
        logger.error(f"Knowledge stats retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
