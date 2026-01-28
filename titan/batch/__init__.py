"""
Titan Batch - Batch Research Pipeline

Enables users to submit multiple research prompts and receive
multiple completed artifacts with distributed worker execution.

Features:
- Batch job submission and management
- Distributed Celery workers for parallel processing
- Load-aware scheduling with automatic offloading
- S3/MinIO artifact persistence
- Cross-session synthesis
- Real-time progress streaming via WebSocket

Example usage:

    from titan.batch import (
        BatchOrchestrator,
        BatchSubmitRequest,
        get_batch_orchestrator,
    )

    # Get orchestrator
    orchestrator = get_batch_orchestrator()

    # Submit batch
    request = BatchSubmitRequest(
        topics=["AI safety", "Prompt engineering", "Agent architectures"],
        workflow="expansive",
        budget_limit_usd=10.0,
        max_concurrent=3,
    )
    batch = await orchestrator.submit_batch(request)

    # Start processing
    await orchestrator.start_batch(batch.id)

    # Monitor progress
    async for event in orchestrator.stream_progress(batch.id):
        print(event)

    # Export artifacts
    from titan.batch import get_artifact_store
    store = get_artifact_store()
    archive = await store.export_batch_archive(str(batch.id))
"""

from titan.batch.models import (
    BatchJob,
    BatchProgress,
    BatchStatus,
    BatchSubmitRequest,
    QueuedSession,
    SessionArtifact,
    SessionQueueStatus,
)
from titan.batch.orchestrator import (
    BatchOrchestrator,
    get_batch_orchestrator,
    set_batch_orchestrator,
)
from titan.batch.artifact_store import (
    ArtifactStore,
    FilesystemArtifactStore,
    S3ArtifactStore,
    get_artifact_store,
    set_artifact_store,
)
from titan.batch.scheduler import (
    BatchScheduler,
    LoadLevel,
    SchedulingDecision,
    SchedulingStrategy,
    SystemLoad,
    WorkerStatus,
    get_batch_scheduler,
    get_system_load,
)
from titan.batch.synthesizer import (
    BatchSynthesizer,
    SynthesisResult,
    get_batch_synthesizer,
)

__all__ = [
    # Models
    "BatchJob",
    "BatchProgress",
    "BatchStatus",
    "BatchSubmitRequest",
    "QueuedSession",
    "SessionArtifact",
    "SessionQueueStatus",
    # Orchestrator
    "BatchOrchestrator",
    "get_batch_orchestrator",
    "set_batch_orchestrator",
    # Artifact Store
    "ArtifactStore",
    "FilesystemArtifactStore",
    "S3ArtifactStore",
    "get_artifact_store",
    "set_artifact_store",
    # Scheduler
    "BatchScheduler",
    "LoadLevel",
    "SchedulingDecision",
    "SchedulingStrategy",
    "SystemLoad",
    "WorkerStatus",
    "get_batch_scheduler",
    "get_system_load",
    # Synthesizer
    "BatchSynthesizer",
    "SynthesisResult",
    "get_batch_synthesizer",
]
