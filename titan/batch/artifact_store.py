"""
Titan Batch - Artifact Store

Persistence layer for batch session artifacts.
Supports filesystem and S3/MinIO backends.
"""

from __future__ import annotations

import hashlib
import io
import logging
import os
import zipfile
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, cast
from uuid import UUID

from titan.batch.models import SessionArtifact

logger = logging.getLogger("titan.batch.artifact_store")


# =============================================================================
# Abstract Base
# =============================================================================

class ArtifactStore(ABC):
    """
    Abstract base for artifact storage.

    Provides interface for storing and retrieving session artifacts.
    """

    @abstractmethod
    async def save_artifact(
        self,
        batch_id: str,
        session_id: str,
        content: bytes,
        format: str = "markdown",
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """
        Save an artifact.

        Args:
            batch_id: Batch job ID
            session_id: Session ID
            content: Artifact content
            format: Content format (markdown, json, etc.)
            metadata: Optional metadata

        Returns:
            Artifact URI
        """
        pass

    @abstractmethod
    async def get_artifact(self, artifact_uri: str) -> bytes:
        """
        Retrieve an artifact.

        Args:
            artifact_uri: Artifact URI

        Returns:
            Artifact content
        """
        pass

    @abstractmethod
    async def delete_artifact(self, artifact_uri: str) -> bool:
        """
        Delete an artifact.

        Args:
            artifact_uri: Artifact URI

        Returns:
            True if deleted
        """
        pass

    @abstractmethod
    async def list_artifacts(
        self,
        batch_id: str,
    ) -> list[SessionArtifact]:
        """
        List all artifacts for a batch.

        Args:
            batch_id: Batch job ID

        Returns:
            List of artifact metadata
        """
        pass

    @abstractmethod
    async def export_batch_archive(
        self,
        batch_id: str,
        format: str = "zip",
    ) -> bytes:
        """
        Export all batch artifacts as archive.

        Args:
            batch_id: Batch job ID
            format: Archive format (zip)

        Returns:
            Archive content
        """
        pass

    async def artifact_exists(self, artifact_uri: str) -> bool:
        """Check if artifact exists."""
        try:
            await self.get_artifact(artifact_uri)
            return True
        except (FileNotFoundError, Exception):
            return False


# =============================================================================
# Filesystem Store
# =============================================================================

class FilesystemArtifactStore(ArtifactStore):
    """
    Filesystem-based artifact storage.

    Stores artifacts in a local directory structure:
    {base_path}/{batch_id}/{session_id}.{format}
    """

    def __init__(
        self,
        base_path: str | Path | None = None,
    ) -> None:
        """
        Initialize filesystem store.

        Args:
            base_path: Base directory for artifacts
        """
        if base_path is None:
            base_path = os.getenv(
                "TITAN_ARTIFACT_PATH",
                os.path.expanduser("~/.titan/artifacts"),
            )
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Filesystem artifact store: {self.base_path}")

    def _get_artifact_path(
        self,
        batch_id: str,
        session_id: str,
        format: str,
    ) -> Path:
        """Get full path for an artifact."""
        batch_dir = self.base_path / batch_id
        batch_dir.mkdir(parents=True, exist_ok=True)
        extension = self._format_to_extension(format)
        return batch_dir / f"{session_id}.{extension}"

    def _format_to_extension(self, format: str) -> str:
        """Map format to file extension."""
        mapping = {
            "markdown": "md",
            "json": "json",
            "yaml": "yaml",
            "text": "txt",
            "html": "html",
        }
        return mapping.get(format, format)

    def _uri_to_path(self, uri: str) -> Path:
        """Convert URI to filesystem path."""
        if uri.startswith("file://"):
            return Path(uri[7:])
        return self.base_path / uri

    async def save_artifact(
        self,
        batch_id: str,
        session_id: str,
        content: bytes,
        format: str = "markdown",
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Save artifact to filesystem."""
        path = self._get_artifact_path(batch_id, session_id, format)

        # Write content
        path.write_bytes(content)

        # Write metadata if provided
        if metadata:
            import json
            meta_path = path.with_suffix(f"{path.suffix}.meta.json")
            meta_path.write_text(json.dumps(metadata, indent=2))

        uri = f"file://{path}"
        logger.debug(f"Saved artifact: {uri} ({len(content)} bytes)")
        return uri

    async def get_artifact(self, artifact_uri: str) -> bytes:
        """Retrieve artifact from filesystem."""
        path = self._uri_to_path(artifact_uri)

        if not path.exists():
            raise FileNotFoundError(f"Artifact not found: {artifact_uri}")

        return path.read_bytes()

    async def delete_artifact(self, artifact_uri: str) -> bool:
        """Delete artifact from filesystem."""
        path = self._uri_to_path(artifact_uri)

        if path.exists():
            path.unlink()
            # Also delete metadata file if exists
            meta_path = path.with_suffix(f"{path.suffix}.meta.json")
            if meta_path.exists():
                meta_path.unlink()
            logger.debug(f"Deleted artifact: {artifact_uri}")
            return True
        return False

    async def list_artifacts(
        self,
        batch_id: str,
    ) -> list[SessionArtifact]:
        """List all artifacts for a batch."""
        import json

        batch_dir = self.base_path / batch_id
        if not batch_dir.exists():
            return []

        artifacts = []
        for path in batch_dir.iterdir():
            if path.suffix == ".json" and ".meta" in path.name:
                continue  # Skip metadata files

            # Load metadata if exists
            meta_path = path.with_suffix(f"{path.suffix}.meta.json")
            metadata = {}
            if meta_path.exists():
                try:
                    metadata = json.loads(meta_path.read_text())
                except Exception:
                    pass

            # Calculate checksum
            content = path.read_bytes()
            checksum = hashlib.sha256(content).hexdigest()

            # Extract session ID from filename
            session_id = path.stem

            artifact = SessionArtifact(
                session_id=UUID(session_id) if self._is_uuid(session_id) else UUID(int=0),
                batch_id=UUID(batch_id) if self._is_uuid(batch_id) else UUID(int=0),
                topic=metadata.get("topic", ""),
                artifact_uri=f"file://{path}",
                format=self._extension_to_format(path.suffix),
                size_bytes=len(content),
                created_at=datetime.fromtimestamp(path.stat().st_mtime),
                checksum=checksum,
                metadata=metadata,
            )
            artifacts.append(artifact)

        return artifacts

    async def export_batch_archive(
        self,
        batch_id: str,
        format: str = "zip",
    ) -> bytes:
        """Export all batch artifacts as ZIP archive."""
        artifacts = await self.list_artifacts(batch_id)

        if format != "zip":
            raise ValueError(f"Unsupported archive format: {format}")

        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            for artifact in artifacts:
                content = await self.get_artifact(artifact.artifact_uri)
                filename = f"{artifact.topic[:50]}_{artifact.session_id}.md"
                # Sanitize filename
                filename = "".join(c for c in filename if c.isalnum() or c in "._- ")
                zf.writestr(filename, content)

        return buffer.getvalue()

    def _is_uuid(self, s: str) -> bool:
        """Check if string is a valid UUID."""
        try:
            UUID(s)
            return True
        except (ValueError, AttributeError):
            return False

    def _extension_to_format(self, extension: str) -> str:
        """Map file extension to format."""
        mapping = {
            ".md": "markdown",
            ".json": "json",
            ".yaml": "yaml",
            ".yml": "yaml",
            ".txt": "text",
            ".html": "html",
        }
        return mapping.get(extension.lower(), "text")


# =============================================================================
# S3/MinIO Store
# =============================================================================

class S3ArtifactStore(ArtifactStore):
    """
    S3/MinIO-based artifact storage.

    Stores artifacts in an S3 bucket with structure:
    s3://{bucket}/{prefix}/{batch_id}/{session_id}.{format}
    """

    def __init__(
        self,
        bucket: str | None = None,
        prefix: str = "artifacts",
        endpoint_url: str | None = None,
        access_key: str | None = None,
        secret_key: str | None = None,
        region: str = "us-east-1",
    ) -> None:
        """
        Initialize S3 store.

        Args:
            bucket: S3 bucket name
            prefix: Key prefix for artifacts
            endpoint_url: Custom endpoint (for MinIO)
            access_key: AWS access key
            secret_key: AWS secret key
            region: AWS region
        """
        self.bucket = bucket or os.getenv("TITAN_S3_BUCKET", "titan-artifacts")
        self.prefix = prefix
        self.endpoint_url = endpoint_url or os.getenv("TITAN_S3_ENDPOINT")
        self.region = region

        # Import boto3 lazily
        try:
            import boto3
            from botocore.config import Config

            config = Config(
                region_name=region,
                retries={"max_attempts": 3, "mode": "adaptive"},
            )

            self._client = boto3.client(
                "s3",
                endpoint_url=self.endpoint_url,
                aws_access_key_id=access_key or os.getenv("AWS_ACCESS_KEY_ID"),
                aws_secret_access_key=secret_key or os.getenv("AWS_SECRET_ACCESS_KEY"),
                config=config,
            )
            logger.info(f"S3 artifact store: s3://{self.bucket}/{self.prefix}")

        except ImportError:
            raise ImportError("boto3 is required for S3 artifact store")

    def _get_key(
        self,
        batch_id: str,
        session_id: str,
        format: str,
    ) -> str:
        """Get S3 key for an artifact."""
        extension = self._format_to_extension(format)
        return f"{self.prefix}/{batch_id}/{session_id}.{extension}"

    def _format_to_extension(self, format: str) -> str:
        """Map format to file extension."""
        mapping = {
            "markdown": "md",
            "json": "json",
            "yaml": "yaml",
            "text": "txt",
            "html": "html",
        }
        return mapping.get(format, format)

    def _uri_to_key(self, uri: str) -> str:
        """Convert URI to S3 key."""
        if uri.startswith("s3://"):
            parts = uri[5:].split("/", 1)
            if len(parts) == 2:
                return parts[1]
        return uri

    async def save_artifact(
        self,
        batch_id: str,
        session_id: str,
        content: bytes,
        format: str = "markdown",
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Save artifact to S3."""
        key = self._get_key(batch_id, session_id, format)

        # Prepare metadata
        s3_metadata = {
            "batch-id": batch_id,
            "session-id": session_id,
            "format": format,
        }
        if metadata:
            import json
            s3_metadata["custom"] = json.dumps(metadata)

        # Upload to S3
        self._client.put_object(
            Bucket=self.bucket,
            Key=key,
            Body=content,
            ContentType=self._format_to_content_type(format),
            Metadata=s3_metadata,
        )

        uri = f"s3://{self.bucket}/{key}"
        logger.debug(f"Saved artifact: {uri} ({len(content)} bytes)")
        return uri

    async def get_artifact(self, artifact_uri: str) -> bytes:
        """Retrieve artifact from S3."""
        key = self._uri_to_key(artifact_uri)

        try:
            response = self._client.get_object(Bucket=self.bucket, Key=key)
            return cast(bytes, response["Body"].read())
        except self._client.exceptions.NoSuchKey:
            raise FileNotFoundError(f"Artifact not found: {artifact_uri}")

    async def delete_artifact(self, artifact_uri: str) -> bool:
        """Delete artifact from S3."""
        key = self._uri_to_key(artifact_uri)

        try:
            self._client.delete_object(Bucket=self.bucket, Key=key)
            logger.debug(f"Deleted artifact: {artifact_uri}")
            return True
        except Exception as e:
            logger.warning(f"Failed to delete artifact: {e}")
            return False

    async def list_artifacts(
        self,
        batch_id: str,
    ) -> list[SessionArtifact]:
        """List all artifacts for a batch."""
        import json

        prefix = f"{self.prefix}/{batch_id}/"
        artifacts = []

        paginator = self._client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]

                # Get object metadata
                head = self._client.head_object(Bucket=self.bucket, Key=key)
                s3_meta = head.get("Metadata", {})

                # Parse custom metadata
                custom_meta = {}
                if "custom" in s3_meta:
                    try:
                        custom_meta = json.loads(s3_meta["custom"])
                    except Exception:
                        pass

                # Extract session ID from key
                filename = key.split("/")[-1]
                session_id = filename.rsplit(".", 1)[0]

                artifact = SessionArtifact(
                    session_id=UUID(session_id) if self._is_uuid(session_id) else UUID(int=0),
                    batch_id=UUID(batch_id) if self._is_uuid(batch_id) else UUID(int=0),
                    topic=custom_meta.get("topic", ""),
                    artifact_uri=f"s3://{self.bucket}/{key}",
                    format=s3_meta.get("format", "markdown"),
                    size_bytes=obj["Size"],
                    created_at=obj["LastModified"],
                    checksum=obj.get("ETag", "").strip('"'),
                    metadata=custom_meta,
                )
                artifacts.append(artifact)

        return artifacts

    async def export_batch_archive(
        self,
        batch_id: str,
        format: str = "zip",
    ) -> bytes:
        """Export all batch artifacts as ZIP archive."""
        artifacts = await self.list_artifacts(batch_id)

        if format != "zip":
            raise ValueError(f"Unsupported archive format: {format}")

        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            for artifact in artifacts:
                content = await self.get_artifact(artifact.artifact_uri)
                topic = artifact.topic or "unknown"
                filename = f"{topic[:50]}_{artifact.session_id}.md"
                # Sanitize filename
                filename = "".join(c for c in filename if c.isalnum() or c in "._- ")
                zf.writestr(filename, content)

        return buffer.getvalue()

    def _format_to_content_type(self, format: str) -> str:
        """Map format to content type."""
        mapping = {
            "markdown": "text/markdown",
            "json": "application/json",
            "yaml": "application/yaml",
            "text": "text/plain",
            "html": "text/html",
        }
        return mapping.get(format, "application/octet-stream")

    def _is_uuid(self, s: str) -> bool:
        """Check if string is a valid UUID."""
        try:
            UUID(s)
            return True
        except (ValueError, AttributeError):
            return False


# =============================================================================
# Factory Functions
# =============================================================================

_default_store: ArtifactStore | None = None


def get_artifact_store() -> ArtifactStore:
    """
    Get the default artifact store.

    Automatically selects based on environment configuration.
    """
    global _default_store

    if _default_store is None:
        # Check for S3 configuration
        if os.getenv("TITAN_S3_BUCKET") or os.getenv("TITAN_S3_ENDPOINT"):
            try:
                _default_store = S3ArtifactStore()
            except ImportError:
                logger.warning("boto3 not available, using filesystem store")
                _default_store = FilesystemArtifactStore()
        else:
            _default_store = FilesystemArtifactStore()

    return _default_store


def set_artifact_store(store: ArtifactStore) -> None:
    """Set the default artifact store."""
    global _default_store
    _default_store = store
