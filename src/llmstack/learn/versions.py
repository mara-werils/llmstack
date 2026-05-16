"""Model version management — track, compare, and rollback model versions.

Maintains a versioned history of fine-tuned models with quality metrics,
enabling automatic rollback if a new version regresses.
"""

from __future__ import annotations

import json
import logging
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from llmstack.learn.store import FeedbackStore

logger = logging.getLogger(__name__)

VERSIONS_DIR = Path.home() / ".llmstack" / "model_versions"


@dataclass
class ModelVersion:
    """A versioned model snapshot."""

    version: str
    base_model: str
    adapter_path: str = ""
    timestamp: float = field(default_factory=time.time)
    quality_score: float = 0.0
    is_active: bool = False
    train_run_id: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def display_name(self) -> str:
        ts = time.strftime("%Y-%m-%d %H:%M", time.localtime(self.timestamp))
        status = " [active]" if self.is_active else ""
        return f"v{self.version} ({ts}) quality={self.quality_score:.3f}{status}"


class ModelVersionManager:
    """Manages model versions with quality tracking and rollback.

    Maintains a registry of all fine-tuned model versions,
    tracks quality metrics over time, and supports automatic
    rollback when regression is detected.
    """

    def __init__(self, store: FeedbackStore, versions_dir: Path | None = None):
        self.store = store
        self.versions_dir = versions_dir or VERSIONS_DIR
        self.versions_dir.mkdir(parents=True, exist_ok=True)

    def create_version(
        self,
        base_model: str,
        adapter_path: str,
        train_run_id: int = 0,
        quality_score: float = 0.0,
        activate: bool = True,
        metadata: dict | None = None,
    ) -> ModelVersion:
        """Create and register a new model version."""
        version = self._next_version()

        # Copy adapter to versioned directory
        version_dir = self.versions_dir / version
        version_dir.mkdir(parents=True, exist_ok=True)

        if adapter_path and Path(adapter_path).exists():
            dest = version_dir / "adapter"
            if Path(adapter_path).is_dir():
                shutil.copytree(adapter_path, str(dest), dirs_exist_ok=True)
            else:
                shutil.copy2(adapter_path, str(dest))
            stored_path = str(dest)
        else:
            stored_path = adapter_path

        # Save version metadata
        meta = metadata or {}
        meta_path = version_dir / "version.json"
        meta_path.write_text(json.dumps({
            "version": version,
            "base_model": base_model,
            "adapter_path": stored_path,
            "train_run_id": train_run_id,
            "quality_score": quality_score,
            "timestamp": time.time(),
            "metadata": meta,
        }, indent=2))

        # Register in store
        self.store.add_model_version(
            version=version,
            base_model=base_model,
            adapter_path=stored_path,
            train_run_id=train_run_id,
            quality_score=quality_score,
            is_active=activate,
            metadata=meta,
        )

        mv = ModelVersion(
            version=version,
            base_model=base_model,
            adapter_path=stored_path,
            quality_score=quality_score,
            is_active=activate,
            train_run_id=train_run_id,
            metadata=meta,
        )

        logger.info("Created model version %s (quality=%.3f)", version, quality_score)
        return mv

    def get_active(self) -> ModelVersion | None:
        """Get the currently active model version."""
        data = self.store.get_active_version()
        if not data:
            return None
        return ModelVersion(
            version=data["version"],
            base_model=data["base_model"],
            adapter_path=data.get("adapter_path", ""),
            timestamp=data["timestamp"],
            quality_score=data.get("quality_score", 0.0),
            is_active=True,
            train_run_id=data.get("train_run_id", 0),
            metadata=json.loads(data.get("metadata", "{}")),
        )

    def activate(self, version: str) -> bool:
        """Activate a specific model version."""
        versions = self.store.get_versions()
        found = None
        for v in versions:
            if v["version"] == version:
                found = v
                break

        if not found:
            logger.error("Version %s not found", version)
            return False

        self.store.add_model_version(
            version=found["version"],
            base_model=found["base_model"],
            adapter_path=found.get("adapter_path", ""),
            train_run_id=found.get("train_run_id", 0),
            quality_score=found.get("quality_score", 0.0),
            is_active=True,
            metadata=json.loads(found.get("metadata", "{}")),
        )
        logger.info("Activated model version %s", version)
        return True

    def rollback(self) -> ModelVersion | None:
        """Rollback to the previous model version.

        Returns the newly activated version, or None if no previous version.
        """
        versions = self.store.get_versions(limit=10)
        if len(versions) < 2:
            logger.warning("No previous version to rollback to")
            return None

        # Find the first non-active version
        current = None
        previous = None
        for v in versions:
            if v.get("is_active"):
                current = v
            elif current and not previous:
                previous = v

        if not previous:
            # Just use the second one
            previous = versions[1]

        self.activate(previous["version"])
        logger.info(
            "Rolled back from %s to %s",
            current["version"] if current else "unknown",
            previous["version"],
        )

        return ModelVersion(
            version=previous["version"],
            base_model=previous["base_model"],
            adapter_path=previous.get("adapter_path", ""),
            timestamp=previous["timestamp"],
            quality_score=previous.get("quality_score", 0.0),
            is_active=True,
        )

    def list_versions(self, limit: int = 20) -> list[ModelVersion]:
        """List all model versions."""
        rows = self.store.get_versions(limit=limit)
        return [
            ModelVersion(
                version=r["version"],
                base_model=r["base_model"],
                adapter_path=r.get("adapter_path", ""),
                timestamp=r["timestamp"],
                quality_score=r.get("quality_score", 0.0),
                is_active=bool(r.get("is_active")),
                train_run_id=r.get("train_run_id", 0),
                metadata=json.loads(r.get("metadata", "{}")),
            )
            for r in rows
        ]

    def compare(self, version_a: str, version_b: str) -> dict[str, Any]:
        """Compare quality metrics between two versions."""
        metrics_a = self.store.get_quality_trend(version_a, "overall", limit=1)
        metrics_b = self.store.get_quality_trend(version_b, "overall", limit=1)

        score_a = metrics_a[0]["value"] if metrics_a else 0.0
        score_b = metrics_b[0]["value"] if metrics_b else 0.0

        return {
            "version_a": version_a,
            "version_b": version_b,
            "score_a": score_a,
            "score_b": score_b,
            "improvement": score_b - score_a,
            "better": version_b if score_b > score_a else version_a,
        }

    def _next_version(self) -> str:
        """Generate the next version number."""
        versions = self.store.get_versions(limit=1)
        if not versions:
            return "1"
        try:
            latest = int(versions[0]["version"])
            return str(latest + 1)
        except (ValueError, KeyError):
            return str(int(time.time()))
