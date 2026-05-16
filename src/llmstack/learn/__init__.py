"""Adaptive Learning Pipeline — self-improving AI that learns from your corrections.

The missing flywheel for local AI: automatically collects feedback,
generates training data, triggers incremental fine-tuning, and tracks
quality improvements over time. Your model gets measurably better
the more you use it — entirely offline.

Components:
- feedback: Feedback signal types and data structures
- store: SQLite persistence for all learning data
- dataset: Automatic training data generation from feedback
- scheduler: Training trigger conditions and orchestration
- versions: Model version management with rollback
- regression: Quality regression detection with auto-rollback
- optimizer: Prompt optimization from feedback patterns
- preferences: User preference learning (length, format, tone)
- patterns: Code pattern/convention learning
- evaluator: Model quality evaluation against corrections
- collector: Simple feedback collection API for commands
- pipeline: Unified orchestrator tying all components together
- dpo_trainer: Direct Preference Optimization training
- synthetic: Data augmentation for sparse feedback
- context_memory: Learns optimal context selection for RAG
- drift: Query and feedback distribution drift detection
- active: Active learning for intelligent feedback requests
"""

from llmstack.learn.feedback import Feedback, FeedbackType
from llmstack.learn.store import FeedbackStore
from llmstack.learn.dataset import DatasetGenerator, DatasetStrategy
from llmstack.learn.scheduler import TrainScheduler
from llmstack.learn.versions import ModelVersionManager
from llmstack.learn.pipeline import LearningPipeline, get_pipeline
from llmstack.learn.collector import FeedbackCollector

__all__ = [
    "Feedback",
    "FeedbackType",
    "FeedbackStore",
    "DatasetGenerator",
    "DatasetStrategy",
    "TrainScheduler",
    "ModelVersionManager",
    "LearningPipeline",
    "get_pipeline",
    "FeedbackCollector",
]
