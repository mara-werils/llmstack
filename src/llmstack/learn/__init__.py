"""Adaptive Learning Pipeline — self-improving AI that learns from your corrections.

The missing flywheel for local AI: automatically collects feedback,
generates training data, triggers incremental fine-tuning, and tracks
quality improvements over time. Your model gets measurably better
the more you use it — entirely offline.
"""

from llmstack.learn.feedback import Feedback, FeedbackType
from llmstack.learn.store import FeedbackStore
from llmstack.learn.dataset import DatasetGenerator
from llmstack.learn.scheduler import TrainScheduler
from llmstack.learn.versions import ModelVersionManager

__all__ = [
    "Feedback",
    "FeedbackType",
    "FeedbackStore",
    "DatasetGenerator",
    "TrainScheduler",
    "ModelVersionManager",
]
