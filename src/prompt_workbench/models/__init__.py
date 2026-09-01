"""Central typed data: use cases, runs, metrics, settings, and port contracts.

A dependency-free leaf — nothing here imports from ``core``, ``services`` or the
UI — so a caller can write ``from prompt_workbench.models import UseCase``
regardless of file layout. Behavioural classes live in ``core`` or ``services``.
"""

from prompt_workbench.models.chat import ChatMessage, ChatThread
from prompt_workbench.models.evaluation import (
    CaseEvaluation,
    EvaluationEvidence,
    EvaluationRun,
    Grade,
    MetricScore,
)
from prompt_workbench.models.ground_truth import CaseCategory, GroundTruthCase
from prompt_workbench.models.identifiers import IdFactory, random_id, sequential_ids
from prompt_workbench.models.metrics import MetricDefinition, MetricKind
from prompt_workbench.models.model_info import ModelInfo
from prompt_workbench.models.model_settings import ModelSettings
from prompt_workbench.models.runs import PromptRun
from prompt_workbench.models.use_case import MockBlock, UseCase

__all__ = [
    "CaseCategory",
    "CaseEvaluation",
    "ChatMessage",
    "ChatThread",
    "EvaluationEvidence",
    "EvaluationRun",
    "Grade",
    "GroundTruthCase",
    "IdFactory",
    "MetricDefinition",
    "MetricKind",
    "MetricScore",
    "MockBlock",
    "ModelInfo",
    "ModelSettings",
    "PromptRun",
    "UseCase",
    "random_id",
    "sequential_ids",
]
