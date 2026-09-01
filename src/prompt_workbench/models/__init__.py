"""Central typed data: artifacts, snapshots, settings, and port contracts.

The re-exported types below form a dependency-free leaf — they import nothing
from ``core``/``services``/``app`` — so callers can write
``from prompt_workbench.models import ModelSettings`` regardless of file layout.
Behavioural classes live in ``core`` or ``services``, not here.
"""

from prompt_workbench.models.brief import BRIEF_FIELDS, BriefSnapshot, PromptBrief
from prompt_workbench.models.candidates import CandidatePrompt, PromptTechnique
from prompt_workbench.models.chat import ChatMessage, ChatThread, ThreadConfig
from prompt_workbench.models.evaluation import (
    CaseEvaluation,
    EvaluationRun,
    Grade,
    MetricScore,
)
from prompt_workbench.models.execution import ExecutionRecord
from prompt_workbench.models.ground_truth import (
    CaseCategory,
    GroundTruthCase,
    GroundTruthDataset,
)
from prompt_workbench.models.identifiers import IdFactory, random_id, sequential_ids
from prompt_workbench.models.metrics import MetricDefinition, MetricKind
from prompt_workbench.models.model_info import ModelInfo
from prompt_workbench.models.model_settings import ModelSettings
from prompt_workbench.models.provenance import SourceRef

__all__ = [
    "BRIEF_FIELDS",
    "BriefSnapshot",
    "CandidatePrompt",
    "CaseCategory",
    "CaseEvaluation",
    "ChatMessage",
    "ChatThread",
    "EvaluationRun",
    "ExecutionRecord",
    "Grade",
    "GroundTruthCase",
    "GroundTruthDataset",
    "IdFactory",
    "MetricDefinition",
    "MetricKind",
    "MetricScore",
    "ModelInfo",
    "ModelSettings",
    "PromptBrief",
    "PromptTechnique",
    "SourceRef",
    "ThreadConfig",
    "random_id",
    "sequential_ids",
]
