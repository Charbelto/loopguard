"""LoopGuard: A semantic and structural loop detector for AI agents."""

from .models import AgentStep, State
from .core import LoopGuard, LoopDetectedError
from .rules import (
    BaseRule,
    CycleDetectionRule,
    MaxConsecutiveSimilarityRule,
    ActionOscillationRule,
    StateFrequencyRule,
    ResourceLimitRule,
    RegexGuardRule,
    JSONGuardRule,
)
from .healing import HealingStrategy
from .telemetry import TelemetryVisualizer

__all__ = [
    "AgentStep",
    "State",
    "LoopGuard",
    "LoopDetectedError",
    "BaseRule",
    "CycleDetectionRule",
    "MaxConsecutiveSimilarityRule",
    "ActionOscillationRule",
    "StateFrequencyRule",
    "ResourceLimitRule",
    "RegexGuardRule",
    "JSONGuardRule",
    "HealingStrategy",
    "TelemetryVisualizer",
]
