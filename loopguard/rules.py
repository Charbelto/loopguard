import re
import time
from abc import ABC, abstractmethod
from typing import List, Optional, Any
from .models import AgentStep


class BaseRule(ABC):
    """Abstract base class for all telemetry constraint rules in LoopGuard."""

    @property
    @abstractmethod
    def name(self) -> str:
        """The identifier name of the rule."""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """A user-friendly description of the rule constraints."""
        pass

    @abstractmethod
    def check(
        self, steps: List[AgentStep], state_sequence: List[int], tracker: Any
    ) -> bool:
        """
        Evaluate history to detect rule violations.

        Args:
            steps: The list of agent execution steps.
            state_sequence: The sequence of semantic state IDs.
            tracker: The StateTracker instance tracking the clustering.

        Returns:
            True if the rule is violated, False if the execution is safe.
        """
        pass


class CycleDetectionRule(BaseRule):
    """Detects repeating cycles of state transitions (identical semantic patterns)."""

    def __init__(self, min_length: int = 1, max_length: int = 5, min_repeats: int = 3):
        self.min_length = min_length
        self.max_length = max_length
        self.min_repeats = min_repeats
        self.violated_cycle: Optional[List[int]] = None

    @property
    def name(self) -> str:
        return "CycleDetection"

    @property
    def description(self) -> str:
        return (
            f"Detects repeating cycles of state transitions (length {self.min_length}-{self.max_length}, "
            f"repeating {self.min_repeats} times)"
        )

    def check(
        self, steps: List[AgentStep], state_sequence: List[int], tracker: Any
    ) -> bool:
        cycle, _ = tracker.check_cycles(
            min_length=self.min_length,
            max_length=self.max_length,
            min_repeats=self.min_repeats,
        )
        if cycle:
            self.violated_cycle = cycle
            return True
        return False


class MaxConsecutiveSimilarityRule(BaseRule):
    """Flags when consecutive execution steps are semantically identical/similar."""

    def __init__(self, action: Optional[str] = None, max_consecutive: int = 3):
        self.action = action
        self.max_consecutive = max_consecutive

    @property
    def name(self) -> str:
        return "MaxConsecutiveSimilarity"

    @property
    def description(self) -> str:
        action_desc = f"for action '{self.action}' " if self.action else ""
        return f"Flags when the same semantic state repeats consecutively {self.max_consecutive} times {action_desc}"

    def check(
        self, steps: List[AgentStep], state_sequence: List[int], tracker: Any
    ) -> bool:
        if len(state_sequence) < self.max_consecutive:
            return False

        last_states = state_sequence[-self.max_consecutive :]
        if len(set(last_states)) != 1:
            return False

        if self.action:
            last_steps = steps[-self.max_consecutive :]
            if any(step.action != self.action for step in last_steps):
                return False

        return True


class ActionOscillationRule(BaseRule):
    """Detects repeating oscillations between action names (e.g. ping-ponging tools)."""

    def __init__(self, max_oscillations: int = 3, window_size: int = 6):
        self.max_oscillations = max_oscillations
        self.window_size = window_size

    @property
    def name(self) -> str:
        return "ActionOscillation"

    @property
    def description(self) -> str:
        return f"Detects repeating oscillations between action names within the last {self.window_size} steps"

    def check(
        self, steps: List[AgentStep], state_sequence: List[int], tracker: Any
    ) -> bool:
        if len(steps) < self.window_size:
            return False

        actions = [step.action for step in steps[-self.window_size :]]
        # We look for pattern lengths of 2 (A-B-A-B...) or 3 (A-B-C-A-B-C...)
        for pattern_len in (2, 3):
            if pattern_len * self.max_oscillations > len(actions):
                continue
            sub = actions[-pattern_len:]
            is_oscillating = True
            for r in range(1, self.max_oscillations):
                start = -pattern_len * (r + 1)
                end = -pattern_len * r
                chunk = actions[start:end]
                if chunk != sub:
                    is_oscillating = False
                    break
            if is_oscillating:
                return True
        return False


class StateFrequencyRule(BaseRule):
    """Flags when any single semantic state occurs more than N times in a window."""

    def __init__(self, max_frequency: int = 3, window_size: int = 10):
        self.max_frequency = max_frequency
        self.window_size = window_size

    @property
    def name(self) -> str:
        return "StateFrequency"

    @property
    def description(self) -> str:
        return (
            f"Flags when any single semantic state occurs more than {self.max_frequency} times "
            f"within a window of {self.window_size} steps"
        )

    def check(
        self, steps: List[AgentStep], state_sequence: List[int], tracker: Any
    ) -> bool:
        window = state_sequence[-self.window_size :]
        from collections import Counter

        counts = Counter(window)
        return bool(counts and max(counts.values()) > self.max_frequency)


class ResourceLimitRule(BaseRule):
    """Enforces upper limits on execution steps or duration."""

    def __init__(
        self,
        max_steps: Optional[int] = None,
        max_duration_seconds: Optional[float] = None,
    ):
        self.max_steps = max_steps
        self.max_duration_seconds = max_duration_seconds
        self.start_time = time.time()

    @property
    def name(self) -> str:
        return "ResourceLimit"

    @property
    def description(self) -> str:
        limits = []
        if self.max_steps:
            limits.append(f"{self.max_steps} steps")
        if self.max_duration_seconds:
            limits.append(f"{self.max_duration_seconds}s execution time")
        return f"Limits execution resources to: {', '.join(limits)}"

    def check(
        self, steps: List[AgentStep], state_sequence: List[int], tracker: Any
    ) -> bool:
        if self.max_steps and len(steps) >= self.max_steps:
            return True
        if (
            self.max_duration_seconds
            and (time.time() - self.start_time) >= self.max_duration_seconds
        ):
            return True
        return False


class RegexGuardRule(BaseRule):
    """Validates inputs, observations, or actions against a regex pattern."""

    def __init__(self, pattern: str, target: str = "observation", block: bool = True):
        self.pattern = re.compile(pattern)
        self.target = target
        self.block = block

    @property
    def name(self) -> str:
        return "RegexGuard"

    @property
    def description(self) -> str:
        rule_type = "blocklist" if self.block else "allowlist"
        return f"Validates {self.target} against regex '{self.pattern.pattern}' ({rule_type})"

    def check(
        self, steps: List[AgentStep], state_sequence: List[int], tracker: Any
    ) -> bool:
        if not steps:
            return False
        last_step = steps[-1]
        val = getattr(last_step, self.target, None)
        if val is None:
            val = ""
        matches = bool(self.pattern.search(str(val)))
        return matches if self.block else not matches


class JSONGuardRule(BaseRule):
    """Ensures inputs or observations contain valid JSON parsing format."""

    def __init__(self, target: str = "observation"):
        self.target = target

    @property
    def name(self) -> str:
        return "JSONGuard"

    @property
    def description(self) -> str:
        return f"Ensures the value of the {self.target} field is well-formed JSON"

    def check(
        self, steps: List[AgentStep], state_sequence: List[int], tracker: Any
    ) -> bool:
        if not steps:
            return False
        last_step = steps[-1]
        val = getattr(last_step, self.target, None)
        if val is None:
            return False
        import json

        try:
            json.loads(str(val))
            return False
        except json.JSONDecodeError:
            return True
