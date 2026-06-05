from typing import List, Callable, Optional, Dict, Any
from .models import AgentStep
from .graph import StateTracker


class LoopDetectedError(Exception):
    """Exception raised when an agentic execution loop is detected."""

    def __init__(self, cycle: List[int], repeat_count: int, message: str):
        super().__init__(message)
        self.cycle = cycle
        self.repeat_count = repeat_count


class LoopGuard:
    """The developer interface class coordinating LoopGuard's telemetry."""

    def __init__(
        self,
        similarity_threshold: float = 0.75,
        min_repeats: int = 3,
        max_cycle_length: int = 5,
        embedding_fn: Optional[Callable[[str], List[float]]] = None,
        on_loop_detected: Optional[Callable[[List[int], int], None]] = None,
    ):
        self.tracker = StateTracker(
            similarity_threshold=similarity_threshold, embedding_fn=embedding_fn
        )
        self.min_repeats = min_repeats
        self.max_cycle_length = max_cycle_length
        self.on_loop_detected = on_loop_detected
        self.steps_history: List[AgentStep] = []

    def add_step(
        self,
        action: str,
        input_text: str,
        observation: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Record an execution step.
        Returns True if safe, raises LoopDetectedError if a loop is detected.
        """
        step = AgentStep(
            action=action,
            input_text=input_text,
            observation=observation,
            metadata=metadata or {},
        )
        self.steps_history.append(step)
        self.tracker.add_step(step)

        cycle, length = self.tracker.check_cycles(
            min_length=1, max_length=self.max_cycle_length, min_repeats=self.min_repeats
        )

        if cycle:
            message = (
                f"Loop detected: State cycle {cycle} (length {length}) "
                f"repeated {self.min_repeats} times."
            )
            if self.on_loop_detected:
                self.on_loop_detected(cycle, self.min_repeats)
            raise LoopDetectedError(cycle, self.min_repeats, message)

        return True

    def get_healing_prompt(self) -> str:
        """Get a system instructions prompt to guide the LLM out of its execution loop."""
        if not self.steps_history:
            return ""

        last_step = self.steps_history[-1]
        return (
            "\n[SYSTEM NOTICE]: You have repeated the action '{action}' with inputs/results "
            "similar to '{input_val}' multiple times. This indicates you are stuck in an execution cycle. "
            "Do not retry this action again with similar parameters. Try a completely different reasoning path "
            "or check for alternative tools to achieve your goal."
        ).format(action=last_step.action, input_val=last_step.input_text[:120])
